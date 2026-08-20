import argparse
import itertools
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rampnet.model import KeypointModel
from rampnet.loading import load_checkpoint

# Learning-rate defaults per preset: training from ImageNet initialization
# uses the paper's 1e-5; fine-tuning released/earlier RampNet weights wants a
# much smaller step so it refines rather than forgets.
PRESET_LR = {'scratch': 1e-5, 'finetune': 3e-6}

#: Learning-rate schedules. 'constant' is the paper recipe and the default, so every
#: existing invocation is byte-for-byte unaffected; 'cosine' is the #135 rung.
LR_SCHEDULES = ('constant', 'cosine')


def lr_at_step(step, total_steps, peak_lr, schedule='constant', final_frac=0.0):
    """The learning rate for training step ``step``, from the step index alone.

    **Deliberately stateless, and that is the whole point.** Stage 2 runs on klone's
    preemptible ``ckpt-all`` partition -- Run A was requeued five times -- and resumes from
    ``latest_checkpoint.pth``. A stateful scheduler (``CosineAnnealingLR`` and friends
    keep ``last_epoch`` internally) would restart its decay from the peak on every
    requeue unless its state were also saved and restored, turning a smooth cosine
    into a sawtooth. That failure is silent: the run completes, the loss curve looks
    plausible, and the schedule under test was never actually applied.

    Computing the rate from ``global_step`` -- which is already checkpointed, already
    broadcast to every rank, and already correct across resume -- makes that
    impossible by construction rather than by remembering to serialize one more field.

    ``final_frac`` is the fraction of ``peak_lr`` the cosine lands on at the end
    (0.0 = decay to zero). There is **no warmup**: adding one would be a second
    change, and the rung exists to isolate the decay.
    """
    if schedule == 'constant':
        return peak_lr
    if schedule != 'cosine':
        raise ValueError(f"unknown lr schedule {schedule!r}")
    progress = min(max(step / max(1, total_steps), 0.0), 1.0)
    floor = peak_lr * final_frac
    return floor + (peak_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ResumeSkipSampler(Sampler):
    """Wraps a sampler so a resumed epoch can drop the batches it already did.

    **Why this exists.** Resuming used to fast-forward with ``if i < batch_idx_in_epoch:
    continue`` inside the training loop, which pulls each skipped batch all the way
    through the DataLoader -- reading and decoding a 2048x4096 panorama per skipped
    step -- only to throw it away. Measured on the #135 cosine rung: **4.6 min from job
    start to the first training step on a fresh start, against 23-30 min when resuming**,
    growing with position in the epoch.

    On a preemptible partition that is not a slow resume, it is a livelock. Once the
    resume cost exceeds the interval between preemptions, every incarnation spends its
    whole slice re-reading panoramas, reaches no checkpoint, and the run advances by
    nothing -- which is exactly what happened for 8h54m on 2026-08-19/20 at step 9,000.

    Dropping the *indices* instead means the workers never fetch them, so the cost falls
    to essentially zero. **The batches that remain, and their order, are identical**:
    ``DistributedSampler``'s permutation is a pure function of (seed, epoch), so taking
    an islice off the front leaves the rest of the sequence untouched. This is a speed
    fix, not a change to what the model sees.

    ``skip`` is set per epoch by the training loop; ``epoch_length`` is the full,
    unskipped per-rank count, because the LR horizon must not move when a run resumes.
    """

    def __init__(self, base):
        self.base = base
        self.skip = 0
        self.epoch_length = len(base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        indices = iter(self.base)
        return itertools.islice(indices, self.skip, None) if self.skip else indices

    def __len__(self):
        return max(0, self.epoch_length - self.skip)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the stage-2 panorama curb ramp detector.")
    parser.add_argument('--data-root', default='../dataset',
                        help="Dataset root containing train/ and val/ splits (default: ../dataset)")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of training epochs (default: 1, as in the paper)")
    parser.add_argument('--lr', type=float, default=None,
                        help="Learning rate; overrides the --preset default")
    parser.add_argument('--preset', choices=sorted(PRESET_LR), default='scratch',
                        help="'scratch' trains from ImageNet weights (lr 1e-5); "
                             "'finetune' warm-starts from --init-weights (lr 3e-6)")
    parser.add_argument('--init-weights', default=None,
                        help="Checkpoint to warm-start from (e.g. the released RampNet weights). "
                             "Ignored when latest_checkpoint.pth exists: resuming an interrupted "
                             "run always takes precedence, and warm-starting applies at step 0 only.")
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help="Directory for per-epoch checkpoints (default: checkpoints)")
    parser.add_argument('--lr-schedule', choices=LR_SCHEDULES, default='constant',
                        help="'constant' is the paper recipe and the default -- every "
                             "existing invocation is unaffected. 'cosine' decays --lr to "
                             "--lr-final-frac of itself over the whole run, with no "
                             "warmup (#135 rung).")
    parser.add_argument('--lr-final-frac', type=float, default=0.0,
                        help="Fraction of --lr the cosine ends at (default 0.0).")
    parser.add_argument('--checkpoint-interval-steps', type=int, default=1000,
                        help="Steps between writes of latest_checkpoint.pth (default 1000, "
                             "the paper recipe). This is the granularity a preemption "
                             "rewinds to, so on a preemptible partition it must be well "
                             "under the interval between preemptions or nothing is banked.")
    args = parser.parse_args()
    if not 0.0 <= args.lr_final_frac < 1.0:
        parser.error("--lr-final-frac must be in [0, 1)")
    if args.checkpoint_interval_steps < 1:
        parser.error("--checkpoint-interval-steps must be >= 1")
    if args.preset == 'finetune' and args.init_weights is None:
        parser.error("--preset finetune requires --init-weights")
    if args.lr is None:
        args.lr = PRESET_LR[args.preset]
    return args


args = parse_args()


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

rank, local_rank, world_size = setup_distributed()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

new_root_dir = args.data_root

def generate_heatmap_from_points(points_normalized, heatmap_shape=(512, 1024), sigma=10.0):
    heatmap_h, heatmap_w = heatmap_shape
    heatmap = np.zeros(heatmap_shape, dtype=np.float32)
    
    if not points_normalized:
        return torch.from_numpy(heatmap).unsqueeze(0)

    sigma_sq = 2 * sigma * sigma
    radius = int(math.ceil(3 * sigma))

    for p_norm in points_normalized:
        if not (isinstance(p_norm, (list, tuple)) and len(p_norm) == 2):
            if rank == 0: print(f"Warning: Skipping invalid point format in generate_heatmap: {p_norm}")
            continue
        
        center_x_float = p_norm[0] * heatmap_w
        center_y_float = p_norm[1] * heatmap_h

        cx = int(round(center_x_float))
        cy = int(round(center_y_float))

        cx = max(0, min(cx, heatmap_w - 1))
        cy = max(0, min(cy, heatmap_h - 1))

        x_min = max(0, cx - radius)
        x_max = min(heatmap_w, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(heatmap_h, cy + radius + 1)

        for y_coord in range(y_min, y_max):
            for x_coord in range(x_min, x_max):
                dist_sq = float((x_coord - cx)**2 + (y_coord - cy)**2)
                gaussian_val = math.exp(-dist_sq / sigma_sq)
                heatmap[y_coord, x_coord] = max(heatmap[y_coord, x_coord], gaussian_val)
    
    return torch.from_numpy(heatmap).unsqueeze(0)

class EquiHeatmapDataset(Dataset):
    def __init__(self, root_dir, split, target_heatmap_shape=(512, 1024),
                 transform_input=None, points_to_heatmap_transform_fn=None,
                 apply_horizontal_flip=True):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.target_heatmap_shape = target_heatmap_shape
        
        self.transform_input = transform_input
        self.points_to_heatmap_transform_fn = points_to_heatmap_transform_fn
        self.apply_horizontal_flip = apply_horizontal_flip

        self.image_paths = []
        self.json_paths = []

        if not os.path.isdir(self.split_dir):
            if rank == 0: print(f"Error: Split directory not found: {self.split_dir}")
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        all_files_in_split = sorted(os.listdir(self.split_dir))
        
        for filename in all_files_in_split:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name, _ = os.path.splitext(filename)
                
                json_filename = base_name + '.json'
                json_full_path = os.path.join(self.split_dir, json_filename)

                if os.path.exists(json_full_path):
                    self.image_paths.append(os.path.join(self.split_dir, filename))
                    self.json_paths.append(json_full_path)
        
        if rank == 0:
            print(f"Initialized EquiHeatmapDataset for split '{self.split}' with {len(self.image_paths)} samples. Horizontal flip: {self.apply_horizontal_flip}")
            if len(self.image_paths) == 0:
                print(f"Warning: No image/JSON pairs found in {self.split_dir}. Check dataset structure and paths.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            if rank == 0: print(f"Warning: Index {idx} out of bounds for dataset size {len(self.image_paths)}.")
            dummy_img_size = (2048, 4096) 
            dummy_img = torch.zeros((3, dummy_img_size[0], dummy_img_size[1])) 
            dummy_heatmap = torch.zeros((1, self.target_heatmap_shape[0], self.target_heatmap_shape[1]))
            return dummy_img, dummy_heatmap

        input_path = self.image_paths[idx]
        json_path = self.json_paths[idx]

        try:
            image = Image.open(input_path).convert('RGB')
        except Exception as e:
            if rank == 0: print(f"Error loading image {input_path}: {e}. Returning dummy.")
            dummy_img_size = (2048, 4096) 
            dummy_img = torch.zeros((3, dummy_img_size[0], dummy_img_size[1])) 
            if self.transform_input:
                try:
                    
                    pil_dummy = Image.new('RGB', (dummy_img_size[1], dummy_img_size[0])) 
                    dummy_img = self.transform_input(pil_dummy)
                except: pass
            dummy_heatmap = torch.zeros((1, self.target_heatmap_shape[0], self.target_heatmap_shape[1]))
            return dummy_img, dummy_heatmap

        points_normalized = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            points_normalized_raw = data.get("curb_ramp_points_normalized", [])
            
            
            points_normalized = [p for p in points_normalized_raw 
                                 if isinstance(p, (list, tuple)) and len(p) == 2 and
                                 isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))]
            
            if len(points_normalized) != len(points_normalized_raw) and rank == 0:
                 print(f"Warning: Malformed or incomplete points found in {json_path}. Raw: {points_normalized_raw}, Filtered: {points_normalized}")

        except Exception as e:
            if rank == 0: print(f"Error loading or parsing JSON {json_path}: {e}. Using empty points list.")
            points_normalized = []

        if self.apply_horizontal_flip and random.random() < 0.5:
            image = F.hflip(image)
            flipped_points = []
            for x_norm, y_norm in points_normalized:
                flipped_points.append((1.0 - x_norm, y_norm))
            points_normalized = flipped_points
        
        if self.transform_input:
            image = self.transform_input(image)

        if self.points_to_heatmap_transform_fn:
            heatmap = self.points_to_heatmap_transform_fn(points_normalized, 
                                                          heatmap_shape=self.target_heatmap_shape)
        else:
            heatmap = torch.zeros((1, self.target_heatmap_shape[0], self.target_heatmap_shape[1]))
            if rank == 0 and len(points_normalized) > 0 :
                 print(f"Warning: points_to_heatmap_transform_fn not provided, but points found for {json_path}. Generating zero heatmap.")
        
        return image, heatmap

heatmap_output_shape = (512, 1024) 

input_transform = transforms.Compose([
    transforms.Resize((2048, 4096)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

points_to_heatmap_fn = generate_heatmap_from_points 

train_dataset = EquiHeatmapDataset(
    root_dir=new_root_dir,
    split='train',
    target_heatmap_shape=heatmap_output_shape,
    transform_input=input_transform,
    points_to_heatmap_transform_fn=points_to_heatmap_fn,
    apply_horizontal_flip=True
)

val_dataset = EquiHeatmapDataset(
    root_dir=new_root_dir,
    split='val',
    target_heatmap_shape=heatmap_output_shape,
    transform_input=input_transform,
    points_to_heatmap_transform_fn=points_to_heatmap_fn,
    apply_horizontal_flip=False
)

if len(train_dataset) == 0:
    if rank == 0: print("Error: Training dataset is empty. Exiting.")
    cleanup_distributed()
    exit()
if len(val_dataset) == 0 and rank == 0:
    print("Warning: Validation dataset is empty.")

train_sampler = ResumeSkipSampler(
    DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True))
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if len(val_dataset) > 0 else None

train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=4, pin_memory=True) if val_sampler else None

model = KeypointModel(heatmap_size=heatmap_output_shape, pretrained_backbone=True).cuda(local_rank)
if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

num_epochs = args.epochs
# The cosine's horizon. epoch_length is per-rank and drop_last=True, so this is the
# same 9,378 steps per epoch the paper run and Run A took at world size 16 -- the decay
# is defined over the whole run, not per epoch. Read off the sampler's FULL length
# rather than len(train_loader), which shrinks on the epoch a resume lands in: the
# horizon a resumed run decays over has to be the one it started with.
total_train_steps = num_epochs * train_sampler.epoch_length

if rank == 0:
    os.makedirs("peek_training", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='runs/experiment_1')
    print(f"Preset: {args.preset}, lr: {args.lr}, epochs: {args.epochs}, data root: {new_root_dir}")
    print(f"LR schedule: {args.lr_schedule}"
          + (f" -> {args.lr * args.lr_final_frac:.3g} over {total_train_steps} steps"
             if args.lr_schedule != 'constant' else " (no decay, as in the paper)"))

else:
    writer = None

checkpoint_interval_steps = args.checkpoint_interval_steps
start_epoch = 0
global_step = 0
batch_idx_in_epoch = 0
best_val_loss = float('inf')
checkpoint_file = "latest_checkpoint.pth"

if args.init_weights:
    if os.path.exists(checkpoint_file):
        # A resume file means this run was interrupted mid-training; restoring
        # it (weights + optimizer + step) must win, otherwise a stale
        # latest_checkpoint.pth would silently defeat the warm start.
        if rank == 0:
            print(f"Ignoring --init-weights {args.init_weights}: resume checkpoint "
                  f"{checkpoint_file} exists and takes precedence.")
    else:
        model_to_init = model.module if isinstance(model, DDP) else model
        load_checkpoint(model_to_init, args.init_weights, map_location='cpu')
        if rank == 0:
            print(f"Warm-started model weights from {args.init_weights}")

if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    batch_idx_in_epoch = checkpoint.get('batch_idx_in_epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    if rank == 0:
        print(f"Resumed training from epoch {start_epoch}, global_step {global_step}, batch_idx {batch_idx_in_epoch}")

if world_size > 1:
    states = torch.tensor([start_epoch, global_step, batch_idx_in_epoch, best_val_loss], dtype=torch.float64).cuda(local_rank)
    dist.broadcast(states, src=0)
    start_epoch = int(states[0].item())
    global_step = int(states[1].item())
    batch_idx_in_epoch = int(states[2].item())
    best_val_loss = states[3].item()
    dist.barrier()

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    return img_tensor.mul(std).add(mean)


for epoch in range(start_epoch, num_epochs):
    if world_size > 1:
        train_sampler.set_epoch(epoch)
    
    model.train()
    
    
    
    # Batches this epoch already did before the interruption. The sampler drops their
    # INDICES, so the DataLoader never reads them back off disk -- see ResumeSkipSampler.
    # Every rank has the same value (it is broadcast above), so DDP stays in lockstep.
    resume_offset = batch_idx_in_epoch if epoch == start_epoch else 0
    train_sampler.skip = resume_offset

    if rank == 0:
        progress_bar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}",
                            total=train_sampler.epoch_length,
                            initial=resume_offset)


    for i, (images, target_heatmaps) in enumerate(train_loader):
        # `i` counts batches actually processed this epoch; the skipped ones are gone
        # from the sampler, so the position within the epoch is resume_offset + i.
        #
        # Absolute, 0-based index of the step about to be taken. Reconstructed from
        # the checkpointed global_step, so it is correct on the first launch and after
        # any number of requeues -- see lr_at_step for why the schedule reads this
        # rather than keeping state of its own.
        iter_in_epoch = i
        step_index = global_step + iter_in_epoch
        current_lr = lr_at_step(step_index, total_train_steps, args.lr,
                                args.lr_schedule, args.lr_final_frac)
        if args.lr_schedule != 'constant':
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        images = images.cuda(local_rank, non_blocking=True)
        target_heatmaps = target_heatmaps.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, target_heatmaps)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        
        
        
        
        current_total_step = step_index + 1

        if rank == 0:
            writer.add_scalar('Loss/train_step', loss.item(), current_total_step)
            # Logged every step so the schedule that was ACTUALLY applied is
            # recoverable from the committed events afterwards. A sawtooth from a
            # mis-resumed scheduler is invisible in the loss curve and obvious here.
            writer.add_scalar('LR', current_lr, current_total_step)
            progress_bar.set_postfix(loss=loss.item(), step=current_total_step)
            progress_bar.update(1) 

            if current_total_step % checkpoint_interval_steps == 0:
                model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    'epoch': epoch, 
                    'global_step': current_total_step,
                    'batch_idx_in_epoch': resume_offset + i + 1,
                    'model_state_dict': model_state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_file)
                if rank == 0: print(f"Saved latest checkpoint at step {current_total_step}")
    
    if rank == 0 and isinstance(progress_bar, tqdm): 
        progress_bar.close()

    
    
    # len(train_loader) already excludes the skipped batches (ResumeSkipSampler.__len__),
    # so it IS the count processed this epoch -- no second subtraction.
    num_batches_processed_this_epoch = len(train_loader)
    global_step += num_batches_processed_this_epoch
    batch_idx_in_epoch = 0
    train_sampler.skip = 0

    
    if val_loader is not None:
        model.eval()
        val_loss_sum = 0.0
        num_val_samples = 0
        val_pbar_desc = f"Epoch {epoch+1} Validating"
        val_pbar = tqdm(val_loader, desc=val_pbar_desc, disable=(rank != 0)) if rank == 0 else val_loader

        with torch.no_grad():
            for images_val, target_heatmaps_val in val_pbar:
                images_val = images_val.cuda(local_rank, non_blocking=True)
                target_heatmaps_val = target_heatmaps_val.cuda(local_rank, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs_val = model(images_val)
                    loss_val = criterion(outputs_val, target_heatmaps_val)
                
                val_loss_sum += loss_val.item() * images_val.size(0)
                num_val_samples += images_val.size(0)

        if world_size > 1:
            val_loss_tensor = torch.tensor([val_loss_sum, num_val_samples], dtype=torch.float64).cuda(local_rank)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            total_val_loss_sum = val_loss_tensor[0].item()
            total_num_val_samples = val_loss_tensor[1].item()
        else:
            total_val_loss_sum = val_loss_sum
            total_num_val_samples = num_val_samples
            
        avg_val_loss = total_val_loss_sum / total_num_val_samples if total_num_val_samples > 0 else 0

        if rank == 0:
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Loss/val_epoch', avg_val_loss, global_step)

            epoch_checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}_step_{global_step}.pth")
            model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'batch_idx_in_epoch': 0,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'current_val_loss': avg_val_loss
            }, epoch_checkpoint_path)
            print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving best_model.pth")
                torch.save(model_state_to_save, "best_model.pth")

            if len(val_dataset) > 0 and len(val_dataset.image_paths) > 0 :
                rand_idx = random.randint(0, len(val_dataset) - 1)
                image_vis, gt_heatmap_vis = val_dataset[rand_idx] 
                
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    
                    
                    pred_heatmap_tensor = model(image_vis.unsqueeze(0).cuda(local_rank))
                pred_heatmap_np = pred_heatmap_tensor.cpu().numpy()[0, 0]

                img_unnorm = unnormalize(image_vis).clamp(0, 1).cpu()
                img_pil = transforms.ToPILImage()(img_unnorm)

                heatmap_normalized = (pred_heatmap_np - pred_heatmap_np.min()) / (pred_heatmap_np.max() - pred_heatmap_np.min() + 1e-8)
                
                fig_w, fig_h = img_pil.width / 100, img_pil.height / 100 
                fig, ax = plt.subplots(figsize=(fig_w if fig_w > 0 else 1, fig_h if fig_h > 0 else 1) , dpi=100)
                ax.imshow(heatmap_normalized, cmap='jet', interpolation='nearest', aspect='auto')
                ax.axis('off')
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                pil_heatmap_overlay = Image.open(buf).convert("RGBA")
                
                pil_heatmap_overlay = pil_heatmap_overlay.resize((img_pil.width, img_pil.height), Image.Resampling.NEAREST)
                
                alpha_data = np.array(pil_heatmap_overlay.convert("L")) 
                alpha_data = (alpha_data > 20) * 90 
                new_alpha = Image.fromarray(alpha_data.astype(np.uint8), mode='L')
                pil_heatmap_overlay.putalpha(new_alpha)
                
                img_pil.paste(pil_heatmap_overlay, (0, 0), pil_heatmap_overlay)

                visualization_path = f"peek_training/epoch_{epoch+1}_step_{global_step}.jpg"
                img_pil.save(visualization_path)
                print(f"Saved visualization to {visualization_path}")
            
            model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch + 1, 
                'global_step': global_step,
                'batch_idx_in_epoch': 0,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_file)
            if rank == 0: print(f"Saved latest checkpoint at end of epoch {epoch+1}")


if rank == 0 and writer is not None:
    writer.close()

cleanup_distributed()
print(f"Rank {rank} finished.")