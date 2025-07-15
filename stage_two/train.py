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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import timm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

new_root_dir = '../dataset'

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

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if len(val_dataset) > 0 else None

train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=4, pin_memory=True) if val_sampler else None

class KeypointModel(nn.Module):
    def __init__(self, heatmap_size=(512, 1024)):
        super(KeypointModel, self).__init__()
        backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = True
        
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        in_channels = backbone.num_features
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, image):
        features = self.feature_extractor(image)
        heatmap = self.head(features)
        return heatmap

model = KeypointModel(heatmap_size=heatmap_output_shape).cuda(local_rank)
if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler()

if rank == 0:
    os.makedirs("peek_training", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir='runs/experiment_1')
else:
    writer = None

num_epochs = 1
checkpoint_interval_steps = 1000
start_epoch = 0
global_step = 0
batch_idx_in_epoch = 0
best_val_loss = float('inf')
checkpoint_file = "latest_checkpoint.pth"

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
    
    
    
    if rank == 0:
        progress_bar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}", 
                            total=len(train_loader), 
                            initial=batch_idx_in_epoch if epoch == start_epoch else 0)
    
    
    for i, (images, target_heatmaps) in enumerate(train_loader):
        if epoch == start_epoch and i < batch_idx_in_epoch:
            continue 

        images = images.cuda(local_rank, non_blocking=True)
        target_heatmaps = target_heatmaps.cuda(local_rank, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, target_heatmaps)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        
        
        
        
        current_iter_in_epoch = i - (batch_idx_in_epoch if epoch == start_epoch else 0)
        current_total_step = global_step + current_iter_in_epoch + 1 

        if rank == 0:
            writer.add_scalar('Loss/train_step', loss.item(), current_total_step)
            progress_bar.set_postfix(loss=loss.item(), step=current_total_step)
            progress_bar.update(1) 

            if current_total_step % checkpoint_interval_steps == 0:
                model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    'epoch': epoch, 
                    'global_step': current_total_step,
                    'batch_idx_in_epoch': i + 1, 
                    'model_state_dict': model_state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_file)
                if rank == 0: print(f"Saved latest checkpoint at step {current_total_step}")
    
    if rank == 0 and isinstance(progress_bar, tqdm): 
        progress_bar.close()

    
    
    num_batches_processed_this_epoch = len(train_loader) - (batch_idx_in_epoch if epoch == start_epoch else 0)
    global_step += num_batches_processed_this_epoch
    batch_idx_in_epoch = 0 

    
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

            epoch_checkpoint_path = os.path.join("checkpoints", f"epoch_{epoch+1}_step_{global_step}.pth")
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