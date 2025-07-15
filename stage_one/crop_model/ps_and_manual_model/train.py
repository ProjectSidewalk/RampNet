import os
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm
import timm

torch.manual_seed(42)
random.seed(42)

IMAGE_TARGET_DIMS = (1024, 352) 

def generate_heatmap(keypoint, heatmap_size, image_target_dims, sigma=12):
    
    scale_x = heatmap_size[1] / image_target_dims[1] 
    scale_y = heatmap_size[0] / image_target_dims[0]
    
    x = keypoint[0] * scale_x
    y = keypoint[1] * scale_y
    
    height, width = heatmap_size
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    heatmap = np.exp(-((xv - x)**2 + (yv - y)**2) / (2 * sigma**2))
    return torch.tensor(heatmap, dtype=torch.float32)

class KeypointDataset(Dataset):
    def __init__(self, root_dir, transform=None, heatmap_size=(256, 88), 
                 image_target_dims=(1024, 352), sigma=12, 
                 apply_horizontal_flip=False, max_keypoints=20):
        self.root_dir = root_dir
        self.base_transform = transform
        self.heatmap_size = heatmap_size
        self.image_target_dims = image_target_dims
        self.sigma = sigma
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.apply_horizontal_flip = apply_horizontal_flip
        self.max_keypoints = max_keypoints
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.root_dir, file_name)
        image_pil = Image.open(img_path).convert('RGB')

        parsed_adj_keypoints = []
        parts = file_name.split('_-_')
        for part in parts[1:]:
            part = part.replace(".jpg", "")
            if "_" not in part:
                continue
            kp_str = part.split('_')
            orig_x = float(kp_str[0])
            orig_y = float(kp_str[1])
            adj_x = orig_x * 0.5
            adj_y = orig_y * 0.5
            parsed_adj_keypoints.append([adj_x, adj_y])

        if self.apply_horizontal_flip and random.random() < 0.5:
            image_pil = F.hflip(image_pil)
            img_target_width_for_kp_flip = self.image_target_dims[1]
            for kp in parsed_adj_keypoints:
                kp[0] = img_target_width_for_kp_flip - 1 - kp[0]
        
        image_tensor = image_pil
        if self.base_transform:
            image_tensor = self.base_transform(image_pil)


        target_heatmap_combined = torch.zeros(self.heatmap_size, dtype=torch.float32)
        
        if parsed_adj_keypoints:
            individual_heatmaps = []
            for adj_x, adj_y in parsed_adj_keypoints:
                heatmap_part = generate_heatmap((adj_x, adj_y), self.heatmap_size, 
                                                self.image_target_dims, self.sigma)
                individual_heatmaps.append(heatmap_part)
            
            if individual_heatmaps:
                stacked_heatmaps = torch.stack(individual_heatmaps, dim=0)
                target_heatmap_combined = torch.max(stacked_heatmaps, dim=0)[0]
        
        target_heatmap_combined = target_heatmap_combined.unsqueeze(0)

        output_keypoints_for_return = parsed_adj_keypoints[:] 
        while len(output_keypoints_for_return) < self.max_keypoints:
            output_keypoints_for_return.append([-1, -1])
        
        output_keypoints_for_return = output_keypoints_for_return[:self.max_keypoints]
        output_keypoints_tensor = torch.tensor(output_keypoints_for_return, dtype=torch.float32)
        
        return image_tensor, target_heatmap_combined, output_keypoints_tensor

transform = transforms.Compose([
    transforms.Resize(IMAGE_TARGET_DIMS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

HEATMAP_SIZE = (256, 88)

train_dataset = KeypointDataset(
    root_dir='dataset_1/train', 
    transform=transform, 
    heatmap_size=HEATMAP_SIZE,
    image_target_dims=IMAGE_TARGET_DIMS,
    sigma=12,
    apply_horizontal_flip=True
)
val_dataset = KeypointDataset(
    root_dir='dataset_1/val', 
    transform=transform, 
    heatmap_size=HEATMAP_SIZE,
    image_target_dims=IMAGE_TARGET_DIMS,
    sigma=12,
    apply_horizontal_flip=False
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

class KeypointModel(nn.Module):
    def __init__(self, heatmap_size=(256, 88)):
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

model = KeypointModel(heatmap_size=HEATMAP_SIZE)

checkpoint = torch.load('ps_model.pth', map_location='cpu')
model.load_state_dict(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("peek_training", exist_ok=True)

num_epochs = 100
best_val_loss = float('inf')

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, target_heatmaps, _ in progress_bar:
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, target_heatmaps)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item())
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, target_heatmaps, _ in val_loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)
            outputs = model(images)
            loss = criterion(outputs, target_heatmaps)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")
    
    model.eval()
    rand_idx = random.randint(0, len(val_dataset) - 1)
    image_tensor_vis, _, keypoints_for_vis = val_dataset[rand_idx] 
    
    with torch.no_grad():
        pred_heatmap = model(image_tensor_vis.unsqueeze(0).to(device)).cpu().numpy()[0, 0]
    
    img_vis = unnormalize(image_tensor_vis.cpu()).clamp(0, 1)
    img_pil = transforms.ToPILImage()(img_vis)

    draw = ImageDraw.Draw(img_pil)
    for kp_vis in keypoints_for_vis:
        if kp_vis[0] == -1 and kp_vis[1] == -1:
            continue
        x, y = kp_vis[0], kp_vis[1]
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(255, 0, 0))

    heatmap_normalized = (pred_heatmap - pred_heatmap.min()) / (pred_heatmap.max() - pred_heatmap.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(pred_heatmap.shape[1]/100, pred_heatmap.shape[0]/100))
    ax.imshow(heatmap_normalized, cmap='jet', interpolation='nearest')
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    heatmap_pil = Image.open(buf)
    heatmap_pil = heatmap_pil.convert("RGBA")

    heatmap_pil_resized = heatmap_pil.resize((img_pil.width, img_pil.height))
    
    r, g, b, a = heatmap_pil_resized.split()
    alpha_value = 90 
    new_alpha = Image.eval(a, lambda x: int(alpha_value * (x/255)))
    heatmap_pil_transparent = Image.merge('RGBA', (r, g, b, new_alpha))

    img_pil.paste(heatmap_pil_transparent, (0, 0), heatmap_pil_transparent)

    img_pil.save(f"peek_training/{epoch+1}.jpg")
    print(f"Saved visualization to peek_training/{epoch+1}.jpg")