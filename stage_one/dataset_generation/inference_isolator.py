import cv2
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((1024, 352)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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
    
    def forward(self, x):
        features = self.feature_extractor(x)
        heatmap = self.head(features)
        return heatmap


model = KeypointModel(heatmap_size=(256, 88))
model.load_state_dict(torch.load("../crop_model/ps_and_manual_model/best_model.pth", map_location=device))
model.to(device)
model.eval()

def infer_image(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(img_rgb)
    
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    heatmap = output.squeeze(0).squeeze(0).cpu().numpy()
    return heatmap