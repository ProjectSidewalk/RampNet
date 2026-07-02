import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from rampnet.model import KeypointModel
from rampnet.loading import load_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((1024, 352)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


model = KeypointModel(heatmap_size=(256, 88))
load_checkpoint(model, "../crop_model/ps_and_manual_model/best_model.pth", map_location=device)
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