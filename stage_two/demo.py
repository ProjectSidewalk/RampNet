import os
import torch
import gradio as gr
from PIL import Image, ImageOps, ImageDraw 
import numpy as np
from torchvision import transforms

import timm
import torch.nn as nn

from skimage.feature import peak_local_max 

class KeypointModel(nn.Module):
    def __init__(self, heatmap_size=(512, 1024)):
        super(KeypointModel, self).__init__()
        backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=False)

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

MODEL_CHECKPOINT_PATH = "checkpoints/epoch_1_step_9378.pth"
MODEL_INPUT_SIZE = (2048, 4096) 
MODEL_HEATMAP_SIZE = (512, 1024) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIO_PORT = 25566

def load_trained_model(checkpoint_path, heatmap_size):
    """Loads the KeypointModel and weights from a checkpoint."""
    print(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = KeypointModel(heatmap_size=heatmap_size)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    if all(key.startswith('module.') for key in state_dict.keys()):
        print("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model

preprocess_transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_and_visualize(input_image_pil):
    if input_image_pil is None:
        return None

    def _crop(image):
        img_array = np.array(image)

        if img_array.shape[2] == 4: 
            mask = img_array[:, :, 3] > 0 
            y_nonzero, x_nonzero = np.nonzero(mask)
        else: 

            if img_array.ndim == 3: 
                 non_black_pixels = np.any(img_array > 0, axis=2)
            else: 
                 non_black_pixels = img_array > 0
            y_nonzero, x_nonzero = np.nonzero(non_black_pixels)

        if y_nonzero.size > 0 and x_nonzero.size > 0:
            cropped_panorama_np = img_array[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]

            h_crop, w_crop = cropped_panorama_np.shape[:2]
            max_allowed_width = h_crop * 2
            if w_crop > max_allowed_width:
                cropped_panorama_np = cropped_panorama_np[:, :max_allowed_width]

            cropped_img = Image.fromarray(cropped_panorama_np)
        else: 
            cropped_img = image 

        return cropped_img.resize((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]), Image.Resampling.LANCZOS)

    cropped_pil = _crop(input_image_pil) 

    img_tensor_original = preprocess_transform(cropped_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_heatmap_original_raw = model(img_tensor_original)
    pred_heatmap_original_np = np.clip(pred_heatmap_original_raw.squeeze().cpu().numpy(), a_min=0, a_max=1)

    cropped_flipped_pil = ImageOps.mirror(cropped_pil)
    img_tensor_flipped = preprocess_transform(cropped_flipped_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_heatmap_flipped_raw = model(img_tensor_flipped)

    pred_heatmap_flipped_oriented_np = pred_heatmap_flipped_raw.squeeze().cpu().numpy()
    pred_heatmap_flipped_reverted_np = np.fliplr(pred_heatmap_flipped_oriented_np) 
    pred_heatmap_flipped_reverted_np = np.clip(pred_heatmap_flipped_reverted_np, a_min=0, a_max=1)

    pred_heatmap_np = np.maximum(pred_heatmap_original_np, pred_heatmap_flipped_reverted_np)

    output_image_pil = cropped_pil.convert("RGB") 
    draw = ImageDraw.Draw(output_image_pil)

    heatmap_height, heatmap_width = pred_heatmap_np.shape

    image_pil_width, image_pil_height = output_image_pil.size

    scale_x = image_pil_width / heatmap_width   
    scale_y = image_pil_height / heatmap_height 

    peak_coordinates = peak_local_max(
        pred_heatmap_np,
        min_distance=10,
        threshold_abs=0.4
    )

    x_mark_size = 20  
    x_mark_width = 11  

    for r_heatmap, c_heatmap in peak_coordinates:

        center_x_img = int(c_heatmap * scale_x)
        center_y_img = int(r_heatmap * scale_y)

        draw.line(
            [(center_x_img - x_mark_size, center_y_img - x_mark_size),
             (center_x_img + x_mark_size, center_y_img + x_mark_size)],
            fill="red",
            width=x_mark_width
        )

        draw.line(
            [(center_x_img + x_mark_size, center_y_img - x_mark_size),
             (center_x_img - x_mark_size, center_y_img + x_mark_size)],
            fill="red",
            width=x_mark_width
        )

    return output_image_pil 

try:
    model = load_trained_model(MODEL_CHECKPOINT_PATH, MODEL_HEATMAP_SIZE)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model checkpoint exists and the model definition is correct.")
    print("If using scikit-image, ensure it's installed (`pip install scikit-image`)")
    exit()

print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil", label="Upload Equirectangular Image"),
    outputs=gr.Image(type="pil", label="Image with Predicted Curb Ramps"),
    title="RampDet - Curb Ramp Detector Demo",
    description="Upload an equirectangular image."
)

print(f"Launching Gradio server on port {GRADIO_PORT}...")
iface.launch(server_name="0.0.0.0", server_port=GRADIO_PORT)