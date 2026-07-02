import argparse
import os
import torch
import gradio as gr
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from torchvision import transforms

from skimage.feature import peak_local_max

from rampnet.model import KeypointModel
from rampnet.loading import load_checkpoint

MODEL_INPUT_SIZE = (2048, 4096)
MODEL_HEATMAP_SIZE = (512, 1024)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for the RampNet curb ramp detector.")
    parser.add_argument('--checkpoint', default="projectsidewalk/rampnet-model",
                        help="Local .pth checkpoint path, or a HuggingFace repo id such as "
                             "'projectsidewalk/rampnet-model' (the default: the released weights)")
    parser.add_argument('--port', type=int, default=25566, help="Gradio server port")
    return parser.parse_args()


def load_trained_model(checkpoint):
    """Loads the model from a local checkpoint file or a HuggingFace repo id."""
    if os.path.exists(checkpoint):
        print(f"Loading local model checkpoint from: {checkpoint}")
        model = KeypointModel(heatmap_size=MODEL_HEATMAP_SIZE)
        load_checkpoint(model, checkpoint, map_location=DEVICE)
    else:
        print(f"'{checkpoint}' is not a local file; loading it as a HuggingFace repo id.")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
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

args = parse_args()
model = load_trained_model(args.checkpoint)

print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil", label="Upload Equirectangular Image"),
    outputs=gr.Image(type="pil", label="Image with Predicted Curb Ramps"),
    title="RampDet - Curb Ramp Detector Demo",
    description="Upload an equirectangular image."
)

print(f"Launching Gradio server on port {args.port}...")
iface.launch(server_name="0.0.0.0", server_port=args.port)