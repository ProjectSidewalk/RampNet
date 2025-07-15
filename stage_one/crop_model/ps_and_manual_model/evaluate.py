import os
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import timm
import torch.nn as nn
from tqdm import tqdm
from skimage.feature import peak_local_max

MODEL_CHECKPOINT_PATH = "best_model.pth"
DATASET_ROOT_DIR = './dataset_1'
TEST_SPLIT_NAME = 'test'

MODEL_INPUT_SIZE = (1024, 352)
MODEL_HEATMAP_SIZE = (256, 88)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RADIUS_THRESHOLD_NORMALIZED = 0.132
PEAK_MIN_DISTANCE = 10
PEAK_THRESHOLD_ABS = 0.0

CACHE_DIR = "evaluate_cache"
HEATMAP_CACHE_DIR = os.path.join(CACHE_DIR, "heatmaps")
RESULTS_DIR = "evaluation_results"
VISUALIZATIONS_BASE_DIR = "visualizations"


DATASET_ID_STR = f"{os.path.basename(os.path.normpath(DATASET_ROOT_DIR))}_{TEST_SPLIT_NAME}"

VISUALIZATION_CONF_THRESHOLD = 0.5
POINT_RADIUS_VIS = 5


class KeypointModel(nn.Module):
    def __init__(self, heatmap_size=(256, 88)):
        super(KeypointModel, self).__init__()
        backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=False, num_classes=0, global_pool='')
        self.feature_extractor = nn.Sequential(*list(backbone.children()))
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

def load_trained_model(checkpoint_path, heatmap_size):
    print(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}. Please ensure it exists.")
    model = KeypointModel(heatmap_size=heatmap_size)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    except AttributeError as e:
        print(f"Warning: AttributeError during torch.load: {e}. Attempting workaround.")
        import pickle
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except AttributeError:
                    print(f"Could not find class {name} in module {module}, returning generic object.")
                    return object 
        with open(checkpoint_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            checkpoint = unpickler.load()
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
    if all(key.startswith('module.') for key in state_dict.keys()): 
        print("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()} 
    try:
        model.load_state_dict(state_dict) 
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch between the model architecture defined here (possibly a fallback) and the one in the checkpoint.")
        print("Attempting to load with strict=False, unmatched keys will be ignored.")
        try:
            model.load_state_dict(state_dict, strict=False) 
        except RuntimeError as e_strict_false:
            print(f"Loading with strict=False also failed: {e_strict_false}")
            raise e_strict_false
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully (possibly with ignored keys if strict=False was used).")
    return model

preprocess_transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_peaks_from_heatmap(heatmap_np, min_distance, threshold_abs, heatmap_shape):
    heatmap_h, heatmap_w = heatmap_shape
    if heatmap_np.ndim > 2:
        heatmap_np = heatmap_np.squeeze()
    heatmap_np_contiguous = np.ascontiguousarray(heatmap_np)
    coordinates = peak_local_max(heatmap_np_contiguous, min_distance=min_distance, threshold_abs=threshold_abs, exclude_border=False)
    peaks_normalized = []
    for r, c in coordinates:
        confidence = heatmap_np[r, c]
        x_norm = c / heatmap_w
        y_norm = r / heatmap_h
        peaks_normalized.append((x_norm, y_norm, confidence))
    return peaks_normalized

def get_image_files(data_dir):
    image_files = []
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Please ensure it exists.")

    print(f"Scanning for images in: {data_dir}")
    for f_name in sorted(os.listdir(data_dir)):
        if f_name.lower().endswith(IMG_EXTENSIONS):
            image_files.append(os.path.join(data_dir, f_name))
    
    print(f"Found {len(image_files)} images in {data_dir}.")
    return image_files


def calculate_ap_and_pr_curve(all_predictions_details, total_gt_points):
    if total_gt_points == 0:
        return 0.0, [0.0], [0.0], [], []
    if not all_predictions_details:
        return 0.0, [0.0], [0.0], [], []

    sorted_preds = sorted(all_predictions_details, key=lambda x: x[0], reverse=True)
    sorted_confidences = [p[0] for p in sorted_preds]
    sorted_tp_flags = [p[1] for p in sorted_preds]

    tp_count = 0
    fp_count = 0
    ap = 0.0
    last_recall_val = 0.0
    raw_recalls_list = []
    raw_precisions_list = []

    for i in range(len(sorted_preds)):
        is_tp = sorted_tp_flags[i]
        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        current_precision = tp_count / (tp_count + fp_count)
        current_recall = tp_count / total_gt_points if total_gt_points > 0 else 0.0
        raw_recalls_list.append(current_recall)
        raw_precisions_list.append(current_precision)

        if is_tp:
            ap += current_precision * (current_recall - last_recall_val)
            last_recall_val = current_recall

    plot_recalls = [0.]
    plot_precisions = [1.0]

    if raw_recalls_list:
        interp_precisions = list(raw_precisions_list)
        for k_interp in range(len(interp_precisions) - 2, -1, -1):
            interp_precisions[k_interp] = max(interp_precisions[k_interp], interp_precisions[k_interp+1])
        plot_recalls.extend(raw_recalls_list)
        plot_precisions.extend(interp_precisions)

    if not plot_recalls or plot_recalls[-1] < 1.0 :
        if plot_recalls:
             plot_recalls.append(plot_recalls[-1])
             plot_precisions.append(0.)
    return ap, plot_recalls, plot_precisions, sorted_confidences, sorted_tp_flags

def calculate_pr_rc_confidence_curves(sorted_confidences, sorted_tp_flags, total_gt_points):
    if not sorted_confidences:
        conf_thresholds_for_plot = [0.0, 1.0]
        if total_gt_points > 0:
            precisions_at_thresholds_for_plot = [1.0, 1.0]
            recalls_at_thresholds_for_plot = [0.0, 0.0]
        else: 
            precisions_at_thresholds_for_plot = [0.0, 0.0]
            recalls_at_thresholds_for_plot = [0.0, 0.0]
        return conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot

    conf_thresholds_unique_desc = []
    precisions_at_thresholds_desc = []
    recalls_at_thresholds_desc = []
    tp_count_cumulative = 0
    for i in range(len(sorted_confidences)):
        if sorted_tp_flags[i]:
            tp_count_cumulative += 1
        num_preds_cumulative = i + 1
        current_precision = tp_count_cumulative / num_preds_cumulative
        if total_gt_points > 0:
            current_recall = tp_count_cumulative / total_gt_points
        else:
            current_recall = 0.0
        current_confidence_threshold = sorted_confidences[i]
        print(current_recall)
        print(current_precision)
        print(current_confidence_threshold)
        print("-----")
        if i == len(sorted_confidences) - 1 or sorted_confidences[i+1] < current_confidence_threshold:
            conf_thresholds_unique_desc.append(current_confidence_threshold)
            precisions_at_thresholds_desc.append(current_precision)
            recalls_at_thresholds_desc.append(current_recall)

    conf_thresholds_for_plot = list(reversed(conf_thresholds_unique_desc))
    precisions_at_thresholds_for_plot = list(reversed(precisions_at_thresholds_desc))
    recalls_at_thresholds_for_plot = list(reversed(recalls_at_thresholds_desc))

    if not conf_thresholds_for_plot or conf_thresholds_for_plot[0] > 0.0:
        conf_thresholds_for_plot.insert(0, 0.0)
        precisions_at_thresholds_for_plot.insert(0, precisions_at_thresholds_for_plot[0] if precisions_at_thresholds_for_plot else (0.0 if total_gt_points > 0 else 0.0))
        recalls_at_thresholds_for_plot.insert(0, recalls_at_thresholds_for_plot[0] if recalls_at_thresholds_for_plot else 0.0)

    if not conf_thresholds_for_plot or conf_thresholds_for_plot[-1] < 1.0:
        conf_thresholds_for_plot.append(1.0)
        precisions_at_thresholds_for_plot.append(1.0 if total_gt_points > 0 and (not precisions_at_thresholds_desc or precisions_at_thresholds_desc[-1] > 0) else 0.0) 
        recalls_at_thresholds_for_plot.append(0.0)
    elif recalls_at_thresholds_for_plot[-1] == 0.0 and conf_thresholds_for_plot[-1] == 1.0: 
         precisions_at_thresholds_for_plot[-1] = 1.0 if total_gt_points > 0 and (not precisions_at_thresholds_desc or precisions_at_thresholds_desc[-1] > 0) else 0.0


    return conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot

def main():
    os.makedirs(HEATMAP_CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    current_visualizations_dir = os.path.join(VISUALIZATIONS_BASE_DIR, DATASET_ID_STR)
    os.makedirs(current_visualizations_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {current_visualizations_dir}")

    print(f"Using device: {DEVICE}")
    print(f"Evaluating on dataset: {DATASET_ID_STR}")
    print(f"Dataset root: {DATASET_ROOT_DIR}, Split: {TEST_SPLIT_NAME}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    try:
        model = load_trained_model(MODEL_CHECKPOINT_PATH, MODEL_HEATMAP_SIZE)
    except FileNotFoundError as e:
        print(f"{e}")
        print("Attempting to create a dummy checkpoint for a test run as it's missing.")
        try:
            dummy_model_for_checkpoint = KeypointModel(heatmap_size=MODEL_HEATMAP_SIZE)
            torch.save({'model_state_dict': dummy_model_for_checkpoint.state_dict()}, MODEL_CHECKPOINT_PATH)
            print(f"Dummy checkpoint created at {MODEL_CHECKPOINT_PATH}. Please re-run the script.")
        except Exception as ex_create:
            print(f"Could not create dummy checkpoint: {ex_create}. Exiting.")
        return
    except Exception as load_err:
        print(f"Failed to load model: {load_err}. Exiting.")
        return

    heatmap_h, heatmap_w = MODEL_HEATMAP_SIZE
    RADIUS_THRESHOLD_PIXELS = RADIUS_THRESHOLD_NORMALIZED * (341 / 4)
    RADIUS_THRESHOLD_PIXELS_SQ = RADIUS_THRESHOLD_PIXELS**2
    print(f"Heatmap dimensions (H, W): ({heatmap_h}, {heatmap_w})")
    print(f"RADIUS_THRESHOLD_NORMALIZED: {RADIUS_THRESHOLD_NORMALIZED}")
    print(f"Effective RADIUS_THRESHOLD_PIXELS for matching: {RADIUS_THRESHOLD_PIXELS:.2f}")

    test_data_dir = os.path.join(DATASET_ROOT_DIR, TEST_SPLIT_NAME)
    try:
        image_paths = get_image_files(test_data_dir)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure the dataset directory structure is correct (e.g., ./dataset_1/test/). Exiting.")
        return

    if not image_paths:
        print(f"No images found in {test_data_dir}. Exiting.")
        return

    all_pred_details_for_ap = []
    total_gt_count = 0
    for i in tqdm(range(len(image_paths)), desc=f"Evaluating on {DATASET_ID_STR}"):
        img_path = image_paths[i]
        base_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        cached_heatmap_path = os.path.join(HEATMAP_CACHE_DIR, f"{base_name_no_ext}_heatmap.npy")

        gt_points_normalized = []
        input_image_pil_for_vis = None 

        try:
            input_image_pil = Image.open(img_path).convert("RGB")
            input_image_pil_for_vis = input_image_pil.copy() 
            original_w, original_h = input_image_pil.size

            filename_parts = base_name_no_ext.split('_-_')
            
            if len(filename_parts) > 1: 
                for point_str in filename_parts[1:]:
                    try:
                        x_pixel_str, y_pixel_str = point_str.split('_')
                        x_pixel = int(x_pixel_str)
                        y_pixel = int(y_pixel_str)
                        x_norm = x_pixel / original_w
                        y_norm = y_pixel / original_h
                        gt_points_normalized.append((x_norm, y_norm))
                    except ValueError:
                        print(f"Warning: Could not parse point string '{point_str}' in filename {os.path.basename(img_path)}. Skipping point.")
                    except ZeroDivisionError:
                        print(f"Warning: Original image dimensions are zero for {os.path.basename(img_path)}. Skipping point normalization.")
            elif '_' in base_name_no_ext and len(filename_parts) == 1 :
                 print(f"Warning: Filename {os.path.basename(img_path)} does not seem to follow the 'id_-_x1_y1_-_x2_y2' format for points. Assuming no GT points for this image.")
            
        except FileNotFoundError:
            print(f"Image file not found during GT extraction: {img_path}. Skipping.")
            continue
        except Exception as e_gt:
            print(f"Error loading image or parsing GT points from filename {os.path.basename(img_path)}: {e_gt}. Skipping.")
            continue
        
        total_gt_count += len(gt_points_normalized)

        if os.path.exists(cached_heatmap_path):
            combined_heatmap_np = np.load(cached_heatmap_path)
        else:
            if input_image_pil is None: 
                try:
                    input_image_pil = Image.open(img_path).convert("RGB")
                except Exception as e_load_heatmap:
                    print(f"Error re-loading image {img_path} for heatmap: {e_load_heatmap}. Skipping.")
                    continue
            
            img_tensor_original = preprocess_transform(input_image_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_heatmap_original_raw = model(img_tensor_original)
            pred_heatmap_original_np = np.clip(pred_heatmap_original_raw.squeeze().cpu().numpy(), 0, 1)
            
            input_image_flipped_pil = ImageOps.mirror(input_image_pil)
            img_tensor_flipped = preprocess_transform(input_image_flipped_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_heatmap_flipped_raw = model(img_tensor_flipped)
            pred_heatmap_flipped_oriented_np = np.clip(pred_heatmap_flipped_raw.squeeze().cpu().numpy(), 0, 1)
            pred_heatmap_flipped_reverted_np = np.fliplr(pred_heatmap_flipped_oriented_np)
            
            combined_heatmap_np = np.maximum(pred_heatmap_original_np, pred_heatmap_flipped_reverted_np)
            np.save(cached_heatmap_path, combined_heatmap_np)

        pred_peaks_normalized = extract_peaks_from_heatmap(
            combined_heatmap_np,
            min_distance=PEAK_MIN_DISTANCE,
            threshold_abs=PEAK_THRESHOLD_ABS,
            heatmap_shape=MODEL_HEATMAP_SIZE
        )
        
        current_image_preds_sorted_by_conf = sorted(pred_peaks_normalized, key=lambda p_item: p_item[2], reverse=True)
        
        k_gt_mappings = [False] * len(gt_points_normalized)
        for pred_x_norm, pred_y_norm, pred_conf in current_image_preds_sorted_by_conf:
            pred_x_px = pred_x_norm * (341 / 4)
            pred_y_px = pred_y_norm * (1024 / 4)
            
            passes = False
            
            for k_gt, (gt_x_norm, gt_y_norm) in enumerate(gt_points_normalized):
                gt_x_px = gt_x_norm * (341 / 4)
                gt_y_px = gt_y_norm * (1024 / 4)
                
                dist_sq = (pred_x_px - gt_x_px)**2 + (pred_y_px - gt_y_px)**2
                
                if dist_sq < RADIUS_THRESHOLD_PIXELS_SQ:
                    passes = True
                    if not k_gt_mappings[k_gt]:
                        all_pred_details_for_ap.append((pred_conf, True))
                    
                    k_gt_mappings[k_gt] = True
            
            if not passes:
                all_pred_details_for_ap.append((pred_conf, False))
        if input_image_pil_for_vis:
            draw = ImageDraw.Draw(input_image_pil_for_vis)
            original_w_vis, original_h_vis = input_image_pil_for_vis.size

            for gt_x_norm, gt_y_norm in gt_points_normalized:
                gt_x_px_orig = int(gt_x_norm * original_w_vis)
                gt_y_px_orig = int(gt_y_norm * original_h_vis)
                draw.ellipse([
                    (gt_x_px_orig - POINT_RADIUS_VIS, gt_y_px_orig - POINT_RADIUS_VIS),
                    (gt_x_px_orig + POINT_RADIUS_VIS, gt_y_px_orig + POINT_RADIUS_VIS)
                ], outline="yellow", width=2)

            for pred_x_norm_hm, pred_y_norm_hm, pred_conf in pred_peaks_normalized:
                if pred_conf >= VISUALIZATION_CONF_THRESHOLD:
                    pred_x_px_orig = int(pred_x_norm_hm * original_w_vis)
                    pred_y_px_orig = int(pred_y_norm_hm * original_h_vis)
                    draw.ellipse([
                        (pred_x_px_orig - POINT_RADIUS_VIS, pred_y_px_orig - POINT_RADIUS_VIS),
                        (pred_x_px_orig + POINT_RADIUS_VIS, pred_y_px_orig + POINT_RADIUS_VIS)
                    ], outline="blue", width=2)
            
            vis_filename = f"{base_name_no_ext}_visualization.png"
            vis_path = os.path.join(current_visualizations_dir, vis_filename)
            try:
                input_image_pil_for_vis.save(vis_path)
            except Exception as e_vis_save:
                print(f"Warning: Could not save visualization image {vis_path}: {e_vis_save}")


    ap, recalls_curve_plot, precisions_curve_plot, sorted_confidences, sorted_tp_flags = \
        calculate_ap_and_pr_curve(all_pred_details_for_ap, total_gt_count)
    
    num_pred_total_above_peak_thresh = len(sorted_confidences)
    print(f"\n--- Evaluation Results ({DATASET_ID_STR} dataset) ---")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Total Ground Truth Points: {total_gt_count}")
    print(f"Total Predictions (above PEAK_THRESHOLD_ABS={PEAK_THRESHOLD_ABS}): {num_pred_total_above_peak_thresh}")

    params_str = f"r{RADIUS_THRESHOLD_NORMALIZED}_pt{PEAK_THRESHOLD_ABS}"
    if num_pred_total_above_peak_thresh > 0 and total_gt_count > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(recalls_curve_plot, precisions_curve_plot, marker='.', linestyle='-')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (AP: {ap:.4f}) for {DATASET_ID_STR}\nParams: {params_str}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        pr_curve_filename = f"pr_curve_{DATASET_ID_STR}_{params_str}.png"
        pr_curve_path = os.path.join(RESULTS_DIR, pr_curve_filename)
        plt.savefig(pr_curve_path)
        plt.close()
        print(f"Precision-Recall curve saved to {pr_curve_path}")
    elif num_pred_total_above_peak_thresh == 0 and total_gt_count > 0:
        print(f"No predictions made (above PEAK_THRESHOLD_ABS={PEAK_THRESHOLD_ABS}), PR curve is effectively at (0,0). Skipping PR curve plot.")
    elif total_gt_count == 0:
        print("No ground truth points, AP is 0. Skipping PR curve plot.")
    
    conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot = \
        calculate_pr_rc_confidence_curves(sorted_confidences, sorted_tp_flags, total_gt_count)
    
    if not sorted_confidences and total_gt_count > 0:
        print(f"No predictions made above PEAK_THRESHOLD_ABS={PEAK_THRESHOLD_ABS} (or all predictions were filtered out). Plotting default P/R vs Confidence curves.")
    elif not sorted_confidences and total_gt_count == 0:
        print(f"No predictions and no ground truth. Plotting default P/R vs Confidence curves.")

    plt.figure(figsize=(10, 7))
    plt.plot(conf_thresholds_for_plot, precisions_at_thresholds_for_plot, marker='.', linestyle='-', label='Precision', color='blue')
    plt.plot(conf_thresholds_for_plot, recalls_at_thresholds_for_plot, marker='.', linestyle='-', label='Recall', color='red')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.title(f"Precision and Recall vs. Confidence for {DATASET_ID_STR}\nParams: {params_str}")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.legend()
    pr_rc_vs_c_filename = f"pr_rc_vs_c_curve_{DATASET_ID_STR}_{params_str}.png"
    pr_rc_vs_c_curve_path = os.path.join(RESULTS_DIR, pr_rc_vs_c_filename)
    plt.savefig(pr_rc_vs_c_curve_path)
    plt.close()
    print(f"Precision-Recall vs. Confidence curve saved to {pr_rc_vs_c_curve_path}")

if __name__ == "__main__":
    main()