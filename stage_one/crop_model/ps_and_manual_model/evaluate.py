import os
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import peak_local_max

from rampnet.model import KeypointModel
from rampnet.loading import load_checkpoint, checkpoint_fingerprint
from rampnet.metrics import (
    calculate_ap_and_pr_curve,
    calculate_pr_rc_confidence_curves,
    match_predictions,
)
from rampnet.subcell import COARSE_ATOL, coarse_from_heatmap, coarse_mismatch, detect_peaks

MODEL_CHECKPOINT_PATH = "best_model.pth"
DATASET_ROOT_DIR = './dataset_1'
TEST_SPLIT_NAME = 'test'

MODEL_INPUT_SIZE = (1024, 352)
MODEL_HEATMAP_SIZE = (256, 88)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RADIUS_THRESHOLD_NORMALIZED = 0.132
PEAK_MIN_DISTANCE = 10
PEAK_THRESHOLD_ABS = 0.0
# Peak decode (#221). "argmax" is the historical behaviour and leaves every output,
# cache path and file name exactly as before. "gaussian" refines each peak to its
# sub-cell position from the raw branches' 32x11 coarse maps (rampnet/subcell.py;
# measured on this model in docs/crop_decode_221.md). It caches those coarse maps under
# evaluate_cache/coarse/<checkpoint fingerprint>/ (the heatmap cache holds the clipped
# TTA combine, which cannot be decoded exactly) and tags result files with _dgaussian.
PEAK_DECODE = "argmax"

CACHE_DIR = "evaluate_cache"
HEATMAP_CACHE_DIR = os.path.join(CACHE_DIR, "heatmaps")
COARSE_CACHE_DIR = os.path.join(CACHE_DIR, "coarse")
RESULTS_DIR = "evaluation_results"
VISUALIZATIONS_BASE_DIR = "visualizations"


DATASET_ID_STR = f"{os.path.basename(os.path.normpath(DATASET_ROOT_DIR))}_{TEST_SPLIT_NAME}"

VISUALIZATION_CONF_THRESHOLD = 0.5
POINT_RADIUS_VIS = 5


def load_trained_model(checkpoint_path, heatmap_size):
    print(f"Loading model checkpoint from: {checkpoint_path}")
    model = KeypointModel(heatmap_size=heatmap_size)
    load_checkpoint(model, checkpoint_path, map_location=DEVICE)
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

def predict_heatmap_and_coarse(model, input_image_pil):
    """evaluate.py's flip-TTA combine plus the float32 ``(2, 32, 11)`` stack of the raw
    branches' coarse maps, each oriented like the combined heatmap (for decode="gaussian")."""
    raws = []
    for img in (input_image_pil, ImageOps.mirror(input_image_pil)):
        t = preprocess_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raws.append(model(t).squeeze().cpu().numpy())
    raws[1] = np.fliplr(raws[1])
    combined = np.maximum(np.clip(raws[0], 0, 1), np.clip(raws[1], 0, 1))
    coarse = np.stack([coarse_from_heatmap(h) for h in raws]).astype(np.float32)
    return combined, coarse


def extract_peaks_decoded(heatmap_np, coarse, min_distance, threshold_abs, heatmap_shape,
                          decode):
    """Like extract_peaks_from_heatmap, with a sub-cell decode read from ``coarse``.

    Refuses a coarse stack that does not re-build the heatmap it is paired with (stale
    cache from other weights or code)."""
    worst = coarse_mismatch(heatmap_np, coarse, clip=True)
    if worst > COARSE_ATOL:
        raise ValueError(f"cached coarse maps disagree with the cached heatmap by {worst:.3g}; "
                         f"delete {CACHE_DIR}/ and re-run")
    heatmap_h, heatmap_w = heatmap_shape
    rcs = detect_peaks(heatmap_np, threshold_abs, min_distance=min_distance, decode=decode,
                       exclude_border=False, clip=True, coarse=coarse, wrap_x=False)
    return [(c / heatmap_w, r / heatmap_h, s) for r, c, s in rcs]


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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    decode_tag = "" if PEAK_DECODE == "argmax" else f"_d{PEAK_DECODE}"
    current_visualizations_dir = os.path.join(VISUALIZATIONS_BASE_DIR,
                                              DATASET_ID_STR + decode_tag)
    os.makedirs(current_visualizations_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {current_visualizations_dir}")

    print(f"Using device: {DEVICE}")
    print(f"Evaluating on dataset: {DATASET_ID_STR}")
    print(f"Dataset root: {DATASET_ROOT_DIR}, Split: {TEST_SPLIT_NAME}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    model = load_trained_model(MODEL_CHECKPOINT_PATH, MODEL_HEATMAP_SIZE)

    # Cached heatmaps are only valid for the weights that produced them, so
    # the cache directory is keyed by the checkpoint's content hash.
    ckpt_fingerprint = checkpoint_fingerprint(MODEL_CHECKPOINT_PATH)
    heatmap_cache_dir = os.path.join(HEATMAP_CACHE_DIR, ckpt_fingerprint)
    os.makedirs(heatmap_cache_dir, exist_ok=True)
    print(f"Heatmap cache directory: {heatmap_cache_dir}")
    coarse_cache_dir = os.path.join(COARSE_CACHE_DIR, ckpt_fingerprint)
    if PEAK_DECODE != "argmax":
        os.makedirs(coarse_cache_dir, exist_ok=True)
        print(f"Peak decode: {PEAK_DECODE}; coarse cache directory: {coarse_cache_dir}")

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
        cached_heatmap_path = os.path.join(heatmap_cache_dir, f"{base_name_no_ext}_heatmap.npy")

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

        coarse_stack = None
        if PEAK_DECODE != "argmax":
            cached_coarse_path = os.path.join(coarse_cache_dir, f"{base_name_no_ext}_coarse.npy")
            if os.path.exists(cached_coarse_path):
                coarse_stack = np.load(cached_coarse_path)
            else:
                fresh_heatmap, coarse_stack = predict_heatmap_and_coarse(model, input_image_pil)
                np.save(cached_coarse_path, coarse_stack)
                if not os.path.exists(cached_heatmap_path):
                    np.save(cached_heatmap_path, fresh_heatmap)

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

        if PEAK_DECODE == "argmax":
            pred_peaks_normalized = extract_peaks_from_heatmap(
                combined_heatmap_np,
                min_distance=PEAK_MIN_DISTANCE,
                threshold_abs=PEAK_THRESHOLD_ABS,
                heatmap_shape=MODEL_HEATMAP_SIZE
            )
        else:
            pred_peaks_normalized = extract_peaks_decoded(
                combined_heatmap_np, coarse_stack,
                min_distance=PEAK_MIN_DISTANCE,
                threshold_abs=PEAK_THRESHOLD_ABS,
                heatmap_shape=MODEL_HEATMAP_SIZE,
                decode=PEAK_DECODE,
            )
        
        all_pred_details_for_ap.extend(match_predictions(
            pred_peaks_normalized,
            gt_points_normalized,
            RADIUS_THRESHOLD_PIXELS_SQ,
            scale_x=341 / 4,
            scale_y=1024 / 4,
            # NOT wrap_x: this is CROP space, not a panorama. The two ends of a crop's
            # x axis are different places, so wrapping here would be a fresh bug, not a
            # fix. See rampnet/geometry.py and #132 section 5; pinned by a test.
        ))
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

    params_str = f"r{RADIUS_THRESHOLD_NORMALIZED}_pt{PEAK_THRESHOLD_ABS}" + decode_tag
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