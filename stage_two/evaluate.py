import argparse
import os
import shutil
import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from skimage.feature import peak_local_max
import csv

from rampnet.model import KeypointModel
from rampnet.loading import load_checkpoint, checkpoint_fingerprint
from rampnet.metrics import (
    calculate_ap_and_pr_curve,
    calculate_pr_rc_confidence_curves,
    match_predictions,
)

MODEL_INPUT_SIZE = (2048, 4096)
MODEL_HEATMAP_SIZE = (512, 1024)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RADIUS_THRESHOLD_NORMALIZED = 0.022
PEAK_MIN_DISTANCE = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the stage-2 panorama curb ramp detector.")
    parser.add_argument('--checkpoint', default="checkpoints/epoch_1_step_9378.pth",
                        help="Path to the trained checkpoint")
    parser.add_argument('--dataset', choices=['manual', 'test'], default='manual',
                        help="'manual' = 1k manually labeled gold set (default); "
                             "'test' = test split of the generated dataset")
    parser.add_argument('--data-root', default='../dataset',
                        help="Dataset root containing the test split (and the images for manual labels)")
    parser.add_argument('--manual-labels', default='../manual_labels',
                        help="Directory of manual .txt label files")
    parser.add_argument('--threshold', type=float, default=0.0,
                        help="Peak-extraction confidence threshold. The default 0.0 keeps every peak "
                             "so the full PR curve can be swept; see the README's 'Choosing a "
                             "Detection Threshold' section for operating points.")
    parser.add_argument('--tta', action=argparse.BooleanOptionalAction, default=True,
                        help="Horizontal-flip test-time augmentation (default: on, as in the paper)")
    parser.add_argument('--cache-dir', default='evaluate_cache',
                        help="Heatmap cache root (keyed by checkpoint hash + dataset + TTA setting)")
    parser.add_argument('--fresh', action='store_true',
                        help="Delete this checkpoint's cached heatmaps before evaluating")
    parser.add_argument('--results-dir', default='evaluation_results',
                        help="Where plots, CSVs, and metrics.json are written")
    return parser.parse_args()


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


def get_test_files(primary_path,
                     secondary_path_or_split_name,
                     is_manual_dataset=False):
    image_files = []
    label_files = []
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    if is_manual_dataset:
        manual_label_dir = primary_path
        image_source_dir = secondary_path_or_split_name
        label_extension = '.txt'

        if not os.path.isdir(manual_label_dir):
            raise FileNotFoundError(f"Manual label directory not found: {manual_label_dir}.")
        if not os.path.isdir(image_source_dir):
            raise FileNotFoundError(f"Image source directory for manual labels not found: {image_source_dir}.")

        print(f"Scanning for manual labels ({label_extension}) in: {manual_label_dir}")
        print(f"Looking for corresponding images in: {image_source_dir}")

        for label_f_name in sorted(os.listdir(manual_label_dir)):
            if label_f_name.lower().endswith(label_extension):
                base_name, _ = os.path.splitext(label_f_name)
                found_image = False
                for img_ext in IMG_EXTENSIONS:
                    img_f_name = base_name + img_ext
                    potential_img_path = os.path.join(image_source_dir, img_f_name)
                    if os.path.exists(potential_img_path):
                        image_files.append(potential_img_path)
                        label_files.append(os.path.join(manual_label_dir, label_f_name))
                        found_image = True
                        break
                if not found_image:
                    print(f"Warning: Manual label file '{label_f_name}' found, but no corresponding image (tried {', '.join(IMG_EXTENSIONS)}) in '{image_source_dir}'. Skipping.")
        print(f"Found {len(image_files)} image/{label_extension[1:].upper()} pairs for manual dataset.")
    else:
        dataset_root = primary_path
        split_name = secondary_path_or_split_name
        data_dir = os.path.join(dataset_root, split_name)
        label_extension = '.json'

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Test split directory not found: {data_dir}.")

        for f_name in sorted(os.listdir(data_dir)):
            if f_name.lower().endswith(IMG_EXTENSIONS):
                base_name, _ = os.path.splitext(f_name)
                label_f_name = base_name + label_extension
                if os.path.exists(os.path.join(data_dir, label_f_name)):
                    image_files.append(os.path.join(data_dir, f_name))
                    label_files.append(os.path.join(data_dir, label_f_name))
        print(f"Found {len(image_files)} image/{label_extension[1:].upper()} pairs in {data_dir}.")
    return image_files, label_files


def load_gt_points(label_path, is_manual_dataset):
    gt_points_normalized = []
    try:
        if is_manual_dataset:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x_norm = float(parts[1])
                        y_norm = float(parts[2])
                        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                            print(f"Warning: Normalized coordinates out of [0,1] range in {label_path}: x={x_norm}, y={y_norm}. Skipping point.")
                            continue
                        gt_points_normalized.append((x_norm, y_norm))
                    except ValueError:
                        print(f"Warning: Could not parse coordinates in line '{line.strip()}' from {label_path}. Skipping line.")
                elif line.strip():
                    print(f"Warning: Malformed line in {label_path}: '{line.strip()}'. Skipping line.")
        else:
            with open(label_path, 'r') as f:
                data = json.load(f)
            raw_points = data.get("curb_ramp_points_normalized", [])
            for p in raw_points:
                if isinstance(p, (list, tuple)) and len(p) == 2 and \
                   isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                    gt_points_normalized.append(tuple(p))
    except FileNotFoundError:
        print(f"Label file not found: {label_path}. Assuming no GT points.")
    except Exception as e:
        print(f"Error loading labels from {label_path}: {e}. Assuming no GT points.")
    return gt_points_normalized


def predict_heatmap(model, input_image_pil, use_tta):
    img_tensor_original = preprocess_transform(input_image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_heatmap_original_raw = model(img_tensor_original)
    pred_heatmap_original_np = np.clip(pred_heatmap_original_raw.squeeze().cpu().numpy(), 0, 1)
    if not use_tta:
        return pred_heatmap_original_np
    input_image_flipped_pil = ImageOps.mirror(input_image_pil)
    img_tensor_flipped = preprocess_transform(input_image_flipped_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_heatmap_flipped_raw = model(img_tensor_flipped)
    pred_heatmap_flipped_oriented_np = np.clip(pred_heatmap_flipped_raw.squeeze().cpu().numpy(), 0, 1)
    pred_heatmap_flipped_reverted_np = np.fliplr(pred_heatmap_flipped_oriented_np)
    return np.maximum(pred_heatmap_original_np, pred_heatmap_flipped_reverted_np)


def evaluate(model, image_paths, label_paths, is_manual_dataset, heatmap_cache_dir,
             peak_threshold_abs=0.0, use_tta=True, dataset_id_str="dataset"):
    """Run detection evaluation and return a metrics dict.

    Importable entry point for automated pipelines (e.g. retraining gates and
    the HF model-card generator); main() adds plots/CSVs around it.
    """
    heatmap_h, heatmap_w = MODEL_HEATMAP_SIZE
    radius_threshold_pixels = RADIUS_THRESHOLD_NORMALIZED * heatmap_w
    radius_threshold_pixels_sq = radius_threshold_pixels**2

    all_pred_details_for_ap = []
    total_gt_count = 0
    for i in tqdm(range(len(image_paths)), desc=f"Evaluating on {dataset_id_str}"):
        img_path = image_paths[i]
        label_path = label_paths[i]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cached_heatmap_path = os.path.join(heatmap_cache_dir, f"{base_name}_heatmap.npy")
        if os.path.exists(cached_heatmap_path):
            combined_heatmap_np = np.load(cached_heatmap_path)
        else:
            try:
                input_image_pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Skipping.")
                continue
            combined_heatmap_np = predict_heatmap(model, input_image_pil, use_tta)
            np.save(cached_heatmap_path, combined_heatmap_np)

        gt_points_normalized = load_gt_points(label_path, is_manual_dataset)
        total_gt_count += len(gt_points_normalized)
        pred_peaks_normalized = extract_peaks_from_heatmap(
            combined_heatmap_np,
            min_distance=PEAK_MIN_DISTANCE,
            threshold_abs=peak_threshold_abs,
            heatmap_shape=MODEL_HEATMAP_SIZE
        )
        all_pred_details_for_ap.extend(match_predictions(
            pred_peaks_normalized,
            gt_points_normalized,
            radius_threshold_pixels_sq,
            scale_x=heatmap_w,
            scale_y=heatmap_h,
        ))

    ap, recalls_curve_plot, precisions_curve_plot, sorted_confidences, sorted_tp_flags = \
        calculate_ap_and_pr_curve(all_pred_details_for_ap, total_gt_count)

    tp_at_threshold = sum(1 for conf, is_tp in all_pred_details_for_ap if is_tp)
    num_preds = len(all_pred_details_for_ap)
    precision_at_threshold = tp_at_threshold / num_preds if num_preds > 0 else 0.0
    recall_at_threshold = tp_at_threshold / total_gt_count if total_gt_count > 0 else 0.0

    return {
        'dataset': dataset_id_str,
        'ap': ap,
        'total_gt_points': total_gt_count,
        'total_predictions': num_preds,
        'peak_threshold_abs': peak_threshold_abs,
        'precision_at_threshold': precision_at_threshold,
        'recall_at_threshold': recall_at_threshold,
        'radius_threshold_normalized': RADIUS_THRESHOLD_NORMALIZED,
        'tta': use_tta,
        'recalls_curve': recalls_curve_plot,
        'precisions_curve': precisions_curve_plot,
        'sorted_confidences': sorted_confidences,
        'sorted_tp_flags': sorted_tp_flags,
    }


def main():
    args = parse_args()
    evaluate_on_manual = args.dataset == 'manual'
    dataset_id_str = 'manual' if evaluate_on_manual else 'test'

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Evaluating on {'MANUAL dataset' if evaluate_on_manual else 'ORIGINAL test dataset'}")
    print(f"Results directory: {args.results_dir}")

    model = load_trained_model(args.checkpoint, MODEL_HEATMAP_SIZE)

    # Cached heatmaps are only valid for the exact weights, dataset, and TTA setting
    # that produced them, so the cache directory is keyed by all three. The dataset id
    # matters because 'manual' and 'test' draw images from the same test split and so
    # share pano ids; without it in the key they collide and silently serve each other's
    # cached heatmaps.
    ckpt_fingerprint = checkpoint_fingerprint(args.checkpoint)
    cache_key = f"{ckpt_fingerprint}_{dataset_id_str}_{'tta' if args.tta else 'notta'}"
    heatmap_cache_dir = os.path.join(args.cache_dir, "heatmaps", cache_key)
    if args.fresh and os.path.isdir(heatmap_cache_dir):
        print(f"--fresh: clearing cached heatmaps in {heatmap_cache_dir}")
        shutil.rmtree(heatmap_cache_dir)
    os.makedirs(heatmap_cache_dir, exist_ok=True)
    print(f"Cache directory: {heatmap_cache_dir}")

    if evaluate_on_manual:
        image_paths, label_paths = get_test_files(
            primary_path=args.manual_labels,
            secondary_path_or_split_name=os.path.join(args.data_root, 'test'),
            is_manual_dataset=True
        )
    else:
        image_paths, label_paths = get_test_files(
            primary_path=args.data_root,
            secondary_path_or_split_name='test',
            is_manual_dataset=False
        )

    if not image_paths:
        raise FileNotFoundError("No image/label pairs found.")

    metrics = evaluate(
        model, image_paths, label_paths,
        is_manual_dataset=evaluate_on_manual,
        heatmap_cache_dir=heatmap_cache_dir,
        peak_threshold_abs=args.threshold,
        use_tta=args.tta,
        dataset_id_str=dataset_id_str,
    )
    ap = metrics['ap']
    total_gt_count = metrics['total_gt_points']
    recalls_curve_plot = metrics['recalls_curve']
    precisions_curve_plot = metrics['precisions_curve']
    sorted_confidences = metrics['sorted_confidences']
    sorted_tp_flags = metrics['sorted_tp_flags']

    num_pred_total_above_peak_thresh = len(sorted_confidences)
    print(f"\n--- Evaluation Results ({dataset_id_str} dataset) ---")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Total Ground Truth Points: {total_gt_count}")
    print(f"Total Predictions (above threshold={args.threshold}): {num_pred_total_above_peak_thresh}")
    if args.threshold > 0.0:
        print(f"Precision at threshold {args.threshold}: {metrics['precision_at_threshold']:.4f}")
        print(f"Recall at threshold {args.threshold}: {metrics['recall_at_threshold']:.4f}")

    params_str = f"r{RADIUS_THRESHOLD_NORMALIZED}_pt{args.threshold}"

    metrics_json = {k: v for k, v in metrics.items()
                    if k not in ('recalls_curve', 'precisions_curve', 'sorted_confidences', 'sorted_tp_flags')}
    metrics_json['checkpoint'] = args.checkpoint
    metrics_json['checkpoint_fingerprint'] = ckpt_fingerprint
    metrics_json_path = os.path.join(args.results_dir, f"metrics_{dataset_id_str}_{params_str}.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    if num_pred_total_above_peak_thresh > 0 and total_gt_count > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(recalls_curve_plot, precisions_curve_plot, marker='.', linestyle='-')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (AP: {ap:.4f}) for {dataset_id_str}\nParams: {params_str}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        pr_curve_filename = f"pr_curve_{dataset_id_str}_{params_str}.png"
        pr_curve_path = os.path.join(args.results_dir, pr_curve_filename)
        plt.savefig(pr_curve_path)
        plt.close()
        print(f"Precision-Recall curve saved to {pr_curve_path}")

        pr_csv_filename = f"pr_data_{dataset_id_str}_{params_str}.csv"
        pr_csv_path = os.path.join(args.results_dir, pr_csv_filename)
        with open(pr_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['recall', 'precision'])
            writer.writerows(zip(recalls_curve_plot, precisions_curve_plot))
        print(f"Precision-Recall data saved to {pr_csv_path}")

    elif num_pred_total_above_peak_thresh == 0 and total_gt_count > 0:
        print(f"No predictions made (above threshold={args.threshold}), PR curve is effectively at (0,0). Skipping PR curve plot.")
    elif total_gt_count == 0:
        print("No ground truth points, AP is 0. Skipping PR curve plot.")

    conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot = \
        calculate_pr_rc_confidence_curves(sorted_confidences, sorted_tp_flags, total_gt_count)
    if not sorted_confidences and total_gt_count > 0:
        print(f"No predictions made above threshold={args.threshold} (or all predictions were filtered out). Plotting default P/R vs Confidence curves.")
    elif not sorted_confidences and total_gt_count == 0:
        print("No predictions and no ground truth. Plotting default P/R vs Confidence curves.")

    plt.figure(figsize=(10, 7))
    plt.plot(conf_thresholds_for_plot, precisions_at_thresholds_for_plot, marker='.', linestyle='-', label='Precision', color='blue')
    plt.plot(conf_thresholds_for_plot, recalls_at_thresholds_for_plot, marker='.', linestyle='-', label='Recall', color='red')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.title(f"Precision and Recall vs. Confidence for {dataset_id_str}\nParams: {params_str}")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.legend()
    pr_rc_vs_c_filename = f"pr_rc_vs_c_curve_{dataset_id_str}_{params_str}.png"
    pr_rc_vs_c_curve_path = os.path.join(args.results_dir, pr_rc_vs_c_filename)
    plt.savefig(pr_rc_vs_c_curve_path)
    plt.close()
    print(f"Precision-Recall vs. Confidence curve saved to {pr_rc_vs_c_curve_path}")

    pr_rc_vs_c_csv_filename = f"pr_rc_vs_c_data_{dataset_id_str}_{params_str}.csv"
    pr_rc_vs_c_csv_path = os.path.join(args.results_dir, pr_rc_vs_c_csv_filename)
    with open(pr_rc_vs_c_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['confidence_threshold', 'precision', 'recall'])
        writer.writerows(zip(conf_thresholds_for_plot, precisions_at_thresholds_for_plot, recalls_at_thresholds_for_plot))
    print(f"Precision-Recall vs. Confidence data saved to {pr_rc_vs_c_csv_path}")


if __name__ == "__main__":
    main()
