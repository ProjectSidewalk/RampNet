import os
import json
import sys

# Run as a script from this directory, so put the repo root on the path to reach
# the canonical evaluation definitions rather than re-implementing them here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rampnet.detection_eval import (  # noqa: E402
    PANO_RADIUS_NORMALIZED, PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for,
)
from rampnet.metrics import match_points  # noqa: E402

DATASET_ROOT_DIR = '../../dataset'
TEST_SPLIT_NAME = 'test'
MANUAL_DATASET_ROOT_DIR = '../../manual_labels'

# Matching radius and the anisotropic scaling of normalized coords (the pano is
# 2:1, so x and y scale differently) are shared with every other evaluator —
# see rampnet/detection_eval.py. Do not fork them here.
RADIUS_THRESHOLD_NORMALIZED = PANO_RADIUS_NORMALIZED
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def get_label_paths_by_basename(
    label_dir,
    label_extension,
    image_dir_for_check
):
    labels_map = {}
    if not os.path.isdir(label_dir):
        print(f"Warning: Label directory not found: {label_dir}")
        return labels_map
    if not os.path.isdir(image_dir_for_check):
        print(f"Warning: Image directory for checking image existence not found: {image_dir_for_check}")
        return labels_map

    for label_f_name in sorted(os.listdir(label_dir)):
        if label_f_name.lower().endswith(label_extension):
            base_name, _ = os.path.splitext(label_f_name)
            for img_ext in IMG_EXTENSIONS:
                img_f_name = base_name + img_ext
                potential_img_path = os.path.join(image_dir_for_check, img_f_name)
                if os.path.exists(potential_img_path):
                    labels_map[base_name] = (os.path.join(label_dir, label_f_name), potential_img_path)
                    break
    return labels_map


def load_manual_label_points(label_path):
    points = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x_norm = float(parts[1])
                    y_norm = float(parts[2])
                    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                        continue
                    points.append((x_norm, y_norm))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return points


def load_test_split_label_points(label_path):
    points = []
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
        raw_points = data.get("curb_ramp_points_normalized", [])
        for p in raw_points:
            if isinstance(p, (list, tuple)) and len(p) == 2 and \
               isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                x_norm, y_norm = p[0], p[1]
                if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                    continue
                points.append((x_norm, y_norm))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    return points


def main():
    manual_label_dir = MANUAL_DATASET_ROOT_DIR
    images_source_dir = os.path.join(DATASET_ROOT_DIR, TEST_SPLIT_NAME)
    test_split_label_dir = os.path.join(DATASET_ROOT_DIR, TEST_SPLIT_NAME)

    manual_labels_map = get_label_paths_by_basename(
        manual_label_dir, '.txt', images_source_dir
    )
    test_labels_map = get_label_paths_by_basename(
        test_split_label_dir, '.json', images_source_dir
    )

    common_files_info = []
    for basename, (manual_l_path, img_path) in manual_labels_map.items():
        if basename in test_labels_map:
            test_l_path, _ = test_labels_map[basename]
            common_files_info.append((basename, manual_l_path, test_l_path, img_path))

    if not common_files_info:
        print("\nNo common images with both types of labels found.")
        return

    print(f"Found {len(common_files_info)} common images.")

    total_true_positives = 0
    total_false_positives = 0
    total_redundant_fp = 0
    total_ground_truth_manual_points = 0

    radius_threshold_sq = radius_sq_for(RADIUS_THRESHOLD_NORMALIZED)

    for basename, manual_label_path, test_label_path, image_path in common_files_info:
        manual_points_gt = load_manual_label_points(manual_label_path)
        test_points_pred = load_test_split_label_points(test_label_path)

        total_ground_truth_manual_points += len(manual_points_gt)

        tp, fp, n_redundant = match_points(
            test_points_pred, manual_points_gt, radius_threshold_sq,
            PANO_SCALE_X, PANO_SCALE_Y,
        )
        total_true_positives += tp
        total_false_positives += fp
        total_redundant_fp += n_redundant

    total_predictions = total_true_positives + total_false_positives
    if total_predictions == 0:
        precision = 0.0 if total_ground_truth_manual_points > 0 else 1.0
    else:
        precision = total_true_positives / total_predictions

    if total_ground_truth_manual_points == 0:
        recall = 1.0 if total_predictions == 0 else 0.0
    else:
        recall = total_true_positives / total_ground_truth_manual_points

    total_false_negatives = total_ground_truth_manual_points - total_true_positives

    print("\n\n--- Dataset Label Accuracy Evaluation ---")
    print(f"Source of 'Ground Truth' labels: {MANUAL_DATASET_ROOT_DIR}")
    print(f"Source of 'Predicted' labels: {os.path.join(DATASET_ROOT_DIR, TEST_SPLIT_NAME)}")
    print(f"Matching radius threshold (normalized): {RADIUS_THRESHOLD_NORMALIZED}")
    print("-------------------------------------------------------------")
    print(f"Number of common images processed: {len(common_files_info)}")
    print(f"Total Ground Truth points (from manual labels in common images): {total_ground_truth_manual_points}")
    print(f"Total 'Predicted' points (TP+FP): {total_predictions}")
    print(f"  -> of which redundant (2nd point on an already-matched ramp, counted as FP): {total_redundant_fp}")
    print("-------------------------------------------------------------")
    print(f"True Positives (TP): {total_true_positives}")
    print(f"False Positives (FP): {total_false_positives}")
    print(f"False Negatives (FN): {total_false_negatives}")
    print("-------------------------------------------------------------")
    print(f"Precision (TP / (TP + FP)): {precision:.4f}")
    print(f"Recall    (TP / Total GT):  {recall:.4f}")
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()
