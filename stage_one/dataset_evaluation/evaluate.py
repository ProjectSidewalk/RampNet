import os
import json


DATASET_ROOT_DIR = '../../dataset'
TEST_SPLIT_NAME = 'test'
MANUAL_DATASET_ROOT_DIR = '../../manual_labels'

RADIUS_THRESHOLD_NORMALIZED = 0.022
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Points are normalized to [0, 1]; the panorama is 2:1, so x and y are scaled by
# different factors before distances are measured (this matches how the labels
# were produced and how the model heatmap is laid out, 1024x512).
X_SCALE = 1024
Y_SCALE = 512


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


def match_points(pred_points, gt_points, radius_sq, x_scale=X_SCALE, y_scale=Y_SCALE):
    """Greedy one-to-one spatial matching of predicted points to ground truth.

    Each ground-truth point can be claimed by at most one prediction (the first
    in-range, still-unclaimed GT wins; a prediction whose nearest in-range GT is
    already claimed keeps looking for another unclaimed one before giving up).

    A prediction that lands within ``radius`` of ground truth but finds only
    already-claimed points is *redundant* — a second generated point on a ramp
    that was already counted — and is scored as a **false positive**. This aligns
    with ``rampnet/metrics.py``'s 1:1 matching semantics; the earlier version of
    this evaluator routed such points to an "ignored" bucket, which made the
    Stage 1 agreement precision slightly optimistic (issue #18).

    Returns ``(tp, fp, n_redundant)`` where ``n_redundant`` is the subset of
    ``fp`` that matched an already-claimed GT, reported for transparency.
    """
    gt_scaled = [(gx * x_scale, gy * y_scale) for gx, gy in gt_points]
    claimed = [False] * len(gt_scaled)
    tp = fp = n_redundant = 0

    for px, py in pred_points:
        px, py = px * x_scale, py * y_scale
        matched_any_gt = False
        is_tp = False

        for i, (gx, gy) in enumerate(gt_scaled):
            if (px - gx) ** 2 + (py - gy) ** 2 < radius_sq:
                matched_any_gt = True
                if not claimed[i]:
                    claimed[i] = True
                    is_tp = True
                    break

        if is_tp:
            tp += 1
        else:
            fp += 1
            if matched_any_gt:
                n_redundant += 1

    return tp, fp, n_redundant


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

    radius_threshold_sq = (RADIUS_THRESHOLD_NORMALIZED * X_SCALE) ** 2

    for basename, manual_label_path, test_label_path, image_path in common_files_info:
        manual_points_gt = load_manual_label_points(manual_label_path)
        test_points_pred = load_test_split_label_points(test_label_path)

        total_ground_truth_manual_points += len(manual_points_gt)

        tp, fp, n_redundant = match_points(
            test_points_pred, manual_points_gt, radius_threshold_sq
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
