import argparse
import os

import cv2
import numpy as np
import py_sod_metrics
from shapely.geometry import Polygon
from shapely.validation import make_valid  # To handle invalid polygons if they arise

IOU_THRESHOLD = 0.95  # IoU threshold to consider a match


def calculate_iou_from_polygons(
    poly1: Polygon, poly2: Polygon, area_threshold: float = 0.001
) -> float:
    """
    Calculates IoU coefficient between two Shapely Polygon objects.

    Args:
        poly1 (Polygon): The first polygon (e.g., ground truth contour).
        poly2 (Polygon): The second polygon (e.g., predicted contour).
        area_threshold (float): Minimum area for a polygon to be considered non-empty.
                                Helps handle floating point inaccuracies for very small overlaps.

    Returns:
        float: iou_score. Returns 0.0 if no valid intersection or if either polygon is effectively empty.
    """
    # 1. Make sure shapely polygon is valid
    if not poly1.is_valid:
        poly1 = make_valid(poly1)
    if not poly2.is_valid:
        poly2 = make_valid(poly2)

    # 2. Check for empty polygons
    if poly1.area < area_threshold or poly2.area < area_threshold:
        return 0.0

    # 3. Calculate intersection and union
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    # 3. Calculate iou
    iou = intersection_area / union_area if union_area > area_threshold else 0.0

    return iou


def opencv_contour_to_shapely_polygon(contour: np.ndarray) -> Polygon:
    """
    Converts an OpenCV contour (numpy array) to a Shapely Polygon.
    OpenCV contours are typically (N, 1, 2) and need to be reshaped to (N, 2).
    Also, handles potential issues with invalid polygons from simple contour points.
    """
    if contour.shape[1] == 1 and contour.shape[2] == 2:
        # Reshape (N, 1, 2) to (N, 2) for Shapely
        polygon_points = contour.reshape(-1, 2)
    else:
        polygon_points = contour  # Assume it's already (N, 2)

    if polygon_points.shape[0] < 3:  # A polygon needs at least 3 points
        return Polygon()  # Return an empty polygon if not enough points

    try:
        # Ensure points are integers for consistency if coming from pixels
        poly = Polygon(polygon_points)
        if not poly.is_valid:
            poly = make_valid(poly)  # Attempt to fix invalid polygons
        return poly
    except Exception:
        # print(f"Warning: Could not create Shapely Polygon from contour. Error: {e}")
        return Polygon()  # Return an empty polygon on error


def evaluate_contours(
    gt_contours: list[np.ndarray],
    pred_contours: list[np.ndarray],
) -> dict:
    """
    Evaluates detected contours against ground truth contours for a single image.
    Calculates True Positives, False Positives, and False Negatives based on IoU matching.

    Args:
        gt_contours (list[np.ndarray]): List of ground truth contours (OpenCV format).
        pred_contours (list[np.ndarray]): List of predicted contours (OpenCV format).

    Returns:
        dict: Contains counts of 'TP', 'FP', 'FN', 'matched_ious', 'mean_iou_matched'.
    """
    # 1. Convert OpenCV contours to Shapely Polygons
    gt_polygons = [opencv_contour_to_shapely_polygon(c) for c in gt_contours]
    pred_polygons = [opencv_contour_to_shapely_polygon(c) for c in pred_contours]

    # 2. Filter out invalid/empty polygons from the conversion
    gt_polygons = [p for p in gt_polygons if not p.is_empty and p.area > 0]
    pred_polygons = [p for p in pred_polygons if not p.is_empty and p.area > 0]

    # 3. Initial check gt and pred
    num_gt = len(gt_polygons)
    num_pred = len(pred_polygons)
    if num_gt == 0 and num_pred == 0:
        return {"TP": 0, "FP": 0, "FN": 0}
    if num_gt == 0 and num_pred > 0:
        return {"TP": 0, "FP": num_pred, "FN": 0}
    if num_gt > 0 and num_pred == 0:
        return {"TP": 0, "FP": 0, "FN": num_gt}

    # 4. Initialize lists to track matches
    # Boolean arrays to mark if a GT or Pred contour has been matched
    gt_matched = [False] * num_gt
    pred_matched = [False] * num_pred

    # 5. Calculate IoU matrix
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt_poly in enumerate(gt_polygons):
        for j, pred_poly in enumerate(pred_polygons):
            iou_matrix[i, j] = calculate_iou_from_polygons(gt_poly, pred_poly)

    # 6. Greedily match predictions to ground truths based on highest IoU
    # Iterate through potential matches from highest IoU downwards
    # Flatten the matrix and get indices sorted by IoU
    sorted_indices = np.argsort(iou_matrix.flatten())[::-1]  # Descending order

    # Store all valid IoU scores for matched pairs
    TP = 0
    for flat_idx in sorted_indices:
        gt_idx = flat_idx // num_pred
        pred_idx = flat_idx % num_pred

        if iou_matrix[gt_idx, pred_idx] >= IOU_THRESHOLD:
            if not gt_matched[gt_idx] and not pred_matched[pred_idx]:
                TP += 1
                gt_matched[gt_idx] = True
                pred_matched[pred_idx] = True

    # 7. Calculate the remaining metrics
    # 7.1. Count False Positives (predictions not matched)
    FP = sum(1 for matched in pred_matched if not matched)
    # 7.2.Count False Negatives (ground truths not matched)
    FN = sum(1 for matched in gt_matched if not matched)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }


def find_contours_with_parent(mask: np.ndarray) -> np.array:
    """Return contours with parent found in mask"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours_with_parent = []
    for i, contour in enumerate(contours):
        parent_idx = hierarchy[0][i][3]
        # Room contour has parent and no grandparent
        if parent_idx != -1 and hierarchy[0][parent_idx][3] == -1:
            contours_with_parent.append(contour)
    return contours_with_parent


def extract_room_contours(mask: np.ndarray, windoor_mask: np.ndarray):
    """Extract room contours from combination of mask and windoor_mask"""
    # 1. Combine mask and windoor_mask
    wall_mask = cv2.bitwise_or(mask, windoor_mask)
    # 2. Dilate to connect close pixels
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_mask = cv2.dilate(wall_mask, dilate_kernel)
    return wall_mask, find_contours_with_parent(wall_mask)


def draw_contours(mask: np.ndarray, contours: np.array) -> np.ndarray:
    """Draw contours"""
    contour_mask = np.zeros_like(mask)
    for i, contour in enumerate(contours):
        cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
    return contour_mask


def calculate_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Calculates the Dice coefficient (mDice) and Intersection over Union (mIoU)
    for a single pair of masks.
    """
    sample_gray_ = dict(with_adaptive=True, with_dynamic=True)
    FMv2_ = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "iou": py_sod_metrics.IOUHandler(**sample_gray_),
            "dice": py_sod_metrics.DICEHandler(**sample_gray_),
        }
    )
    FMv2_.step(pred=pred, gt=gt)
    fmv2_ = FMv2_.get_results()

    return {"mDice": fmv2_["dice"]["dynamic"].mean(), "mIoU": fmv2_["iou"]["dynamic"].mean()}


# 1. Define args
parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, required=True, help="path to the prediction results")
parser.add_argument(
    "--gt_path",
    type=str,
    default="../wall_seg_crop/data_test/masks/",
    help="path to the ground truth masks",
)
parser.add_argument(
    "--masks_windoor_path",
    type=str,
    default="../wall_seg_crop/data_test/masks_windoor/",
    help="path to the windoor masks",
)
args = parser.parse_args()

# 2. Define FmeasureV2
sample_gray = dict(with_adaptive=True, with_dynamic=True)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
    }
)

# 3. Get info from args
pred_root = args.pred_path
mask_root = args.gt_path
mask_windoor_root = args.masks_windoor_path
mask_name_list = sorted(os.listdir(mask_root))

# 4. Evaluate
all_tps = 0
all_fps = 0
all_fns = 0
for i, mask_name in enumerate(mask_name_list):
    # if "0183_1-AP_001-FL_0183-越" not in mask_name:
    #     continue
    # 4.1. Get file info
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name[:-4] + ".png")
    mask_windoor_path = os.path.join(mask_windoor_root, mask_name[:-4] + ".png")
    # 4.2. Read images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    # Apply threshold to pre
    pred = pred >= 0.1 * 255
    pred = (pred * 255).astype(np.uint8)

    # 4.3. Combine mask and pre with windoor mask to extract room contours
    windoor_mask = cv2.imread(mask_windoor_path, cv2.IMREAD_GRAYSCALE)
    _, gt_contours = extract_room_contours(mask, windoor_mask)
    _, pred_contours = extract_room_contours(pred, windoor_mask)

    # 4.4. Draw room contours to create room mask
    mask_room = draw_contours(mask=mask, contours=gt_contours)
    pred_room = draw_contours(mask=pred, contours=pred_contours)

    # 4.3. Segmentation Evaluation using room mask
    FMv2.step(pred=pred_room, gt=mask_room)
    mask_metrics = calculate_mask_metrics(pred=pred_room, gt=mask_room)
    print(f"{mask_metrics['mDice']:.3f}\t{mask_metrics['mIoU']:.3f}")

    # 4.4. Detection Evaluation using contours
    contour_eval_results = evaluate_contours(gt_contours=gt_contours, pred_contours=pred_contours)
    # 4.5. Add evaluation results
    # True Positives
    all_tps += contour_eval_results["TP"]
    # False Positives
    all_fps += contour_eval_results["FP"]
    # False Negatives
    all_fns += contour_eval_results["FN"]
    # 4.6

# After the loop, calculate overall metrics:
overall_precision = all_tps / (all_tps + all_fps) if (all_tps + all_fps) > 0 else 0.0
overall_recall = all_tps / (all_tps + all_fns) if (all_tps + all_fns) > 0 else 0.0
overall_f1 = (
    (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
    if (overall_precision + overall_recall) > 0
    else 0.0
)

fmv2 = FMv2.get_results()

curr_results = {
    "meandice": fmv2["dice"]["dynamic"].mean(),
    "meaniou": fmv2["iou"]["dynamic"].mean(),
}

print("\nEvaluation results:")
print("Precision:   ", format(overall_precision, ".3f"))
print("Recall:      ", format(overall_recall, ".3f"))
print("F1:          ", format(overall_f1, ".3f"))
print("mDice:       ", format(curr_results["meandice"], ".3f"))
print("mIoU:        ", format(curr_results["meaniou"], ".3f"))
