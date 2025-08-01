import argparse
import os

import cv2
import numpy as np
import py_sod_metrics


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
parser.add_argument("--pred_path", type=str, required=True,
                    help="path to the prediction results")
parser.add_argument("--gt_path", type=str,
                    default="../wall_seg_crop/data_test/masks/",
                    help="path to the ground truth masks")
parser.add_argument("--masks_windoor_path", type=str,
                    default="../wall_seg_crop/data_test/masks_windoor/",
                    help="path to the windoor masks")
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
for i, mask_name in enumerate(mask_name_list):
    # if "0183_1-AP_001-FL_0183-越" not in mask_name:
    #     continue
    # 4.1. Get file info
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')
    mask_windoor_path = os.path.join(mask_windoor_root, mask_name[:-4] + '.png')
    # 4.2. Read images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    # Apply threshold to pred
    pred = pred >= 0.1 * 255
    pred = (pred * 255).astype(np.uint8)

    # 4.3. Segmentation Evaluation using room mask
    FMv2.step(pred=pred, gt=mask)
    mask_metrics = calculate_mask_metrics(pred=pred, gt=mask)
    print(f"{mask_metrics['mDice']:.3f}\t{mask_metrics['mIoU']:.3f}")

# After the loop, calculate overall metrics:
fmv2 = FMv2.get_results()

curr_results = {
    "meandice": fmv2["dice"]["dynamic"].mean(),
    "meaniou": fmv2["iou"]["dynamic"].mean(),
}

print("\nEvaluation results:")
print("mDice:       ", format(curr_results['meandice'], '.3f'))
print("mIoU:        ", format(curr_results['meaniou'], '.3f'))
