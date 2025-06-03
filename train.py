import argparse
import os

import cv2
import numpy as np
import py_sod_metrics
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from SAM2UNet import SAM2UNet
from dataset import FullDataset, TestDataset

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--checkpoint", type=str,
                    default="",
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--train_image_path", type=str,
                    default="../data_crop/data_train/images/",
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str,
                    default="../data_crop/data_train/masks/",
                    help="path to the mask file for training")
parser.add_argument("--test_image_path", type=str,
                    default="../data_crop/data_test/images/",
                    help="path to the image that used to evaluate the model")
parser.add_argument("--test_gt_path", type=str,
                    default="../data_crop/data_test/masks/",
                    help="path to the mask file for evaluating")
parser.add_argument('--save_path', type=str,
                    default="../checkpoints_1152/",
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=150,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=3, type=int)
parser.add_argument("--size", default=1152, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--save_interval", default=10, type=int)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# --- Helper function for computing boundary weights (runs on CPU with NumPy/OpenCV) ---
def _compute_boundary_weights_np(mask_np: np.ndarray, weight_factor: float = 10.0,
                                 sigma: float = 5.0) -> np.ndarray:
    """
    Computes a boundary weight map from a single 2D binary mask using distance transform.

    Args:
        mask_np (np.ndarray): A single 2D binary mask (H, W), where foreground is 1 (or >0) and background is 0.
                              Expected dtype: np.uint8.
        weight_factor (float): Multiplier for the boundary weights. Higher means stronger emphasis.
        sigma (float): Standard deviation for the Gaussian applied to the distance map.
                       Controls the "spread" of the boundary emphasis.

    Returns:
        np.ndarray: A 2D weight map (H, W) with float32 dtype.
    """
    # Ensure mask is binary (0 or 255) for distanceTransform
    mask_binary = (mask_np > 0).astype(np.uint8) * 255

    # Compute distance transform for foreground (distance to nearest zero pixel)
    dist_to_bg = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

    # Compute distance transform for background (distance to nearest non-zero pixel)
    dist_to_fg = cv2.distanceTransform(255 - mask_binary, cv2.DIST_L2, 5)

    # Signed distance map: positive inside objects, negative outside, zero at boundary
    signed_distance_map = dist_to_bg - dist_to_fg

    # Apply a Gaussian-like weighting: highest at boundary (signed_distance_map = 0)
    # and decreasing as distance from boundary increases.
    # The formula exp(-(x^2) / (2 * sigma^2)) creates a bell curve centered at 0.
    weights = weight_factor * np.exp(-(signed_distance_map**2) / (2 * sigma**2))

    return weights.astype(np.float32)


# --- Modified structure_loss function ---
def structure_loss_boundary_aware(pred: torch.Tensor, mask: torch.Tensor,
                                  weight_factor: float = 10.0, sigma: float = 5.0) -> torch.Tensor:
    """
    Calculates a boundary-aware structure loss (BCE + IoU) for semantic segmentation.
    The boundary awareness is implemented using distance transform-based weighting.

    Args:
        pred (torch.Tensor): Model predictions (logits). Shape (N, 1, H, W).
        mask (torch.Tensor): Ground truth masks. Shape (N, 1, H, W) or (N, H, W).
                             Expected values: 0 or 1.
        weight_factor (float): Multiplier for the boundary weights. Higher means stronger emphasis.
        sigma (float): Standard deviation for the Gaussian applied to the distance map.
                       Controls the "spread" of the boundary emphasis.

    Returns:
        torch.Tensor: The scalar boundary-aware structure loss.
    """
    # Ensure mask is float32 for loss calculation
    mask = mask.float()

    # Ensure mask is (N, 1, H, W) for consistency, if it's (N, H, W)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    batch_size, _, H, W = mask.shape
    device = mask.device

    # --- Generate boundary weights for the current batch ---
    # Convert mask to NumPy array (N, H, W) for OpenCV processing
    mask_np_batch = mask.squeeze(1).cpu().numpy()  # Squeeze channel dim, move to CPU

    weit_maps = []
    for i in range(batch_size):
        # Compute weights for each mask
        current_weit = _compute_boundary_weights_np(mask_np_batch[i], weight_factor, sigma)
        weit_maps.append(torch.from_numpy(current_weit).to(device))

    # Stack weights to form a batch tensor (N, H, W) and unsqueeze for channel dim (N, 1, H, W)
    weit = torch.stack(weit_maps, dim=0).unsqueeze(1)

    # --- Apply weights to BCE loss (as per your original structure) ---
    # F.binary_cross_entropy_with_logits expects target to be float
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # --- Apply weights to IoU loss (as per your original structure) ---
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # Return the mean of the combined BCE and IoU losses across the batch
    return (wbce + wiou).mean()


# Define eval metrics
sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)


def main(args):
    # 1. Load train data
    dataset = FullDataset(args.train_image_path, args.train_mask_path, args.size, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # 2. Load test data
    test_loader = TestDataset(args.test_image_path, args.test_gt_path, args.size)
    # 3. Load model
    # Set device
    device = torch.device("cuda")
    # Load model to device
    model = SAM2UNet().to(device)

    # 4. Load checkpoint if provided
    if len(args.checkpoint) > 0:
        model.load_state_dict(torch.load(args.checkpoint), strict=True)
    # 5. Set optimizer
    optim = opt.AdamW([{"params": model.parameters(), "initia_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    # 6. Set scheduler
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    # 7. Train
    os.makedirs(args.save_path, exist_ok=True)
    epoch_loss = 2.0
    best_mean_iou = 0.7
    save_interval = args.save_interval
    for epoch in range(args.epoch):
        # 7.1. Train phase
        print("Training:")
        model.train()  # Set model to training mode
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss_boundary_aware(pred0, target)
            loss1 = structure_loss_boundary_aware(pred1, target)
            loss2 = structure_loss_boundary_aware(pred2, target)
            loss = loss0 + loss1 + loss2
            epoch_loss = loss.item()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                print("epoch-{}-{}: loss:{}".format(epoch + 1, i + 1, epoch_loss))
        scheduler.step()

        # 7.2. Evaluation phase
        print("Evaluating", end="")
        # init FMv2
        FMv2 = py_sod_metrics.FmeasureV2(
            metric_handlers={
                "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
                "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
                "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
                "rec": py_sod_metrics.RecallHandler(**sample_gray),
                "fpr": py_sod_metrics.FPRHandler(**sample_gray),
                "iou": py_sod_metrics.IOUHandler(**sample_gray),
                "dice": py_sod_metrics.DICEHandler(**sample_gray),
                "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
                "ber": py_sod_metrics.BERHandler(**sample_gray),
                "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
                "kappa": py_sod_metrics.KappaHandler(**sample_gray),
                "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
                "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
                "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
                "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
                "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
                "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
                "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
                "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
                "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
                "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
                "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
                "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
                "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
                "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
                "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
                "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
                "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
                "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
                "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
                "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
                "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
                "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
            }
        )
        # Set model to evaluation mode
        model.eval()
        # Disable gradient calculations for efficiency and safety
        for i in range(test_loader.size):
            with torch.no_grad():
                image, gt, name = test_loader.load_data()
                image = image.to(device)
                res, _, _ = model(image)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu()
                res = res.numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)
                gt = np.asarray(gt, np.float32)
                # Evaluate
                FMv2.step(pred=res, gt=gt)
                # Print for status
                if i % 10 == 0:
                    print(".", end="", flush=True)

        # Reset test_loader index
        test_loader.reset_index()
        # Get mIoU
        fmv2 = FMv2.get_results()
        mean_iou = fmv2["iou"]["dynamic"].mean()
        print(
            "\nepoch-{}: loss: {} mIoU: {} best_mIoU: {}\n".format(epoch + 1, epoch_loss, mean_iou,
                                                                   best_mean_iou))

        # 7.3. Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == args.epoch:
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print('Saving Snapshot:', save_model_path)
        elif mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print(f'Saving Snapshot best:', save_model_path)


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)
