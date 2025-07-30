import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import py_sod_metrics
import torch
import torch.nn.functional as F
import torch.optim as opt
from scipy import ndimage
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from SAM2UNet import SAM2UNet
from dataset import FullDataset, TestDataset


def compute_boundary_loss(pred, mask):
    """
    Compute boundary loss using distance transform
    Minimal implementation for immediate boundary improvement
    """
    batch_size = mask.shape[0]
    boundary_loss = 0

    for i in range(batch_size):
        # Convert to numpy for distance transform
        mask_np = mask[i, 0].cpu().numpy().astype(np.uint8)

        # Compute distance transform of the boundary
        # Distance from each pixel to nearest boundary
        distance_map = ndimage.distance_transform_edt(mask_np)

        # Create boundary weight map (higher weight near boundaries)
        boundary_weight = np.exp(-distance_map / 5.0)  # 5.0 is sigma parameter
        boundary_weight = torch.from_numpy(boundary_weight).float().to(pred.device)

        # Apply boundary-weighted BCE
        pred_sigmoid = torch.sigmoid(pred[i, 0])
        boundary_bce = F.binary_cross_entropy(pred_sigmoid, mask[i, 0].float(), reduction='none')
        weighted_boundary_loss = (boundary_bce * boundary_weight).mean()
        boundary_loss += weighted_boundary_loss

    return boundary_loss / batch_size


def focal_loss(pred, mask, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance and hard examples
    Reduces false positives by focusing on difficult cases
    """
    pred_sigmoid = torch.sigmoid(pred)

    # Compute focal weight
    pt = torch.where(mask == 1, pred_sigmoid, 1 - pred_sigmoid)
    focal_weight = alpha * (1 - pt)**gamma

    # Compute focal loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    focal = focal_weight * bce_loss

    return focal.mean()


def enhanced_structure_loss(pred, mask, boundary_weight=0.3, focal_weight=0.5):
    """
    Enhanced structure loss with boundary and focal components
    Minimal change to existing structure_loss for easy A/B testing

    Args:
        pred: predicted logits
        mask: ground truth mask
        boundary_weight: weight for boundary loss component (0.3 recommended)
        focal_weight: weight for focal loss component (0.5 recommended)
    """
    # Original structure loss components (unchanged)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # Original loss
    original_loss = (wbce + wiou).mean()

    # NEW: Add boundary loss component
    boundary_loss = compute_boundary_loss(pred, mask)

    # NEW: Add focal loss component
    focal_loss_val = focal_loss(pred, mask)

    # Combine all components
    total_loss = original_loss + boundary_weight * boundary_loss + focal_weight * focal_loss_val

    return total_loss, {
        'original_loss': original_loss.item(),
        'boundary_loss': boundary_loss.item(),
        'focal_loss': focal_loss_val.item(),
        'total_loss': total_loss.item()
    }


# Alternative GPU-accelerated boundary loss (if scipy is too slow)
def gpu_boundary_loss(pred, mask, sigma=5.0):
    """
    GPU-accelerated approximation of boundary loss
    Uses morphological operations instead of distance transform
    """
    # Create boundary map using morphological operations
    kernel = torch.ones(3, 3, device=mask.device)
    dilated = F.conv2d(mask.float(), kernel.unsqueeze(0).unsqueeze(0), padding=1)
    eroded = F.conv2d(mask.float(), -kernel.unsqueeze(0).unsqueeze(0), padding=1) + 9
    boundary_map = (dilated > 0).float() - (eroded >= 9).float()

    # Create distance-like weights
    boundary_weight = torch.where(boundary_map > 0, 2.0, 1.0)

    # Apply weighted BCE
    pred_sigmoid = torch.sigmoid(pred)
    bce_loss = F.binary_cross_entropy(pred_sigmoid, mask.float(), reduction='none')
    weighted_loss = (bce_loss * boundary_weight).mean()

    return weighted_loss


def enhanced_structure_loss_gpu(pred, mask, boundary_weight=0.3, focal_weight=0.5):
    """
    GPU-only version of enhanced structure loss (faster training)
    Use this if the scipy version is too slow during training
    """
    # Original structure loss components
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    original_loss = (wbce + wiou).mean()

    # GPU-accelerated boundary loss
    boundary_loss = gpu_boundary_loss(pred, mask)

    # Focal loss
    focal_loss_val = focal_loss(pred, mask)

    # Combine components
    total_loss = original_loss + boundary_weight * boundary_loss + focal_weight * focal_loss_val

    return total_loss, {
        'original_loss': original_loss.item(),
        'boundary_loss': boundary_loss.item(),
        'focal_loss': focal_loss_val.item(),
        'total_loss': total_loss.item()
    }


# Simple post-processing function for immediate false positive reduction
def post_process_predictions(pred_tensor, min_area=50):
    """
    Simple post-processing to remove small false positive regions
    Apply this to your predictions during evaluation
    """
    import cv2

    pred_np = (pred_tensor.cpu().numpy() * 255).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_np)

    # Remove small components
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            pred_np[labels == i] = 0

    return torch.from_numpy(pred_np.astype(np.float32) / 255.0).to(pred_tensor.device)


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--save_path", type=str, required=False,
                    default="playground/exp_960size_enhanced_losses_composite_score_lr_0001_350e",
                    help="path to store the checkpoint")
parser.add_argument(
    "--checkpoint", type=str, default="", help="path to the checkpoint of sam2-unet"
)
parser.add_argument(
    "--train_image_path",
    type=str,
    default="../wall_seg_crop/data_train/images/",
    help="path to the image that used to train the model",
)
parser.add_argument(
    "--train_mask_path",
    type=str,
    default="../wall_seg_crop/data_train/masks/",
    help="path to the mask file for training",
)
parser.add_argument(
    "--test_image_path",
    type=str,
    default="../wall_seg_crop/data_test/images/",
    help="path to the image that used to evaluate the model",
)
parser.add_argument(
    "--test_gt_path",
    type=str,
    default="../wall_seg_crop/data_test/masks/",
    help="path to the mask file for evaluating",
)
parser.add_argument("--epoch", type=int, default=500, help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--size", default=960, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--save_interval", default=20, type=int)
parser.add_argument("--base_score", default=0.75, type=float)

# NEW: Enhanced loss parameters
parser.add_argument("--boundary_weight", default=0.3, type=float, help="Weight for boundary loss")
parser.add_argument("--focal_weight", default=0.5, type=float, help="Weight for focal loss")
parser.add_argument("--use_enhanced_loss", action='store_true', help="Use enhanced loss function")
parser.add_argument("--post_process", action='store_true', help="Apply post-processing")

args = parser.parse_args()

# Easy A/B testing flag
USE_ENHANCED_LOSS = args.use_enhanced_loss  # or set to True/False directly


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def create_visualization(image, gt, pred, pred_processed, name, save_path, checkpoint_name):
    """
    Create and save visualization showing original image, GT, prediction, and processed prediction
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    if len(image.shape) == 3 and image.shape[0] == 3:  # CHW format
        img_display = np.transpose(image, (1, 2, 0))
    else:
        img_display = image

    # Normalize image for display if needed
    if img_display.max() <= 1.0:
        img_display = (img_display * 255).astype(np.uint8)

    axes[0].imshow(img_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Original prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Post-processed prediction
    if pred_processed is not None:
        axes[3].imshow(pred_processed, cmap='gray')
        axes[3].set_title('Post-processed')
    else:
        axes[3].imshow(pred, cmap='gray')
        axes[3].set_title('Prediction (no post-proc)')
    axes[3].axis('off')

    # Create filename
    image_name_clean = os.path.splitext(name)[0]  # Remove extension
    filename = f"{checkpoint_name}_{image_name_clean}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_loss_metrics(metrics_history, save_path, epoch):
    """
    Save loss component evolution to understand training dynamics
    """
    if not metrics_history:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = list(range(len(metrics_history)))

    # Extract metrics
    original_losses = [m['original_loss'] for m in metrics_history]
    boundary_losses = [m['boundary_loss'] for m in metrics_history]
    focal_losses = [m['focal_loss'] for m in metrics_history]
    total_losses = [m['total_loss'] for m in metrics_history]

    axes[0, 0].plot(epochs, original_losses, label='Original Loss')
    axes[0, 0].set_title('Original Structure Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, boundary_losses, label='Boundary Loss', color='red')
    axes[0, 1].set_title('Boundary Loss Component')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, focal_losses, label='Focal Loss', color='green')
    axes[1, 0].set_title('Focal Loss Component')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, total_losses, label='Total Enhanced Loss', color='purple')
    axes[1, 1].set_title('Total Enhanced Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'loss_evolution_epoch_{epoch}.png'), dpi=150,
                bbox_inches='tight')
    plt.close()


# Define eval metrics
sample_gray = dict(with_adaptive=True, with_dynamic=True)


def main(args):
    # 1. Load train data
    dataset = FullDataset(args.train_image_path, args.train_mask_path, args.size, mode="train")
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
    optim = opt.AdamW(
        [{"params": model.parameters(), "initia_lr": args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # 6. Set scheduler
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    # 7. Train
    os.makedirs(args.save_path, exist_ok=True)

    # NEW: Create visualization directory
    vis_path = os.path.join(args.save_path, "vis")
    os.makedirs(vis_path, exist_ok=True)

    # NEW: Track loss metrics for enhanced loss
    metrics_history = []

    epoch_loss = 2.0
    base_score = args.base_score
    save_interval = args.save_interval

    for epoch in range(args.epoch):
        # 7.1. Train phase
        print("Training:")
        model.train()  # Set model to training mode
        epoch_metrics = []

        for i, batch in enumerate(dataloader):
            x = batch["image"]
            target = batch["label"]
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)

            # ENHANCED LOSS INTEGRATION
            if USE_ENHANCED_LOSS:
                loss0, metrics0 = enhanced_structure_loss_gpu(pred0, target, args.boundary_weight,
                                                              args.focal_weight)
                loss1, metrics1 = enhanced_structure_loss_gpu(pred1, target, args.boundary_weight,
                                                              args.focal_weight)
                loss2, metrics2 = enhanced_structure_loss_gpu(pred2, target, args.boundary_weight,
                                                              args.focal_weight)
                loss = loss0 + loss1 + loss2

                # Store metrics for this batch
                epoch_metrics.append(metrics0)  # Use metrics from first prediction head

                # Optional: Log detailed metrics every 10 iterations
                if i % 10 == 0:
                    print(f"epoch-{epoch + 1}-{i + 1}: Enhanced loss:{loss.item():.4f} "
                          f"(Orig: {metrics0['original_loss']:.4f}, "
                          f"Boundary: {metrics0['boundary_loss']:.4f}, "
                          f"Focal: {metrics0['focal_loss']:.4f})")
            else:
                # Original loss for comparison
                loss0 = structure_loss(pred0, target)
                loss1 = structure_loss(pred1, target)
                loss2 = structure_loss(pred2, target)
                loss = loss0 + loss1 + loss2

                if i % 10 == 0:
                    print(f"epoch-{epoch + 1}-{i + 1}: Original loss:{loss.item():.4f}")

            epoch_loss = loss.item()
            loss.backward()
            optim.step()

        scheduler.step()

        # NEW: Average epoch metrics for enhanced loss
        if USE_ENHANCED_LOSS and epoch_metrics:
            avg_metrics = {
                'original_loss': np.mean([m['original_loss'] for m in epoch_metrics]),
                'boundary_loss': np.mean([m['boundary_loss'] for m in epoch_metrics]),
                'focal_loss': np.mean([m['focal_loss'] for m in epoch_metrics]),
                'total_loss': np.mean([m['total_loss'] for m in epoch_metrics])
            }
            metrics_history.append(avg_metrics)

        # 7.2. Evaluation phase
        print("Evaluating", end="")
        # init FMv2
        FMv2 = py_sod_metrics.FmeasureV2(
            metric_handlers={
                "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
                "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
                "iou": py_sod_metrics.IOUHandler(**sample_gray),
            }
        )
        # Set model to evaluation mode
        model.eval()

        # NEW: Create checkpoint name for visualization filenames
        checkpoint_name = f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}"

        # Disable gradient calculations for efficiency and safety
        for i in range(test_loader.size):
            with torch.no_grad():
                image, gt, name, padding = test_loader.load_data()

                # Store original image for visualization
                original_image = image.clone()

                image = image.to(device)
                res_padded, _, _ = model(image)
                pad_left, pad_top, pad_right, pad_bottom = padding
                res = res_padded[
                      :, :, pad_top: args.size - pad_bottom, pad_left: args.size - pad_right
                      ]
                res = F.interpolate(res, size=gt.shape, mode="bilinear", align_corners=False)
                res = res.sigmoid().data.cpu()
                res = res.numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                # Store original prediction for visualization
                res_original = res.copy()

                # NEW: Add post-processing option
                res_processed = None
                if args.post_process:
                    # Convert to tensor for post-processing
                    res_tensor = torch.from_numpy(res)
                    res_processed_tensor = post_process_predictions(res_tensor, min_area=50)
                    res_processed = res_processed_tensor.numpy()
                    res = res_processed  # Use processed version for metrics

                res = (res * 255).astype(np.uint8)
                gt = np.asarray(gt, np.float32)

                # NEW: Create visualizations for first 10 images
                if i < 10:
                    # Prepare image for visualization
                    vis_image = original_image.squeeze().cpu().numpy()

                    # Convert predictions back to 0-1 range for visualization
                    vis_pred = res_original
                    vis_pred_processed = res_processed if res_processed is not None else None

                    create_visualization(
                        image=vis_image,
                        gt=gt,
                        pred=vis_pred,
                        pred_processed=vis_pred_processed,
                        name=name,
                        save_path=vis_path,
                        checkpoint_name=checkpoint_name,
                    )

                # Evaluate
                FMv2.step(pred=res, gt=gt)
                # Print for status
                if i % 10 == 0:
                    print(".", end="", flush=True)

        # Reset test_loader index
        test_loader.reset_index()
        # Get mIoU
        fmv2 = FMv2.get_results()

        # Combine multiple metrics with weights
        boundary_f03 = fmv2["fm"]["dynamic"].mean()  # Boundary quality
        overall_iou = fmv2["iou"]["dynamic"].mean()  # Region quality
        precision = fmv2["pre"]["dynamic"].mean()  # False positive control

        # Weighted combination
        composite_score = 0.4 * boundary_f03 + 0.4 * overall_iou + 0.2 * precision

        # NEW: Save loss evolution plot for enhanced loss
        if USE_ENHANCED_LOSS and metrics_history:
            save_loss_metrics(metrics_history, vis_path, epoch + 1)

        print(
            "\nepoch-{}: loss: {} score: {} best_score: {}\n".format(
                epoch + 1, epoch_loss, composite_score, base_score
            )
        )

        # 7.3. Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == args.epoch:
            save_model_path = os.path.join(
                args.save_path,
                f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_score-{composite_score:.3f}.pth",
            )
            torch.save(model.state_dict(), save_model_path)
            print("Saving Snapshot:", save_model_path)
        elif composite_score > base_score:
            base_score = composite_score
            save_model_path = os.path.join(
                args.save_path,
                f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_score-{composite_score:.3f}.pth",
            )
            torch.save(model.state_dict(), save_model_path)
            print("Saving Snapshot best:", save_model_path)


if __name__ == "__main__":
    main(args)
