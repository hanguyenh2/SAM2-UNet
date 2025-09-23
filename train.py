import argparse
import os

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
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--hiera_path", type=str,
                    default="",
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--checkpoint", type=str,
                    default="",
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--train_image_path", type=str,
                    default="../wall_seg_crop/data_train/images/",
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str,
                    default="../wall_seg_crop/data_train/masks/",
                    help="path to the mask file for training")
parser.add_argument("--test_image_path", type=str,
                    default="../wall_seg_crop/data_test/images/",
                    help="path to the image that used to evaluate the model")
parser.add_argument("--test_gt_path", type=str,
                    default="../wall_seg_crop/data_test/masks/",
                    help="path to the mask file for evaluating")
parser.add_argument("--epoch", type=int, default=1000, help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--size", default=960, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--save_interval", default=10, type=int)
parser.add_argument("--base_mean_iou", default=0.8, type=float)
parser.add_argument("--auto_save_iou", default=0.84, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 10 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# Define eval metrics
sample_gray = dict(with_adaptive=True, with_dynamic=True)


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
    model = SAM2UNet(checkpoint_path=args.hiera_path).to(device)

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
    base_mean_iou = args.base_mean_iou
    auto_save_iou = args.auto_save_iou
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
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
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
                "iou": py_sod_metrics.IOUHandler(**sample_gray),
            }
        )
        # Set model to evaluation mode
        model.eval()
        # Disable gradient calculations for efficiency and safety
        for i in range(test_loader.size):
            with torch.no_grad():
                image, gt, name, padding = test_loader.load_data()
                image = image.to(device)
                res_padded, _, _ = model(image)
                pad_left, pad_top, pad_right, pad_bottom = padding
                res = res_padded[:, :, pad_top: args.size - pad_bottom,
                      pad_left: args.size - pad_right]
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
                                                                   base_mean_iou))

        # 7.3. Save checkpoint
        if mean_iou > base_mean_iou:
            base_mean_iou = mean_iou
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print(f'Saving Snapshot best:', save_model_path)
        elif mean_iou > auto_save_iou:
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print(f'Auto Saving Good Snapshot:', save_model_path)
        elif (epoch + 1) % save_interval == 0 or (epoch + 1) == args.epoch:
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print('Saving Snapshot:', save_model_path)


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
