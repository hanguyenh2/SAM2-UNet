import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from SAM2UNet import SAM2UNet
from dataset import FullDataset

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
parser.add_argument("--test_mask_path", type=str,
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


def main(args):
    # 1. Load train data
    dataset = FullDataset(args.train_image_path, args.train_mask_path, args.size, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # 2. Load test data
    test_dataset = FullDataset(args.test_image_path, args.train_mask_path, args.size, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=8)
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
    best_loss = 0.4
    save_interval = args.save_interval
    for epoch in range(args.epoch):
        # 7.1. Train phase
        print("Training:")
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
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch-{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
        scheduler.step()

        # 7.2. Evaluation phase
        print("Evaluating ", end="")
        for i, batch in enumerate(test_dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            epoch_loss = loss.item()
            if i % 10 == 0:
                print(".", end="")
        print("\nepoch-{}: loss:{} best_loss:{}\n".format(epoch + 1, epoch_loss, best_loss))

        # 7.3. Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == args.epoch:
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet-{epoch + 1}-{epoch_loss:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print('[Saving Snapshot:]', save_model_path)

        # 7.3. Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model_path = os.path.join(args.save_path,
                                           f"SAM2-UNet-{epoch + 1}-{epoch_loss:.3f}.pth")
            torch.save(model.state_dict(), save_model_path)
            print(f'[Saving Snapshot best: {epoch + 1}-{epoch_loss:.3f}]', save_model_path)


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
