import argparse
import os

import torch

from SAM2UNet import SAM2UNet

# 1. Define parser
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
args = parser.parse_args()

# 2. Set device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Save model
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
model_path = f"{checkpoint_name}.pth"
torch.save(model, model_path)