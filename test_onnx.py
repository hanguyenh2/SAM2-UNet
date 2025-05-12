import argparse
import os
import time

import imageio
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from dataset import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
parser.add_argument("--size", default=1536, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, args.size)
# 1. Create an ONNX Runtime session
ort_session = ort.InferenceSession(args.checkpoint)
# 2. Get the name of the input layer
input_name = ort_session.get_inputs()[0].name
os.makedirs(args.save_path, exist_ok=True)
test_time = []
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        print("=========")
        print(image.shape)
        image_ = torch.randn(1, 3, args.size, args.size).cuda()
        print(image.shape)
        time_start = time.time()
        # 3. Run the model
        res, _, _ = ort_session.run(None, {input_name: image_})

        process_time = time.time() - time_start
        test_time.append(process_time)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        # If you want to binarize the prediction results, please uncomment the following three lines. 
        # Note that this action will affect the calculation of evaluation metrics.
        # lambda = 0.5
        # res[res >= int(255 * lambda)] = 255
        # res[res < int(255 * lambda)] = 0
        print("Saving " + name)
        print("process_time:", process_time)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)

print("test_time:", np.mean(test_time))
