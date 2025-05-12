import argparse
import os
import time

import cv2
import imageio
import numpy as np
import onnxruntime as ort
import torch

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
        image = image.cpu().numpy()
        time_start = time.time()
        # 3. Run the model
        res, _, _ = ort_session.run(None, {input_name: image})

        process_time = time.time() - time_start
        test_time.append(process_time)

        gt = np.asarray(gt, np.float32)
        gt_shape = gt.shape
        print(res.shape)
        res_sigmoid = 1 / (1 + np.exp(-res))
        res = np.squeeze(res_sigmoid)
        res = cv2.resize(res, (gt_shape[-1], gt_shape[-2]), interpolation=cv2.INTER_LINEAR)
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print(res.shape)
        print("Saving " + name)
        print("process_time:", process_time)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)

print("test_time:", np.mean(test_time))
