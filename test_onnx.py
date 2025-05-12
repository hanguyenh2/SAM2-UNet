import argparse
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from SAM2UNet import SAM2UNet
from dataset import TestDataset

import onnxruntime as ort
import numpy as np

def use_onnx_model(onnx_path, input_data):
    """
    Loads an ONNX model and performs inference.

    Args:
        onnx_path (str): Path to the ONNX model file.
        input_data (numpy.ndarray):  The input data for the model, as a NumPy array.
            The shape and data type must match the model's expected input.

    Returns:
        numpy.ndarray: The model's output.
    """
    # 1. Create an ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # 2. Get the name of the input layer
    input_name = ort_session.get_inputs()[0].name

    # 3. Run the model
    outputs = ort_session.run(None, {input_name: input_data})

    # 4.  Get the output (assuming there's one output)
    output = outputs[0]  # outputs is a list;  take the first element.
    return output

def main():
    """Example usage."""
    # 1. Path to your ONNX model (the one you converted)
    onnx_model_path = "resnet18.onnx"  # Replace with the actual path to your ONNX file

    # 2.  Create some dummy input data  (replace with *your* actual data!)
    #    This is *CRUCIAL*:  The shape and type MUST match what the model expects.
    #    For example, if your model expects a batch of 1 image, 3 channels, 224x224:
    input_shape = (1, 3, 224, 224)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # 3. Use the model for inference
    output_data = use_onnx_model(onnx_model_path, input_data)

    # 4. Process the output (this depends on what your model does)
    print("Model output shape:", output_data.shape)
    print("Model output data:", output_data)
    #  Example:  If it's a classification model, you might take the argmax:
    #  predicted_class = np.argmax(output_data, axis=1)
    #  print("Predicted class:", predicted_class)

if __name__ == "__main__":
    main()

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
        time_start = time.time()
        # 3. Run the model
        res, _, _ = ort_session.run(None, {input_name: image})

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
