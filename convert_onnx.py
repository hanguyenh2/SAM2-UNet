import argparse
import os

import torch
import torch.onnx

from SAM2UNet import SAM2UNet


def convert_pth_to_onnx(model, dummy_input, onnx_path, verbose=False):
    """
    Converts a PyTorch .pth model to the ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model (loaded from .pth).
        dummy_input (torch.Tensor): A dummy input tensor with the correct shape
            for the model.  This is used to trace the model's execution.
        onnx_path (str): The path where the ONNX file should be saved (e.g., "model.onnx").
        verbose (bool, optional):  If True, prints detailed information during the
            conversion process. Defaults to False.
    """
    # 1. Check if the model is in eval mode
    if model.training:
        model.eval()  # Set to evaluation mode for export
        print("Model set to evaluation mode for ONNX export.")

    # 2. Export the model to ONNX
    try:
        input_names = ["input_0"]
        output_names = ["output_0","output_1","output_2"]
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,  # Store trained parameter weights inside the ONNX file
            opset_version=10,  # Or a suitable opset version (check compatibility)
            do_constant_folding=True,  # Optimize the graph by folding constants
            input_names=input_names,  # Name for the input tensor in the ONNX graph
            output_names=output_names,  # Name for the output tensor in the ONNX graph
            verbose=verbose,  # Print detailed information during export
        )
        print(f"Model successfully exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return False  # Indicate failure

    return True  # Indicate success


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--size", default=1536, type=int)
args = parser.parse_args()

# 1. Load your PyTorch model from .pth (replace with your actual model loading)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()

# 2. Create a dummy input tensor with the correct shape
#  This should match the input size of your model.
# 1 batch, 3 channels, 1536x1536 image
dummy_input = torch.randn(1, 3, args.size, args.size).cuda()

# 3. Define the output path for the ONNX file
checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
onnx_path = f"{checkpoint_name}.onnx"  # Or "your_model.onnx"
print("Converting to ONNX:", onnx_path)

# 4. Convert the model
# Set verbose=True for debugging
success = convert_pth_to_onnx(model, dummy_input, onnx_path, verbose=True)
