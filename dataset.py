import os
import random

import numpy as np
import torch  # Import torch for dtype conversions
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ToTensor(object):
    """
    Converts PIL Images in the 'image' and 'label' fields to PyTorch Tensors.
    - Image tensor will be float32 (normalized to 0-1).
    - Label tensor will be torch.long (for integer class IDs).
    """

    def __call__(self, data):
        image, label = data['image'], data['label']

        # Image: Convert to float32 tensor, normalized to [0, 1]
        image_tensor = F.to_tensor(image)

        # Label (Mask):
        # For multi-class segmentation, labels should be integer class IDs (e.g., 0, 1, 2...).
        # PIL's 'L' mode gives pixel values from 0-255. When converted to a tensor,
        # F.to_tensor would typically scale this to [0,1] if it detects an image-like range.
        # To ensure we get the raw integer class IDs, we convert the PIL image to a
        # NumPy array first, ensuring its integer dtype, then to a PyTorch tensor of type torch.long.
        label_numpy = np.array(label, dtype=np.int64)  # Ensure integer type for class IDs
        label_tensor = torch.from_numpy(label_numpy).long()  # Convert to torch.long (int64)

        # Most segmentation models expect labels as (H, W) or (N, H, W) (no channel dimension).
        # If your model expects a channel dimension (e.g., (1, H, W)) for labels,
        # you might need to uncomment: label_tensor = label_tensor.unsqueeze(0)
        # However, it's usually (N, C, H, W) for input images and (N, H, W) for labels.
        # So, leaving it as (H, W) after numpy conversion is typically correct.

        return {'image': image_tensor, 'label': label_tensor}


class ResizeLongestSideAndPad(object):
    """
    Resizes the image and label such that the longest side matches `size`,
    maintaining aspect ratio, and then pads the shorter side with zeros
    to make the image square.

    Handles multi-channel images and single-channel (torch.long) labels.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): The target size for the longest side.
                        The output image/label will be (size, size).
        """
        self.size = size

    def __call__(self, data: dict) -> dict:
        image, label = data['image'], data['label']

        # Get H, W from the last two dimensions (works for C, H, W image and H, W label)
        original_h, original_w = image.shape[-2:]

        # Calculate the scaling factor
        scale_factor = self.size / max(original_h, original_w)

        # Calculate new dimensions while maintaining aspect ratio
        new_h = int(round(original_h * scale_factor))
        new_w = int(round(original_w * scale_factor))

        # Resize image and label
        # For image: use bilinear or bicubic for quality
        resized_image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        # Label (mask): use nearest neighbor to preserve discrete integer class values.
        # Unsqueeze/squeeze to handle F.resize expecting 3D tensor (C,H,W).
        resized_label = F.resize(label.unsqueeze(0), [new_h, new_w],
                                 interpolation=InterpolationMode.NEAREST).squeeze(0)

        # Calculate padding amounts
        pad_h = self.size - new_h
        pad_w = self.size - new_w

        # Determine padding for top/bottom and left/right to center the image
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = [pad_left, pad_top, pad_right, pad_bottom]  # (left, top, right, bottom)

        # Apply padding
        # For image: pad with 0 (black). If image is normalized *before* this, consider padding with mean.
        padded_image = F.pad(resized_image, padding, fill=0)

        # For label: pad with 0 (background class).
        # Labels should remain torch.long after padding.
        # Unsqueeze/squeeze to handle F.pad expecting 3D tensor (C,H,W).
        padded_label = F.pad(resized_label.unsqueeze(0), padding, fill=0).squeeze(0)

        return {'image': padded_image, 'label': padded_label}


class Normalize(object):
    """
    Normalizes the image tensor using mean and standard deviation.
    Label tensor is passed through unchanged.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class RandomRotate(object):
    """
    Randomly rotates the image and label by 0, 90, 180, or 270 degrees.
    Image uses bilinear interpolation. Label uses nearest neighbor.
    """

    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        # Randomly choose rotation (0, 90, 180, 270 degrees)
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            # Image: bilinear interpolation for continuous pixel values
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, expand=False)
            # Label: nearest neighbor interpolation to preserve discrete class IDs
            # Unsqueeze/squeeze to handle F.rotate expecting 3D tensor (C,H,W).
            label = F.rotate(label.unsqueeze(0), angle, interpolation=InterpolationMode.NEAREST,
                             expand=False).squeeze(0)
        return {'image': image, 'label': label}


class ToGray(object):
    """
    Converts a 3-channel image to grayscale with a given probability.
    Useful for data augmentation. Label is unaffected.
    """

    def __init__(self, p=0.5, num_output_channels=3):
        """
        Args:
            p (float): Probability of applying the transform.
            num_output_channels (int): Number of output channels (1 or 3).
                                       Typically 3 to keep input shape consistent for models
                                       even after grayscaling.
        """
        self.p = p
        self.num_output_channels = num_output_channels

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            # F.rgb_to_grayscale expects a torch.Tensor (float, CHW)
            # Make sure this transform is applied AFTER ToTensor
            image = F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)

        return {'image': image, 'label': label}


class ColorAugmentations(object):
    """
    Applies a random color augmentation (brightness, contrast, saturation, hue, gamma)
    with a given probability. Label is unaffected.
    """

    def __init__(self, p=0.7):
        self.p = p
        # Define ranges for each type of color jitter.
        # These ranges are common defaults or similar to Albumentations defaults.
        self.brightness_range = (0.8, 1.2)  # For RandomBrightnessContrast
        self.contrast_range = (0.8, 1.2)  # For RandomBrightnessContrast
        self.saturation_range = (0.8, 1.2)  # For HueSaturationValue, ColorJitter
        self.hue_range = (-0.1, 0.1)  # For HueSaturationValue, ColorJitter (-0.5 to 0.5)
        self.gamma_range = (0.7, 1.3)  # For RandomGamma

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            # Randomly choose one of the four color augmentations
            # 0: BrightnessContrast, 1: ColorJitter (all), 2: HueSaturation, 3: Gamma
            choice = random.randint(0, 3)

            if choice == 0:  # RandomBrightnessContrast
                brightness_factor = random.uniform(*self.brightness_range)
                contrast_factor = random.uniform(*self.contrast_range)
                image = F.adjust_brightness(image, brightness_factor)
                image = F.adjust_contrast(image, contrast_factor)
            elif choice == 1:  # ColorJitter (all components)
                # Generate factors for all four: brightness, contrast, saturation, hue
                brightness_factor = random.uniform(*self.brightness_range)
                contrast_factor = random.uniform(*self.contrast_range)
                saturation_factor = random.uniform(*self.saturation_range)
                hue_factor = random.uniform(*self.hue_range)
                image = F.adjust_brightness(image, brightness_factor)
                image = F.adjust_contrast(image, contrast_factor)
                image = F.adjust_saturation(image, saturation_factor)
                image = F.adjust_hue(image, hue_factor)
            elif choice == 2:  # HueSaturationValue
                saturation_factor = random.uniform(*self.saturation_range)
                hue_factor = random.uniform(*self.hue_range)
                image = F.adjust_saturation(image, saturation_factor)
                image = F.adjust_hue(image, hue_factor)
            elif choice == 3:  # RandomGamma
                gamma_value = random.uniform(*self.gamma_range)
                image = F.adjust_gamma(image, gamma_value)
        return {'image': image, 'label': label}


class GaussianBlur(object):
    """
    Applies Gaussian blur to the image with a given probability. Label is unaffected.
    """

    def __init__(self, p=0.3, blur_limit=(3, 5)):
        self.p = p
        # Ensure blur_limit contains only odd integers for kernel_size
        self.kernel_sizes = [k for k in range(blur_limit[0], blur_limit[1] + 1) if k % 2 == 1]
        if not self.kernel_sizes:
            raise ValueError("blur_limit must contain at least one odd integer for kernel_size.")

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            # Randomly choose a kernel size from the allowed odd integers
            kernel_size = random.choice(self.kernel_sizes)

            # F.gaussian_blur expects a torch.Tensor (float, CHW)
            # Its kernel_size argument can be a single integer or a tuple (height, width).
            # For a square kernel, pass (kernel_size, kernel_size).
            image = F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    """
    Dataset class for loading image and segmentation labels during training/validation.
    Supports multi-class segmentation by ensuring labels are loaded as integer class IDs (torch.long).
    """

    def __init__(self, image_root: str, gt_root: str, size: int, mode: str = "train"):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]

        # It's crucial that image and GT filenames match if not explicitly paired.
        # This sorts them to increase the likelihood of correct pairing.
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # Basic check if number of images and GTs are the same.
        # For robust pipelines, consider matching based on base filenames (e.g., 'image_001.png' and 'image_001_mask.png').
        if len(self.images) != len(self.gts):
            print(
                f"Warning: Number of images ({len(self.images)}) does not match number of GTs ({len(self.gts)}). "
                "Ensure image and ground truth files are correctly paired by sorting or naming convention.")

        if mode == 'train':
            self.transform = transforms.Compose([
                ToTensor(),  # Converts to Tensor, label to torch.long
                ResizeLongestSideAndPad(size),  # Resizes and pads, label nearest
                RandomRotate(),  # Rotates, label nearest
                ToGray(),  # Grayscales image, label unaffected
                ColorAugmentations(),  # Augments image colors, label unaffected
                GaussianBlur(),  # Blurs image, label unaffected
                Normalize()  # Normalizes image, label unaffected
            ])
        else:  # 'eval' or 'test' mode (usually less augmentation)
            self.transform = transforms.Compose([
                ToTensor(),
                ResizeLongestSideAndPad(size),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.multiclass_loader(self.gts[idx])  # Use the dedicated multiclass loader
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        """Loads an RGB image from path."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def multiclass_loader(self, path):
        """
        Loads a multi-class segmentation mask as a PIL Image in 'L' (grayscale) mode.
        Assumes pixel values directly correspond to class IDs (e.g., 0 for background, 1 for class A, etc.).
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            # 'L' mode (8-bit pixels, 0-255) is suitable if pixel values directly represent class IDs.
            # Make sure your masks indeed use these integer pixel values for different classes.
            return img.convert('L')


class ImageToTensor(object):
    """
    Converts PIL Image to PyTorch Tensor (float32).
    Used in test dataset where only image is processed for inference.
    """

    def __call__(self, data):
        image = data['image']
        return {'image': F.to_tensor(image)}


class LongestMaxSizeAndPad(object):
    """
    Resizes the image such that the longest side matches `size`,
    maintaining aspect ratio, and then pads the shorter side with zeros
    to make the image square. Returns padding information, useful for unpadding predictions.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): The target size for the longest side.
                        The output image will be (size, size).
        """
        self.size = size

    def __call__(self, data: dict) -> dict:
        image = data['image']
        # Make sure image and label are PyTorch Tensors
        # This assumes image is (C, H, W) and label is (H, W) or (1, H, W)
        original_h, original_w = image.shape[-2:]  # Get H, W from (C, H, W)

        # Calculate the scaling factor
        scale_factor = self.size / max(original_h, original_w)

        # Calculate new dimensions while maintaining aspect ratio
        new_h = int(round(original_h * scale_factor))
        new_w = int(round(original_w * scale_factor))

        # Resize image and label
        # For image: use bilinear or bicubic for quality
        resized_image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

        # Calculate padding amounts
        pad_h = self.size - new_h
        pad_w = self.size - new_w

        # Determine padding for top/bottom and left/right to center the image
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = [pad_left, pad_top, pad_right, pad_bottom]

        # Apply padding
        # For image: pad with 0 (black) or mean pixel value if normalized
        padded_image = F.pad(resized_image, padding, fill=0)  # fill=0 for black padding

        return {'image': padded_image, 'padding': padding}


class NormalizeImage(object):
    """
    Normalizes the image tensor using mean and standard deviation.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, padding = sample['image'], sample['padding']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'padding': padding}


class TestDataset:
    """
    Dataset class for loading test images and their ground truth masks (for evaluation).
    The ground truth masks are loaded as NumPy arrays for easier processing during evaluation,
    and are multi-class compatible.
    """

    def __init__(self, image_root: str, gt_root: str, size: int):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if len(self.images) != len(self.gts):
            print(
                f"Warning: Number of images ({len(self.images)}) does not match number of GTs ({len(self.gts)}). "
                "Ensure image and ground truth files are correctly paired by sorting or naming convention.")

        self.transform = transforms.Compose([
            ImageToTensor(),  # Converts image to tensor
            LongestMaxSizeAndPad(size),  # Resizes and pads image
            NormalizeImage()  # Normalizes image
        ])
        self.size = len(self.images)
        self.index = 0

    def reset_index(self):
        """Resets the dataset index to 0 for re-iteration."""
        self.index = 0

    def load_data(self):
        """
        Loads the next image and its corresponding ground truth mask for testing/evaluation.
        Returns image tensor (ready for model input), ground truth numpy array, filename, and padding info.
        """
        if self.index >= self.size:
            raise IndexError("No more data to load. Reset index or check dataset size.")

        image = self.rgb_loader(self.images[self.index])
        data = {'image': image}
        data = self.transform(data)

        # Add batch dimension for model input (e.g., from (C,H,W) to (1,C,H,W))
        image_tensor = data["image"].unsqueeze(0)
        padding = data["padding"]

        # Load ground truth for evaluation. It should be a multi-class integer mask.
        # It's loaded as a PIL 'L' image, then converted to a NumPy array.
        gt_mask_pil = self.multiclass_loader(self.gts[self.index])  # Use multiclass_loader
        gt_mask_numpy = np.array(gt_mask_pil, dtype=np.int64)  # Ensure integer type for class IDs

        name = os.path.basename(
            self.images[self.index])  # Get filename without path for convenience

        self.index += 1
        return image_tensor, gt_mask_numpy, name, padding

    def rgb_loader(self, path):
        """Loads an RGB image from path."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def multiclass_loader(self, path):
        """
        Loads a multi-class segmentation mask for evaluation.
        Assumes the mask is a grayscale image where pixel values directly
        correspond to class IDs (e.g., 0 for background, 1 for class A, 2 for class B, etc.).
        Returns as a PIL Image in 'L' mode.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_image_path = "./data_mini/data_train/images/"
    train_mask_path = "./data_mini/data_train/masks/"
    size = 960
    batch_size = 1
    dataset = FullDataset(train_image_path, train_mask_path, size, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    for i, batch in enumerate(dataloader):
        x = batch['image']
        target = batch['label'].squeeze().numpy()
        print("==============")
        print(np.min(target))
        print(np.max(target))
        print(np.unique(target))
        print(np.shape(target))
