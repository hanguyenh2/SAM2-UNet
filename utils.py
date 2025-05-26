import cv2
import numpy as np


def get_color_by_cate_id(category_id: int) -> tuple:
    # Generate unique color for each category
    color = (
        int((category_id * 123 + 100) % 255),
        int((category_id * 100 + 200) % 255),
        int((category_id * 200 + 50) % 255),
    )
    return color


def colorize_mask(
    mask: np.ndarray,
    class_num: int,
) -> np.ndarray:
    """Visualize mask"""
    # 1. Create an empty 3-channel image for the colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # 2. Iterate through each class ID and apply its color
    # This can be slow for very large masks, but is explicit
    for class_id in range(class_num):
        # 2.1. Find all pixels belonging to this class_id
        class_pixels = mask == class_id + 1
        # 2.2. Apply the corresponding color
        colored_mask[class_pixels] = get_color_by_cate_id(class_id)

    return colored_mask


def visualize_mask(
    original_img: np.ndarray,
    mask: np.ndarray,
    class_num: int,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays a colored segmentation mask onto an original image with transparency.

    Args:
        original_img: The original 3-channel (B, G, R) image.
        mask: A 2D NumPy array representing the segmentation mask (e.g., values 0, 1, 2, 3).
        class_num: number of classes.
        alpha: Transparency factor for the mask overlay (0.0 for full original, 1.0 for full mask).

    Returns:
        A 3D NumPy array (uint8) (H, W, 3) with the mask overlaid.
    """
    # 1. create a colored version of the mask
    colored_mask = colorize_mask(mask, class_num)

    # 2. Blend the original image and the colored mask
    # result = original_image * (1 - alpha) + colored_mask * alpha
    blended_image = cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0)

    return blended_image
