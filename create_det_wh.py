import os
import shutil
import unicodedata

import cv2
from paddleocr import PaddleOCR

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def get_image_filenames_in_directory(directory_path: str) -> list[str]:
    """
    Retrieves a list of full paths to common image files within a specified directory.

    Args:
        directory_path (str): The path to the directory to scan.

    Returns:
        list[str]: A list of full paths to image files found in the directory.
                   Returns an empty list if the directory does not exist or contains no images.
    """
    image_filenames = []

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it's a file (and not a directory or symlink)
        if os.path.isfile(file_path):
            # Get the file extension and convert to lowercase for case-insensitive comparison
            file_extension = os.path.splitext(filename)[1].lower()

            # Check if the extension is in our list of image extensions
            if file_extension in IMAGE_EXTENSIONS:
                image_filenames.append(filename)

    return sorted(image_filenames)


def standardize_text(text: str) -> str:
    """Standardize the text to simplify post process and unify final result."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(",", "")
    return text


def get_ppocr_results_ltrb(
    img_path: str,
    ocr: PaddleOCR,
    is_horizontal: bool
) -> list[dict]:
    """
    Performs OCR using PP-OCRv5 and formats the results into a list of dictionaries.

    Args:
        img_path (str): The file path to the image for OCR.
        ocr (PaddleOCR): PaddleOCR model.
        is_horizontal (bool): Detect only horizontal if True.

    Returns:
        list: A list of dictionaries, where each dict has the format:
              {"ltrb": (x1, y1, x2, y2), "text": str}
    """
    # 1. Predict using ppocr
    try:
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        if not is_horizontal:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        results = ocr.predict(img)
    except Exception as e:
        print(f"ERROR: An error occurred during OCR: {e}")
        return []

    # 2. Process and format the results
    formatted_results = []
    for result in results:
        for box_points, text, score in zip(
            result["dt_polys"], result["rec_texts"], result["rec_scores"]
        ):  # 2.1. Convert text to float
            text = standardize_text(text)
            print("==")
            print(text)
            # 2.2. Convert the 4-point quad to the LTRB (Left-Top, Right-Bottom) format
            # 2.2.1. Extract x and y coordinates
            x_coords = [point[0] for point in box_points]
            y_coords = [point[1] for point in box_points]

            # 2.2.2. Get the ltrb
            x1 = int(round(min(x_coords)))
            y1 = int(round(min(y_coords)))
            x2 = int(round(max(x_coords)))
            y2 = int(round(max(y_coords)))

            # 2.2.3. Keep only horizontal texts
            if (x2 - x1) * 1.2 < y2 - y1:
                continue

            # 2.2.4. Recalculate ltrb if not is_horizontal
            if not is_horizontal:
                y1_new = img_h - 1 - x2
                y2_new = img_h - 1 - x1
                x1_new = y1
                x2_new = y2
                x1, y1, x2, y2 = (x1_new, y1_new, x2_new, y2_new)

            # 2.2.4. Create desired format and append to list
            formatted_result = {
                "ltrb": (x1, y1, x2, y2),
                "text": text,
                "score": score,
            }
            formatted_results.append(formatted_result)

    return formatted_results


def has_overlapped_center(
    ltrb_1: tuple[float, float, float, float],
    ltrb_2: tuple[float, float, float, float]
) -> bool:
    x1_1, y1_1, x2_1, y2_1 = ltrb_1
    x_center_1 = (x1_1 + x2_1) / 2
    y_center_1 = (y1_1 + y2_1) / 2
    x1_2, y1_2, x2_2, y2_2 = ltrb_2
    x_center_2 = (x1_2 + x2_2) / 2
    y_center_2 = (y1_2 + y2_2) / 2
    return ((x1_2 < x_center_1 < x2_2 and y1_2 < y_center_1 < y2_2)
            or (x1_1 < x_center_2 < x2_1 and y1_1 < y_center_2 < y2_1))


def filter_overlapped_results_using_score(
    results: list[dict],
    reference_results: list[dict]
) -> list[dict]:
    filtered_results = []
    # Loop each result
    for result in results:
        is_filtered = False
        # Loop each reference_result
        for reference_result in reference_results:
            # is_filtered if has_overlapped_center and result has lower score
            if (has_overlapped_center(result["ltrb"], reference_result["ltrb"])
                and result["score"] < reference_result["score"]):
                is_filtered = True
                break
        # Append to filtered_results if not is_filtered
        if not is_filtered:
            filtered_results.append(result)
    return filtered_results


# Load PP-OCRv5 model
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

input_dirs = [
    "../20251029_glass_detection/test",
    # "../20251029_glass_detection/train",
]

for input_dir in input_dirs:
    output_dir = os.path.join(input_dir, "det_wh")
    shutil.rmtree(output_dir, ignore_errors=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get images_dir path
    images_dir = os.path.join(input_dir, "images")

    # Loop each drawing image
    for filename in get_image_filenames_in_directory(images_dir):
        print(filename)
        # 1. Read image
        # Get image_path
        image_path = os.path.join(images_dir, filename)
        # Get image_name
        image_name = os.path.splitext(filename)[0]

        # 2.Detect text
        width_results = get_ppocr_results_ltrb(
            image_path, ocr, is_horizontal=True
        )
        height_results = get_ppocr_results_ltrb(
            image_path, ocr, is_horizontal=False
        )

        # # 3. Filter overlapped results
        width_results = filter_overlapped_results_using_score(width_results, height_results)
        height_results = filter_overlapped_results_using_score(height_results, width_results)

        # 4. Write wh gt
        det_wh_path = os.path.join(output_dir, f"{image_name}.txt")
        with open(det_wh_path, "w") as gt_file:
            for gt_type, results in (
                    ("width", width_results),
                    ("height", height_results),
            ):
                for result in results:
                    x1, y1, x2, y2 = map(int, result["ltrb"])
                    text = result["text"]
                    gt_file.write(f"{x1},{y1},{x2},{y2},{text},{gt_type}\n")
