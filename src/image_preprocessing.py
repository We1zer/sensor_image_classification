import cv2
import numpy as np

from .config import (
    CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT,
    YELLOW_HSV_LOW, YELLOW_HSV_HIGH, INVERTED_THRESHOLD,
)


def read_image(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не вдалось прочитати зображення: {path}")
    return img


def crop_oscilloscope_area(img: np.ndarray,
                           top: int = CROP_TOP,
                           bottom: int = CROP_BOTTOM,
                           left: int = CROP_LEFT,
                           right: int = CROP_RIGHT) -> np.ndarray:
    h, w = img.shape[:2]
    y1 = max(0, top)
    y2 = max(y1 + 1, h - bottom)
    x1 = max(0, left)
    x2 = max(x1 + 1, w - right)
    return img[y1:y2, x1:x2].copy()


def invert_image(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img)


def extract_yellow_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(YELLOW_HSV_LOW, dtype=np.uint8)
    high = np.array(YELLOW_HSV_HIGH, dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    return mask


def extract_dark_mask_inverted(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, INVERTED_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return mask


def denoise_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = np.ones((kernel_size, kernel_size), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def preprocess_pipeline(img: np.ndarray, invert: bool = False) -> dict:
    cropped = crop_oscilloscope_area(img)

    if invert:
        inverted = invert_image(cropped)
        mask = extract_dark_mask_inverted(inverted)
    else:
        inverted = cropped
        mask = extract_yellow_mask(cropped)

    mask = denoise_mask(mask, kernel_size=3)

    return {"cropped": cropped, "inverted": inverted, "mask": mask}
