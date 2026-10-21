
from pathlib import Path

import cv2
import numpy as np


def analyze_leaf_image(image_path: str) -> dict:
    """Inspect an image and report basic visual-quality measurements."""

    path = Path(image_path)

    if not path.is_file():
        return {
            "success": False,
            "error": f"Image file not found: {path}",
        }

    image = cv2.imread(str(path))

    if image is None:
        return {
            "success": False,
            "error": "The file could not be decoded as an image.",
        }

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Basic image-quality indicators for visual screening.
    brightness = round(float(np.mean(gray)), 2)
    minimum_dimension = min(height, width)
    image_resolution_ok = minimum_dimension >= 256

    sharpness = round(
        float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2
    )
    sharpness_ok = sharpness >= 50
    brightness_ok = brightness >= 35

    quality_flags = []

    if not image_resolution_ok:
        quality_flags.append("low_resolution")

    if not sharpness_ok:
        quality_flags.append("possible_blur")

    if not brightness_ok:
        quality_flags.append("too_dark")

    # Estimate the proportion of green pixels.
    green_mask = cv2.inRange(
        hsv,
        np.array([25, 35, 25], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )

    green_pixels = int(cv2.countNonZero(green_mask))
    total_pixels = height * width
    green_percentage = round(
        100 * green_pixels / total_pixels, 2
    )

    return {
        "success": True,
        "image": path.name,
        "width": width,
        "height": height,
        "channels": int(image.shape[2]),
        "green_pixel_percentage": green_percentage,
        "sharpness_score": sharpness,
        "brightness_score": brightness,
        "image_quality": {
            "resolution_ok": image_resolution_ok,
            "sharpness_ok": sharpness_ok,
            "brightness_ok": brightness_ok,
            "needs_retake": bool(quality_flags),
            "flags": quality_flags,
        },
        "opencv_version": cv2.__version__,
        "note": (
            "Visual screening only. Green-pixel percentage and "
            "sharpness do not establish crop health or disease."
        ),
    }
