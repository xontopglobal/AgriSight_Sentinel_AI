
from pathlib import Path

import cv2
import numpy as np

from app.vision import analyze_leaf_image


def test_sample_leaf_image_loads():
    project_root = Path(__file__).resolve().parents[2]
    image_path = project_root / "sample_leaf.png"

    image = cv2.imread(str(image_path))

    assert image is not None, f"Could not load image: {image_path}"
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_analyze_leaf_image_returns_quality_metrics():
    project_root = Path(__file__).resolve().parents[2]
    image_path = project_root / "sample_leaf.png"

    result = analyze_leaf_image(str(image_path))

    assert result["success"] is True
    assert result["width"] > 0
    assert result["height"] > 0
    assert "green_pixel_percentage" in result
    assert "sharpness_score" in result
    assert "brightness_score" in result

    quality = result["image_quality"]

    assert isinstance(quality["resolution_ok"], bool)
    assert isinstance(quality["sharpness_ok"], bool)
    assert isinstance(quality["brightness_ok"], bool)
    assert isinstance(quality["needs_retake"], bool)
    assert isinstance(quality["flags"], list)
