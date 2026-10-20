
from app.vision import analyze_leaf_image


def decide_next_action(image_path: str) -> dict:
    """
    Decide whether a crop image is ready for visual screening
    or should be captured again.
    """

    analysis = analyze_leaf_image(image_path)

    # Stop safely if the image could not be analyzed.
    if not analysis.get("success"):
        return {
            "success": False,
            "action": "stop",
            "reason": analysis.get(
                "error", "Image analysis failed."
            ),
            "human_review_required": False,
        }

    quality = analysis.get("image_quality", {})

    # Request a better image when quality checks fail.
    if quality.get("needs_retake", True):
        flags = quality.get("flags", [])

        return {
            "success": True,
            "action": "request_retake",
            "reason": (
                "The image did not pass all quality checks. "
                "Please capture a clearer, well-lit image."
            ),
            "quality_flags": flags,
            "image_analysis": analysis,
            "human_review_required": False,
        }

    # Good image quality permits the next screening stage.
    return {
        "success": True,
        "action": "continue_screening",
        "reason": (
            "The image passed the preliminary quality checks. "
            "This does not establish that the crop is healthy."
        ),
        "quality_flags": [],
        "image_analysis": analysis,
        "human_review_required": True,
    }
