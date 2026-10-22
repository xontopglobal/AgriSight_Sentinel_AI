
import logging
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent import (
    decide_next_action,
    screen_crop_with_bedrock,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgriSight Sentinel AI",
    description=(
        "Crop image screening with OpenCV, "
        "quality-aware inspection, and Amazon Bedrock."
    ),
    version="0.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://agrisight-sentinel-ai.netlify.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def home():
    return {
        "name": "AgriSight Sentinel AI",
        "status": "running",
        "opencv_endpoint": "/api/analyze",
        "bedrock_enabled": True,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "AgriSight Sentinel AI",
    }


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        result = decide_next_action(str(temporary_path))

        if not isinstance(result, dict):
            logger.error("Image pipeline returned an invalid result.")
            raise HTTPException(
                status_code=500,
                detail="Image analysis could not be completed.",
            )

        should_call_bedrock = (
            result.get("success") is True
            and result.get("action") == "continue_screening"
        )

        if should_call_bedrock:
            image_analysis = result.get("image_analysis")

            if not isinstance(image_analysis, dict):
                logger.error(
                    "Bedrock skipped: image_analysis is missing "
                    "or is not a dictionary."
                )
                bedrock_result = {
                    "success": False,
                    "error": "Image analysis data is missing or invalid.",
                    "human_review_required": True,
                }
            else:
                try:
                    logger.info(
                        "Starting Bedrock screening for image analysis."
                    )
                    bedrock_result = screen_crop_with_bedrock(
                        image_analysis
                    )

                    if not isinstance(bedrock_result, dict):
                        logger.error(
                            "Bedrock function returned an unexpected type: %s",
                            type(bedrock_result).__name__,
                        )
                        bedrock_result = {
                            "success": False,
                            "error": "Bedrock returned an invalid response.",
                            "human_review_required": True,
                        }
                    elif not bedrock_result.get("success"):
                        # Log the returned error, but never log credentials.
                        logger.error(
                            "Bedrock screening reported failure: %s",
                            bedrock_result.get(
                                "error", "No error details provided"
                            ),
                        )
                    else:
                        logger.info("Bedrock screening completed successfully.")

                except Exception:
                    logger.exception(
                        "Unexpected exception during Bedrock screening."
                    )
                    bedrock_result = {
                        "success": False,
                        "error": "Bedrock request failed. Check server logs.",
                        "human_review_required": True,
                    }

            result["bedrock_screening"] = bedrock_result

            if not bedrock_result.get("success"):
                result["human_review_required"] = True
                result["screening_notice"] = (
                    "Preliminary image checks passed, but the AI "
                    "assessment could not be completed. Please try "
                    "again or request expert review."
                )

        return result

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        await file.close()
