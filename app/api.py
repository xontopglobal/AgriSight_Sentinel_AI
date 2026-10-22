
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent import (
    decide_next_action,
    screen_crop_with_bedrock,
)

app = FastAPI(
    title="AgriSight Sentinel AI",
    description=(
        "Crop image screening with OpenCV, "
        "quality-aware inspection, and Amazon Bedrock."
    ),
    version="0.2.0",
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

    suffix = Path(file.filename or "upload.png").suffix

    if suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
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

        # Call Bedrock only after successful image analysis.
        if (
            isinstance(result, dict)
            and result.get("success")
            and result.get("action") == "continue_screening"
        ):
            image_analysis = result.get("image_analysis")

            if not isinstance(image_analysis, dict):
                bedrock_result = {
                    "success": False,
                    "error": "Image analysis data is missing or invalid.",
                    "human_review_required": True,
                }
            else:
                bedrock_result = screen_crop_with_bedrock(
                    image_analysis
                )

                # Protect the API if Bedrock returns an
                # unexpected response type.
                if not isinstance(bedrock_result, dict):
                    bedrock_result = {
                        "success": False,
                        "error": "Bedrock returned an invalid response.",
                        "human_review_required": True,
                    }

            result["bedrock_screening"] = bedrock_result

            if not bedrock_result.get("success"):
                result["human_review_required"] = True
                result["screening_notice"] = (
                    "Preliminary image checks passed, but the "
                    "AI assessment could not be completed. "
                    "Please try again or request expert review."
                )

        return result

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        await file.close()
