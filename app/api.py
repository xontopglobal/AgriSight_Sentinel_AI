
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.agent import decide_next_action

app = FastAPI(
    title="AgriSight Sentinel AI",
    description=(
        "Crop image screening with OpenCV and "
        "quality-aware inspection."
    ),
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "name": "AgriSight Sentinel AI",
        "status": "running",
        "opencv_endpoint": "/api/analyze",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


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
        return result

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        await file.close()
