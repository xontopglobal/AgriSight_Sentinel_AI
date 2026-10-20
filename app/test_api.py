
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "AgriSight Sentinel AI"


def test_analyze_endpoint_accepts_sample_image():
    project_root = Path.cwd()
    image_path = project_root / "sample_leaf.png"

    assert image_path.is_file(), f"Sample image not found: {image_path}"

    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/analyze",
            files={
                "file": (
                    "sample_leaf.png",
                    image_file,
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["success"] is True
    assert result["action"] in {
        "continue_screening",
        "request_retake",
    }
    assert "image_analysis" in result
    assert "human_review_required" in result
