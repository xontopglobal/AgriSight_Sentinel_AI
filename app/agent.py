
import json
import logging
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from app.vision import analyze_leaf_image


logger = logging.getLogger(__name__)

# Load local settings if app/.env exists.
# Render environment variables are used in production.
load_dotenv(Path(__file__).resolve().parent / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "openai.gpt-oss-120b-1:0",
)


def decide_next_action(image_path: str) -> dict:
    """Check whether a crop image is ready for AI screening."""

    analysis = analyze_leaf_image(image_path)

    if not isinstance(analysis, dict) or not analysis.get("success"):
        error = (
            analysis.get("error", "Image analysis failed.")
            if isinstance(analysis, dict)
            else "Image analysis returned an invalid result."
        )
        return {
            "success": False,
            "action": "stop",
            "reason": error,
            "human_review_required": False,
        }

    quality = analysis.get("image_quality", {})

    if not isinstance(quality, dict):
        quality = {}

    if quality.get("needs_retake", True):
        return {
            "success": True,
            "action": "request_retake",
            "reason": (
                "The image did not pass all quality checks. "
                "Please capture a clearer, well-lit image."
            ),
            "quality_flags": quality.get("flags", []),
            "image_analysis": analysis,
            "human_review_required": False,
        }

    return {
        "success": True,
        "action": "continue_screening",
        "reason": (
            "The image passed preliminary quality checks. "
            "This does not establish that the crop is healthy."
        ),
        "quality_flags": [],
        "image_analysis": analysis,
        "human_review_required": True,
    }


def screen_crop_with_bedrock(image_analysis: dict) -> dict:
    """
    Ask Amazon Bedrock to interpret the OpenCV analysis.

    This sends analysis data, not the original image.
    Results are preliminary and require human review.
    """

    if not isinstance(image_analysis, dict) or not image_analysis.get("success"):
        return {
            "success": False,
            "error": "A successful image analysis is required.",
            "human_review_required": True,
        }

    if not AWS_REGION or not BEDROCK_MODEL_ID:
        logger.error("AWS region or Bedrock model ID is missing.")
        return {
            "success": False,
            "error": "AWS region or Bedrock model configuration is missing.",
            "human_review_required": True,
        }

    prompt = f"""
You are an agricultural crop-screening assistant.

Interpret the following computer-vision analysis of a crop leaf:

{json.dumps(image_analysis, ensure_ascii=False, default=str)}

Provide a cautious preliminary assessment.
Distinguish observed evidence from uncertainty.
Do not invent symptoms or claim a confirmed diagnosis.

Return these sections:
1. Preliminary observations
2. Possible concerns supported by the evidence
3. Recommended next steps
4. Whether expert inspection is advisable

If the data is insufficient to identify a disease, say so clearly.
This is not a confirmed plant-disease diagnosis.
"""

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=Config(
                connect_timeout=10,
                read_timeout=90,
                retries={
                    "max_attempts": 2,
                    "mode": "standard",
                },
            ),
        )

        logger.info(
            "Calling Bedrock: region=%s, model_id=%s",
            AWS_REGION,
            BEDROCK_MODEL_ID,
        )

        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": 700,
                "temperature": 0.2,
            },
        )

        message = response.get("output", {}).get("message", {})
        content = message.get("content", [])

        text_blocks = [
            block["text"].strip()
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        ]

        if not text_blocks:
            stop_reason = response.get("stopReason", "unknown")

            logger.error(
                "Bedrock returned no text. stop_reason=%s, "
                "content_block_count=%d",
                stop_reason,
                len(content),
            )

            return {
                "success": False,
                "error": "Bedrock returned no assessment text.",
                "stop_reason": stop_reason,
                "human_review_required": True,
            }

        assessment = "\n".join(text_blocks)

        logger.info("Bedrock returned an assessment successfully.")

        return {
            "success": True,
            "model_id": BEDROCK_MODEL_ID,
            "assessment": assessment,
            "stop_reason": response.get("stopReason", "unknown"),
            "human_review_required": True,
            "diagnosis_confirmed": False,
        }

    except ClientError as error:
        details = error.response.get("Error", {})
        error_code = details.get("Code", "AWSClientError")
        request_id = error.response.get(
            "ResponseMetadata", {}
        ).get("RequestId", "unknown")

        logger.exception(
            "Bedrock AWS error: code=%s, request_id=%s",
            error_code,
            request_id,
        )

        return {
            "success": False,
            "error": f"Bedrock AWS request failed ({error_code}).",
            "aws_error_code": error_code,
            "request_id": request_id,
            "human_review_required": True,
        }

    except BotoCoreError as error:
        logger.exception("Bedrock SDK error")

        return {
            "success": False,
            "error": f"Bedrock SDK error: {type(error).__name__}: {error}",
            "human_review_required": True,
        }

    except Exception as error:
        logger.exception("Unexpected error during Bedrock screening")

        return {
            "success": False,
            "error": (
                f"Bedrock screening failed: "
                f"{type(error).__name__}: {error}"
            ),
            "human_review_required": True,
        }
