
import json
import os

from botocore.config import Config

import json
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from app.vision import analyze_leaf_image

# Load the application's environment configuration.
load_dotenv("app/.env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "openai.gpt-oss-120b-1:0",
)


def decide_next_action(image_path: str) -> dict:
    """Check whether a crop image is ready for visual screening."""

    analysis = analyze_leaf_image(image_path)

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
    Use Amazon Bedrock to interpret existing image-analysis results.

    This sends the analysis data to the model, not the original image.
    Findings are preliminary and require human review.
    """

    if not image_analysis.get("success"):
        return {
            "success": False,
            "error": "A successful image analysis is required.",
            "human_review_required": False,
        }

    from botocore.config import Config


    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(
            connect_timeout=10,
            read_timeout=60,
            retries={
                "max_attempts": 1,
                "mode": "standard",
            },
        ),
    )

    prompt = f"""
You are an agricultural screening assistant.

Review the following computer-vision analysis of a crop leaf:
{json.dumps(image_analysis, ensure_ascii=False)}

Provide a cautious preliminary assessment. Distinguish observed
evidence from uncertainty. Do not claim a confirmed diagnosis when
the provided data is insufficient.

Return a concise response with:
1. Preliminary observations
2. Possible concerns, if supported by the data
3. Recommended next steps
4. Whether expert inspection is advisable

The assessment is not a confirmed plant-disease diagnosis.
"""

    try:
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={"maxTokens": 500},
        )

        message = response.get("output", {}).get("message", {})
        content = message.get("content", [])

        # Extract only user-facing text blocks.
       
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        ]


        if not text_blocks:
            stop_reason = response.get("stopReason", "unknown")
            return {
                "success": False,
                "error": (
                    "Bedrock returned no user-facing text. "
                    f"Stop reason: {stop_reason}. "
                    "Inspect the response format and model output."
                ),
                "stop_reason": stop_reason,
                "response_content_types": [
                    list(block.keys())
                    for block in content
                    if isinstance(block, dict)
                ],
                "human_review_required": True,
            }




        # Return the successful assessment outside the if block.
        return {
            "success": True,
            "model_id": BEDROCK_MODEL_ID,
            "assessment": "\n".join(text_blocks),
            "human_review_required": True,
            "diagnosis_confirmed": False,
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Bedrock screening failed: {error}",
            "human_review_required": True,
        }
    
        # Return the successful assessment outside the if block.
        return {
            "success": True,
            "model_id": BEDROCK_MODEL_ID,
            "assessment": "\n".join(text_blocks),
            "human_review_required": True,
            "diagnosis_confirmed": False,
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Bedrock screening failed: {error}",
            "human_review_required": True,
        }


        # Return the successful assessment outside the if block.
        return {
            "success": True,
            "model_id": BEDROCK_MODEL_ID,
            "assessment": "\n".join(text_blocks),
            "human_review_required": True,
            "diagnosis_confirmed": False,
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Bedrock screening failed: {error}",
            "human_review_required": True,
        }
