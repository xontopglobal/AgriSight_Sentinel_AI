
from pathlib import Path

from app.agent import decide_next_action


def test_agent_handles_missing_image():
    result = decide_next_action(
        str(Path("missing_sample_leaf.png"))
    )

    assert result["success"] is False
    assert result["action"] == "stop"


def test_agent_requests_retake_for_poor_quality(monkeypatch):
    from app import agent

    def fake_analysis(_image_path):
        return {
            "success": True,
            "image_quality": {
                "needs_retake": True,
                "flags": ["possible_blur"],
            },
        }

    monkeypatch.setattr(
        agent, "analyze_leaf_image", fake_analysis
    )

    result = agent.decide_next_action("sample_leaf.png")

    assert result["success"] is True
    assert result["action"] == "request_retake"
    assert "possible_blur" in result["quality_flags"]


def test_agent_continues_with_good_quality(monkeypatch):
    from app import agent

    def fake_analysis(_image_path):
        return {
            "success": True,
            "image_quality": {
                "needs_retake": False,
                "flags": [],
            },
        }

    monkeypatch.setattr(
        agent, "analyze_leaf_image", fake_analysis
    )

    result = agent.decide_next_action("sample_leaf.png")

    assert result["success"] is True
    assert result["action"] == "continue_screening"
    assert result["human_review_required"] is True
