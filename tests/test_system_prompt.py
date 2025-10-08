"""Tests related to the system prompt behaviour."""

from footballer_app import _SYSTEM_PROMPT


def test_system_prompt_mentions_follow_up_questions() -> None:
    """The system prompt should instruct follow-up questions and teammate lookup."""

    assert "follow-up" in _SYSTEM_PROMPT
    assert "played with" in _SYSTEM_PROMPT
