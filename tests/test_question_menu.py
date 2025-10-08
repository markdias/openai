"""Tests for the interactive question menu helpers."""

from __future__ import annotations

import pytest

from footballer_app import MenuSelectionCancelled, _select_question_from_menu


def test_select_question_from_menu_returns_selected_question() -> None:
    """The helper should return the question corresponding to the chosen index."""

    prompts = iter(["2"])
    outputs: list[str] = []

    result = _select_question_from_menu(
        ["First question", "Second question", "Third question"],
        input_fn=lambda _: next(prompts),
        print_fn=outputs.append,
    )

    assert result == "Second question"
    assert any(line.startswith("1.") for line in outputs)


def test_select_question_from_menu_retries_invalid_input() -> None:
    """Invalid selections should prompt the user again until a valid choice is made."""

    prompts = iter(["invalid", "4", "1"])
    outputs: list[str] = []

    result = _select_question_from_menu(
        ["Only option"],
        input_fn=lambda _: next(prompts),
        print_fn=outputs.append,
    )

    assert result == "Only option"
    assert any("not a valid selection" in line for line in outputs)


def test_select_question_from_menu_cancel() -> None:
    """Entering 'q' should cancel the menu and raise a specific exception."""

    with pytest.raises(MenuSelectionCancelled):
        _select_question_from_menu(["Question"], input_fn=lambda _: "q", print_fn=lambda _: None)


def test_select_question_from_menu_requires_questions() -> None:
    """Providing an empty sequence should raise ValueError."""

    with pytest.raises(ValueError):
        _select_question_from_menu([], input_fn=lambda _: "1", print_fn=lambda _: None)
