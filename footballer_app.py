"""Command line utility to query OpenAI about footballers.

This module provides a simple function that sends user queries to the
OpenAI Responses API with a football-focused system prompt.  The script can
also be executed directly from the command line.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import requests
from openai import OpenAI, OpenAIError


_SYSTEM_PROMPT = (
    "You are an expert football analyst. Provide detailed, factual information "
    "about footballers, including career highlights, statistics, playing style, "
    "and notable achievements. Mention sources when relevant and clarify "
    "uncertainties."
)


class SlackPostError(RuntimeError):
    """Raised when posting a message to Slack fails."""


def send_slack_message(message: str, *, token: str, channel: str) -> None:
    """Send a message to a Slack channel via the chat.postMessage API."""

    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json={"channel": channel, "text": message},
        timeout=10,
    )

    if response.status_code != 200:
        raise SlackPostError(
            f"Slack API returned HTTP {response.status_code}: {response.text.strip()}"
        )

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise SlackPostError("Slack API returned invalid JSON response") from exc

    if not payload.get("ok", False):
        error = payload.get("error", "unknown_error")
        raise SlackPostError(f"Slack API error: {error}")


def fetch_footballer_info(
    query: str,
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    career_table: bool = False,
) -> str:
    """Fetch information about a footballer using the OpenAI Responses API.

    Args:
        query: The user question or topic related to a footballer.
        model: The OpenAI model to query.
        temperature: Sampling temperature passed to the model.

    Returns:
        The assistant's textual response.

    Raises:
        OpenAIError: If the API call fails for any reason.
    """

    client = OpenAI()
    user_message = query
    if career_table:
        user_message = (
            f"{query}\n\n"
            "Return the player's professional career history as a chronological "
            "Markdown table. Include club and national-team stints with columns for "
            "Years, Team, Competition/League, Appearances, Goals, and Notes. Use "
            "'N/A' when figures are unavailable and add any important context below "
            "the table."
        )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.output_text.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query OpenAI for footballer information.")
    parser.add_argument("query", help="Question or topic about a footballer to research.")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--career-table",
        dest="career_table",
        action="store_true",
        default=None,
        help="Force the response to be formatted as a career history table.",
    )
    parser.add_argument(
        "--no-career-table",
        dest="career_table",
        action="store_false",
        help="Disable the automatic career history table formatting.",
    )
    parser.add_argument(
        "--slack-channel",
        help="Slack channel (e.g., #general) to post the response to.",
    )
    parser.add_argument(
        "--slack-token",
        help="Slack bot token. If omitted, the SLACK_BOT_TOKEN environment variable is used.",
    )
    return parser


_NAME_CHARS = re.compile(r"^[A-Za-z .'-]+$")


def _looks_like_player_name(text: str) -> bool:
    """Heuristically determine whether the query is just a player's name."""

    stripped = text.strip()
    if not stripped:
        return False

    # Quickly discard anything that looks like a question or command.
    if any(punct in stripped for punct in "?!"):
        return False

    if not _NAME_CHARS.match(stripped):
        return False

    tokens = stripped.split()
    if len(tokens) > 6:
        return False

    # Avoid common prompt starters that would otherwise match the regex.
    lower_first = tokens[0].lower()
    if lower_first in {"tell", "show", "give", "provide"}:
        return False

    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    career_table = args.career_table
    if career_table is None:
        career_table = _looks_like_player_name(args.query)

    try:
        response = fetch_footballer_info(
            args.query,
            model=args.model,
            temperature=args.temperature,
            career_table=career_table,
        )
    except OpenAIError as exc:  # pragma: no cover - depends on external service
        parser.error(f"OpenAI API request failed: {exc}")
        return 2

    print(response)

    if args.slack_channel:
        slack_token = args.slack_token or os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            parser.error(
                "--slack-channel requires a token provided via --slack-token or the SLACK_BOT_TOKEN environment variable."
            )
            return 2

        try:
            send_slack_message(
                response,
                token=slack_token,
                channel=args.slack_channel,
            )
        except SlackPostError as exc:
            parser.error(f"Failed to send Slack message: {exc}")
            return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
