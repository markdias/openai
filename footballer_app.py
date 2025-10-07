"""Command line utility to query OpenAI about footballers.

This module provides a simple function that sends user queries to the
OpenAI Responses API with a football-focused system prompt.  The script can
also be executed directly from the command line.
"""
from __future__ import annotations

import argparse
import os
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


def _ensure_bot_token(token: str) -> None:
    """Ensure the provided Slack token looks like a bot token.

    Slack's Web API distinguishes between bot tokens (``xoxb-``) and other token
    types such as app-level (``xapp-``) or user tokens (``xoxp-``). Only bot
    tokens are permitted to call ``chat.postMessage`` in the context of this
    tool. When the wrong token type is supplied the API responds with the
    ``not_allowed_token_type`` error, which can be confusing to interpret.

    Args:
        token: The Slack token string to validate.

    Raises:
        SlackPostError: If the token does not resemble a bot token.
    """

    if token.startswith("xoxb-"):
        return

    if token.startswith("xapp-"):
        raise SlackPostError(
            "Slack app-level tokens (starting with 'xapp-') cannot be used for chat.postMessage. "
            "Create a bot token and supply it via --slack-token or SLACK_BOT_TOKEN."
        )

    if token.startswith("xoxp-"):
        raise SlackPostError(
            "User tokens (starting with 'xoxp-') are not supported. Provide a bot token that begins with 'xoxb-'."
        )

    raise SlackPostError(
        "Slack token must be a bot token beginning with 'xoxb-'."
    )


def send_slack_message(message: str, *, token: str, channel: str) -> None:
    """Send a message to a Slack channel via the chat.postMessage API."""

    _ensure_bot_token(token)

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


def fetch_footballer_info(query: str, *, model: str = "gpt-4.1-mini", temperature: float = 0.3) -> str:
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
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
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
        "--slack-channel",
        help="Slack channel (e.g., #general) to post the response to.",
    )
    parser.add_argument(
        "--slack-token",
        help="Slack bot token. If omitted, the SLACK_BOT_TOKEN environment variable is used.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        response = fetch_footballer_info(
            args.query,
            model=args.model,
            temperature=args.temperature,
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
