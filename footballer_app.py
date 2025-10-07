"""Command line utility to query OpenAI about footballers.

This module provides a simple function that sends user queries to the
OpenAI Responses API with a football-focused system prompt.  The script can
also be executed directly from the command line.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from openai import OpenAI, OpenAIError


_SYSTEM_PROMPT = (
    "You are an expert football analyst. Provide detailed, factual information "
    "about footballers, including career highlights, statistics, playing style, "
    "and notable achievements. Mention sources when relevant and clarify "
    "uncertainties."
)


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
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
