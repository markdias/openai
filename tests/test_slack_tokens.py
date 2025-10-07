"""Unit tests for Slack token validation helpers."""

from __future__ import annotations

import sys
import types
import unittest


# Provide lightweight stand-ins for optional third-party dependencies so that the
# module under test can be imported without requiring the actual packages to be
# installed in the test environment.
fake_requests = types.ModuleType("requests")
fake_requests.post = lambda *args, **kwargs: None
sys.modules.setdefault("requests", fake_requests)

fake_openai = types.ModuleType("openai")


class _DummyOpenAI:  # pragma: no cover - helper used only for import stubbing
    def __call__(self, *args: object, **kwargs: object) -> None:
        return None


fake_openai.OpenAI = _DummyOpenAI()


class _DummyOpenAIError(Exception):
    """Placeholder for the OpenAIError exception."""


fake_openai.OpenAIError = _DummyOpenAIError
sys.modules.setdefault("openai", fake_openai)

from footballer_app import SlackPostError, _ensure_bot_token


class EnsureBotTokenTests(unittest.TestCase):
    """Tests for the _ensure_bot_token helper."""

    def test_accepts_bot_token(self) -> None:
        """Tokens starting with xoxb- should be accepted."""

        # Should not raise
        _ensure_bot_token("xoxb-valid-token")

    def test_rejects_app_level_token(self) -> None:
        """App level tokens produce a descriptive error."""

        with self.assertRaises(SlackPostError) as ctx:
            _ensure_bot_token("xapp-some-token")

        self.assertIn("app-level", str(ctx.exception))

    def test_rejects_user_token(self) -> None:
        """User tokens should be rejected to avoid API errors."""

        with self.assertRaises(SlackPostError) as ctx:
            _ensure_bot_token("xoxp-user-token")

        self.assertIn("User tokens", str(ctx.exception))

    def test_rejects_malformed_token(self) -> None:
        """Any other string should also fail fast."""

        with self.assertRaises(SlackPostError):
            _ensure_bot_token("not-a-token")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
