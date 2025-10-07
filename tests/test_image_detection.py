"""Tests for image type detection helper."""

from footballer_app import _detect_image_type


def test_detects_common_signatures():
    assert _detect_image_type(b"\xFF\xD8\xFF\xE0rest") == "JPEG"
    assert _detect_image_type(b"\x89PNG\r\n\x1a\nrest") == "PNG"
    assert _detect_image_type(b"GIF89a...") == "GIF"
    assert _detect_image_type(b"RIFF1234WEBPmore") == "WEBP"


def test_unknown_signature_returns_none():
    assert _detect_image_type(b"notanimage") is None
