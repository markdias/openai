"""Command line utility to query OpenAI about footballers.

This module provides a simple function that sends user queries to the
OpenAI Responses API with a football-focused system prompt.  The script can
also be executed directly from the command line.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import textwrap
import unicodedata
import urllib.parse
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import requests
from openai import OpenAI, OpenAIError

if TYPE_CHECKING:
    from pandas import DataFrame
else:  # pragma: no cover - runtime fallback when pandas isn't available
    DataFrame = object  # type: ignore[assignment]

_PANDAS_SPEC = importlib.util.find_spec("pandas")
if _PANDAS_SPEC is not None:  # pragma: no cover - exercised when pandas is installed
    pd = importlib.import_module("pandas")
else:  # pragma: no cover - executed if pandas is missing
    pd = None


@dataclass
class ImageAsset:
    """Image bytes and metadata for embedding into a PDF."""

    label: str
    data: bytes
    image_type: str


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


def _default_pdf_path(query: str) -> str:
    """Return a filesystem-friendly default PDF filename for the query."""

    safe = re.sub(r"[^\w]+", "_", query.strip())
    safe = safe.strip("_") or "footballer_report"
    return f"{safe}.pdf"


def _search_wikipedia_title(term: str) -> Optional[str]:
    """Return the best matching Wikipedia page title for a search term."""

    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": term,
                "format": "json",
                "srlimit": 1,
            },
            timeout=10,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    search_results = payload.get("query", {}).get("search", [])
    if not search_results:
        return None

    return search_results[0].get("title")


def _fetch_wikipedia_summary(title: str) -> Optional[dict]:
    """Fetch the REST summary for a Wikipedia page title."""

    encoded_title = urllib.parse.quote(title)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
    try:
        response = requests.get(url, timeout=10, headers={"Accept": "application/json"})
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except ValueError:
        return None

    if data.get("type") == "disambiguation":
        return None

    return data


_IMAGE_SIGNATURES = {
    b"\xFF\xD8\xFF": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"GIF87a": "GIF",
    b"GIF89a": "GIF",
}


def _detect_image_type(data: bytes) -> Optional[str]:
    """Return a best-effort guess of the image type based on the signature."""

    for signature, image_type in _IMAGE_SIGNATURES.items():
        if data.startswith(signature):
            return image_type

    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "WEBP"

    return None


def _download_image(url: str) -> Optional[tuple[bytes, str]]:
    """Download an image and return its bytes and image type for PDF embedding."""

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    data = response.content
    image_type = None
    path = urllib.parse.urlparse(url).path
    if "." in path:
        extension = path.rsplit(".", 1)[-1].lower()
        if extension in {"jpg", "jpeg"}:
            image_type = "JPEG"
        elif extension == "png":
            image_type = "PNG"
        elif extension == "gif":
            image_type = "GIF"

    if image_type is None:
        detected = _detect_image_type(data)
        if detected:
            image_type = detected

    if image_type is None:
        return None

    return data, image_type


def fetch_entity_image(label: str, *, search_hint: Optional[str] = None) -> Optional[ImageAsset]:
    """Fetch an image for a named entity using Wikipedia as the source."""

    search_term = search_hint or label
    title = _search_wikipedia_title(search_term)
    if not title and search_hint:
        title = _search_wikipedia_title(label)
    if not title:
        return None

    summary = _fetch_wikipedia_summary(title)
    if not summary:
        return None

    image_info = summary.get("originalimage") or summary.get("thumbnail")
    if not image_info:
        return None

    image_url = image_info.get("source")
    if not image_url:
        return None

    download = _download_image(image_url)
    if not download:
        return None

    data, image_type = download
    return ImageAsset(label=label, data=data, image_type=image_type)


_ALIGNMENT_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_alignment_row(cells: list[str]) -> bool:
    cleaned = [cell.strip() for cell in cells]
    if not cleaned:
        return False
    return all(_ALIGNMENT_CELL_RE.match(cell) for cell in cleaned if cell)


def _find_first_markdown_table(markdown: str) -> Optional[tuple[list[str], int, int]]:
    lines = markdown.splitlines()
    start: Optional[int] = None
    table_lines: list[str] = []

    for index, raw_line in enumerate(lines):
        if raw_line.strip().startswith("|"):
            if start is None:
                start = index
            table_lines.append(raw_line)
        elif start is not None:
            end = index
            break
    else:
        if start is None:
            return None
        end = len(lines)

    if start is None or len(table_lines) < 2:
        return None

    return table_lines, start, end


def _format_markdown_table_lines(lines: list[str]) -> Optional[DataFrame]:
    if pd is None:
        return None

    rows = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        rows.append(_split_markdown_row(stripped))

    if len(rows) < 2:
        return None

    header = rows[0]
    data_rows: list[list[str]] = []

    for row in rows[1:]:
        if _is_alignment_row(row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        data_rows.append(row)

    if not data_rows:
        return None

    return pd.DataFrame(data_rows, columns=header)


def _render_markdown_with_formatted_table(markdown: str) -> str:
    if pd is None:
        return markdown

    table_info = _find_first_markdown_table(markdown)
    if not table_info:
        return markdown

    lines, start, end = table_info
    dataframe = _format_markdown_table_lines(lines)
    if dataframe is None:
        return markdown

    ascii_table = dataframe.to_string(index=False)
    ascii_lines = ascii_table.splitlines()

    response_lines = markdown.splitlines()
    formatted_lines = response_lines[:start] + ascii_lines + response_lines[end:]
    return "\n".join(formatted_lines)


def extract_club_names(markdown: str) -> list[str]:
    """Extract club names from the first Markdown table in a response."""

    table_info = _find_first_markdown_table(markdown)
    if not table_info:
        return []

    lines, _, _ = table_info
    clubs: list[str] = []
    seen: set[str] = set()
    header_seen = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or not stripped.startswith("|"):
            continue

        cells = _split_markdown_row(stripped)
        if not header_seen:
            if any(cell.lower() == "team" for cell in cells):
                header_seen = True
            continue

        if _is_alignment_row(cells):
            continue

        lowered_cells = [cell.lower() for cell in cells]
        if any("national" in cell and "team" in cell for cell in lowered_cells):
            break

        if len(cells) < 2:
            continue

        team = re.sub(r"\[[^\]]+\]", "", cells[1]).strip()
        team = re.sub(r"\s+", " ", team)
        if not team or team.lower() in {"team", "total", "totals"}:
            continue

        if team not in seen:
            seen.add(team)
            clubs.append(team)

    return clubs


def prepare_club_images(club_names: Iterable[str]) -> list[ImageAsset]:
    """Fetch club crest images for the given club names."""

    images: list[ImageAsset] = []
    for club in club_names:
        asset = fetch_entity_image(club, search_hint=f"{club} football club")
        if asset:
            images.append(ImageAsset(label=club, data=asset.data, image_type=asset.image_type))
    return images


@dataclass
class _PDFImage:
    width: int
    height: int
    color_space: str
    bits_per_component: int
    data: bytes
    filter_name: str
    decode_parms: Optional[str] = None
    smask: Optional["_PDFImage"] = None


@dataclass
class _PDFPage:
    operations: list[str]
    images: set[str]


class _PDFBuilder:
    """Very small PDF generator supporting text and basic images."""

    def __init__(self, *, page_width: float = 595.28, page_height: float = 841.89, margin: float = 48.0) -> None:
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin
        self.pages: list[_PDFPage] = []
        self.images: Dict[str, _PDFImage] = {}
        self._image_counter = 0
        self.current_page: Optional[_PDFPage] = None
        self.current_y: float = 0.0
        self.add_page()

    def add_page(self) -> None:
        page = _PDFPage(operations=[], images=set())
        self.pages.append(page)
        self.current_page = page
        self.current_y = self.page_height - self.margin

    def _ensure_page(self) -> _PDFPage:
        if self.current_page is None:
            self.add_page()
        assert self.current_page is not None
        return self.current_page

    def add_heading(self, text: str) -> None:
        self._add_text_block(text, font_size=18, bold=True, spacing=10)

    def add_section_heading(self, text: str) -> None:
        self._add_text_block(text, font_size=16, bold=True, spacing=8)

    def add_text(self, text: str) -> None:
        self._add_text_block(text, font_size=12, spacing=6)

    def add_caption(self, text: str) -> None:
        self._add_text_block(text, font_size=12, bold=True, spacing=4)

    def _add_text_block(self, text: str, *, font_size: float, bold: bool = False, spacing: float = 6.0) -> None:
        if not text:
            return

        page = self._ensure_page()
        font_key = "F2" if bold else "F1"
        available_width = self.page_width - 2 * self.margin
        approx_char_width = font_size * 0.5
        max_chars = max(1, int(available_width / approx_char_width))
        lines = textwrap.wrap(text, width=max_chars) or [text]
        line_height = font_size * 1.3

        for line in lines:
            if self.current_y - line_height < self.margin:
                self.add_page()
                page = self._ensure_page()
            normalised_line = _normalise_pdf_text(line)
            escaped_line = _escape_pdf_text(normalised_line)
            operation = (
                f"BT /{font_key} {font_size:.2f} Tf {self.margin:.2f} {self.current_y:.2f} Td ({escaped_line}) Tj ET"
            )
            page.operations.append(operation)
            self.current_y -= line_height

        self.current_y -= spacing

    def register_image(self, image: _PDFImage) -> str:
        self._image_counter += 1
        name = f"Im{self._image_counter}"
        self.images[name] = image
        return name

    def add_image(self, name: str, image: _PDFImage, *, max_width: float = 200.0) -> None:
        page = self._ensure_page()
        usable_width = min(max_width, self.page_width - 2 * self.margin)
        if usable_width <= 0:
            usable_width = self.page_width - 2 * self.margin
        scale = usable_width / image.width
        render_height = image.height * scale
        if self.current_y - render_height < self.margin:
            self.add_page()
            page = self._ensure_page()
        x = (self.page_width - usable_width) / 2
        y = self.current_y - render_height
        operation = f"q {usable_width:.2f} 0 0 {render_height:.2f} {x:.2f} {y:.2f} cm /{name} Do Q"
        page.operations.append(operation)
        page.images.add(name)
        self.current_y = y - 12

    def build(self) -> bytes:
        if not self.pages:
            self.add_page()

        objects: list[bytes] = []

        def add_object(data: bytes) -> int:
            objects.append(data)
            return len(objects)

        font_refs = {
            "F1": add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
            "F2": add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
        }

        image_refs: Dict[int, int] = {}

        def ensure_image_obj(image: _PDFImage) -> int:
            key = id(image)
            if key in image_refs:
                return image_refs[key]

            smask_ref = None
            if image.smask is not None:
                smask_ref = ensure_image_obj(image.smask)

            entries = [
                "/Type /XObject",
                "/Subtype /Image",
                f"/Width {image.width}",
                f"/Height {image.height}",
                f"/ColorSpace {image.color_space}",
                f"/BitsPerComponent {image.bits_per_component}",
                f"/Filter {image.filter_name}",
            ]
            if image.decode_parms:
                entries.append(f"/DecodeParms {image.decode_parms}")
            if smask_ref is not None:
                entries.append(f"/SMask {smask_ref} 0 R")
            entries.append(f"/Length {len(image.data)}")

            header = "<< " + " ".join(entries) + " >>"
            obj_data = header.encode("latin1") + b"\nstream\n" + image.data + b"\nendstream"
            obj_num = add_object(obj_data)
            image_refs[key] = obj_num
            return obj_num

        for name, image in self.images.items():
            ensure_image_obj(image)

        base_index = len(objects)
        pages_obj_num = base_index + len(self.pages) * 2 + 1
        page_numbers: list[int] = []

        for page in self.pages:
            content = "\n".join(page.operations).encode("latin1")
            content_obj = add_object(
                f"<< /Length {len(content)} >>".encode("latin1") + b"\nstream\n" + content + b"\nendstream"
            )

            resource_parts = [
                f"/Font << /F1 {font_refs['F1']} 0 R /F2 {font_refs['F2']} 0 R >>"
            ]
            if page.images:
                xobjects = []
                for image_name in sorted(page.images):
                    image_ref = image_refs[id(self.images[image_name])]
                    xobjects.append(f"/{image_name} {image_ref} 0 R")
                resource_parts.append(f"/XObject << {' '.join(xobjects)} >>")

            resources = "<< " + " ".join(resource_parts) + " >>"
            page_dict = (
                f"<< /Type /Page /Parent {pages_obj_num} 0 R /MediaBox [0 0 {self.page_width:.2f} {self.page_height:.2f}] "
                f"/Resources {resources} /Contents {content_obj} 0 R >>"
            )
            page_obj = add_object(page_dict.encode("latin1"))
            page_numbers.append(page_obj)

        kids = " ".join(f"{num} 0 R" for num in page_numbers)
        pages_dict = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>"
        pages_obj = add_object(pages_dict.encode("latin1"))
        catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("latin1"))

        buffer = bytearray(b"%PDF-1.4\n")
        offsets = [0]

        for index, obj in enumerate(objects, start=1):
            offsets.append(len(buffer))
            buffer.extend(f"{index} 0 obj\n".encode("latin1"))
            buffer.extend(obj)
            buffer.extend(b"\nendobj\n")

        startxref = len(buffer)
        buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin1"))
        buffer.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            buffer.extend(f"{offset:010d} 00000 n \n".encode("latin1"))
        buffer.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{startxref}\n%%EOF".encode("latin1")
        )

        return bytes(buffer)


def create_pdf_report(
    output_path: str,
    *,
    player_name: str,
    analysis_text: str,
    player_image: Optional[ImageAsset] = None,
    club_images: Optional[Iterable[ImageAsset]] = None,
) -> None:
    """Generate a PDF report summarising the analysis and images."""

    builder = _PDFBuilder()
    builder.add_heading(player_name.strip() or "Footballer Report")

    player_pdf_image = _convert_image_asset(player_image) if player_image else None
    if player_pdf_image:
        image_name = builder.register_image(player_pdf_image)
        builder.add_image(image_name, player_pdf_image, max_width=260)

    for paragraph in analysis_text.split("\n\n"):
        cleaned = paragraph.strip()
        if cleaned:
            builder.add_text(cleaned)

    club_assets: list[tuple[str, _PDFImage]] = []
    if club_images:
        for asset in club_images:
            pdf_image = _convert_image_asset(asset)
            if pdf_image:
                club_assets.append((asset.label, pdf_image))

    if club_assets:
        builder.add_section_heading("Club Crests")
        for label, pdf_image in club_assets:
            builder.add_caption(label)
            image_name = builder.register_image(pdf_image)
            builder.add_image(image_name, pdf_image, max_width=160)

    pdf_bytes = builder.build()
    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(pdf_bytes)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


_PDF_TEXT_REPLACEMENTS = {
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2018"): "'",  # left single quote
    ord("\u2019"): "'",  # right single quote / apostrophe
    ord("\u201c"): '"',  # left double quote
    ord("\u201d"): '"',  # right double quote
    ord("\u2026"): "...",  # ellipsis
    ord("\u2022"): "-",  # bullet
}


def _normalise_pdf_text(text: str) -> str:
    """Return text safe to encode using the PDF Latin-1 encoding."""

    normalised = unicodedata.normalize("NFKC", text)
    translated = normalised.translate(_PDF_TEXT_REPLACEMENTS)
    # ``ignore`` ensures we drop any remaining unsupported characters rather than
    # raising during PDF generation. This is preferable to hard failure when the
    # model returns symbols outside the font's encoding.
    return translated.encode("latin1", "ignore").decode("latin1")


def _convert_image_asset(asset: Optional[ImageAsset]) -> Optional[_PDFImage]:
    if asset is None:
        return None

    image_type = asset.image_type.upper()
    if image_type in {"JPG", "JPEG"}:
        return _build_pdf_image_from_jpeg(asset)
    if image_type == "PNG":
        return _build_pdf_image_from_png(asset)

    return None


def _build_pdf_image_from_jpeg(asset: ImageAsset) -> Optional[_PDFImage]:
    try:
        width, height, bits, color_space = _parse_jpeg_dimensions(asset.data)
    except ValueError:
        return None

    return _PDFImage(
        width=width,
        height=height,
        color_space=color_space,
        bits_per_component=bits,
        data=asset.data,
        filter_name="/DCTDecode",
    )


def _parse_jpeg_dimensions(data: bytes) -> tuple[int, int, int, str]:
    if not data.startswith(b"\xFF\xD8"):
        raise ValueError("Not a JPEG file")

    index = 2
    while index + 1 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue

        marker = data[index + 1]
        index += 2

        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 >= len(data):
                break
            length = int.from_bytes(data[index : index + 2], "big")
            if index + length > len(data):
                break
            bits = data[index + 2]
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            components = data[index + 7]
            color_space = {1: "/DeviceGray", 3: "/DeviceRGB"}.get(components)
            if color_space is None:
                raise ValueError("Unsupported JPEG color space")
            return width, height, bits, color_space

        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index : index + 2], "big")
        index += length

    raise ValueError("Could not parse JPEG dimensions")


def _build_pdf_image_from_png(asset: ImageAsset) -> Optional[_PDFImage]:
    try:
        parsed = _parse_png(asset.data)
    except ValueError:
        return None

    return parsed


def _parse_png(data: bytes) -> _PDFImage:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError("Not a PNG file")

    pos = len(signature)
    width = height = None
    bit_depth = None
    color_type = None
    palette = None
    transparency = None
    idat = bytearray()

    while pos + 8 <= len(data):
        chunk_len = int.from_bytes(data[pos : pos + 4], "big")
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + chunk_len]
        pos += 8 + chunk_len + 4  # include CRC

        if chunk_type == b"IHDR":
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
        elif chunk_type == b"PLTE":
            palette = chunk_data
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"tRNS":
            transparency = chunk_data
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG header incomplete")
    if bit_depth != 8:
        raise ValueError("Only 8-bit PNG images are supported")
    if not idat:
        raise ValueError("PNG image missing IDAT data")

    raw = zlib.decompress(bytes(idat))

    color_space = "/DeviceGray" if color_type in {0, 4} else "/DeviceRGB"
    color_components = 1 if color_space == "/DeviceGray" else 3

    bytes_per_pixel = {
        0: 1,
        2: 3,
        3: 1,
        4: 2,
        6: 4,
    }.get(color_type)
    if bytes_per_pixel is None:
        raise ValueError("Unsupported PNG color type")

    row_length = width * bytes_per_pixel
    offset = 0
    previous_row = bytearray([0] * row_length)

    color_rows: list[bytes] = []
    alpha_rows: list[bytes] = []

    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        row_data = bytearray(raw[offset : offset + row_length])
        offset += row_length
        recon = _apply_png_filter(filter_type, row_data, previous_row, bytes_per_pixel)
        previous_row = recon

        if color_type == 6:  # RGBA
            rgb = bytearray()
            alpha = bytearray()
            for index in range(0, len(recon), 4):
                rgb.extend(recon[index : index + 3])
                alpha.append(recon[index + 3])
            color_rows.append(bytes(rgb))
            alpha_rows.append(bytes(alpha))
        elif color_type == 4:  # grayscale + alpha
            gray = bytearray()
            alpha = bytearray()
            for index in range(0, len(recon), 2):
                gray.append(recon[index])
                alpha.append(recon[index + 1])
            color_rows.append(bytes(gray))
            alpha_rows.append(bytes(alpha))
        elif color_type == 3:  # indexed
            if palette is None:
                raise ValueError("Indexed PNG missing palette")
            rgb = bytearray()
            alpha = bytearray()
            for value in recon:
                palette_index = value * 3
                rgb.extend(palette[palette_index : palette_index + 3])
                if transparency:
                    if value < len(transparency):
                        alpha.append(transparency[value])
                    else:
                        alpha.append(255)
            color_rows.append(bytes(rgb))
            if transparency:
                alpha_rows.append(bytes(alpha))
            color_components = 3
        else:
            color_rows.append(bytes(recon))

    def _pack_rows(rows: list[bytes]) -> bytes:
        packed = bytearray()
        for row in rows:
            packed.append(0)
            packed.extend(row)
        return zlib.compress(bytes(packed))

    color_data = _pack_rows(color_rows)
    decode_parms = (
        f"<< /Predictor 15 /Colors {color_components} /BitsPerComponent 8 /Columns {width} >>"
    )

    image = _PDFImage(
        width=width,
        height=height,
        color_space=color_space,
        bits_per_component=8,
        data=color_data,
        filter_name="/FlateDecode",
        decode_parms=decode_parms,
    )

    if alpha_rows:
        alpha_data = _pack_rows(alpha_rows)
        smask = _PDFImage(
            width=width,
            height=height,
            color_space="/DeviceGray",
            bits_per_component=8,
            data=alpha_data,
            filter_name="/FlateDecode",
            decode_parms=f"<< /Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns {width} >>",
        )
        image.smask = smask

    return image


def _apply_png_filter(filter_type: int, row: bytearray, previous: bytearray, bpp: int) -> bytearray:
    result = bytearray(len(row))
    if filter_type == 0:  # None
        return bytearray(row)

    if filter_type == 1:  # Sub
        for i, value in enumerate(row):
            left = result[i - bpp] if i >= bpp else 0
            result[i] = (value + left) & 0xFF
        return result

    if filter_type == 2:  # Up
        for i, value in enumerate(row):
            up = previous[i] if i < len(previous) else 0
            result[i] = (value + up) & 0xFF
        return result

    if filter_type == 3:  # Average
        for i, value in enumerate(row):
            left = result[i - bpp] if i >= bpp else 0
            up = previous[i] if i < len(previous) else 0
            result[i] = (value + ((left + up) // 2)) & 0xFF
        return result

    if filter_type == 4:  # Paeth
        for i, value in enumerate(row):
            left = result[i - bpp] if i >= bpp else 0
            up = previous[i] if i < len(previous) else 0
            up_left = previous[i - bpp] if i >= bpp and i - bpp < len(previous) else 0
            predictor = _paeth_predictor(left, up, up_left)
            result[i] = (value + predictor) & 0xFF
        return result

    raise ValueError("Unsupported PNG filter type")


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c

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
        "--pdf",
        action="store_true",
        help="Generate a PDF report including the response text and available imagery.",
    )
    parser.add_argument(
        "--pdf-path",
        help="Custom output path for the generated PDF report (implies --pdf).",
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

    pdf_path = args.pdf_path
    generate_pdf = args.pdf or bool(pdf_path)
    if generate_pdf and not pdf_path:
        pdf_path = _default_pdf_path(args.query)

    career_table = args.career_table
    if career_table is None:
        career_table = _looks_like_player_name(args.query)
    if generate_pdf and not career_table:
        career_table = True

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

    display_response = _render_markdown_with_formatted_table(response)
    print(display_response)

    if generate_pdf and pdf_path:
        player_label = args.query.strip()
        if not player_label:
            player_label = "Footballer Report"

        first_line = response.splitlines()[0].strip() if response else ""
        cleaned_first_line = first_line.strip("#* ")
        if not _looks_like_player_name(player_label) and _looks_like_player_name(cleaned_first_line):
            player_label = cleaned_first_line

        player_image = fetch_entity_image(player_label, search_hint=f"{player_label} footballer")
        club_names = extract_club_names(response)[:6]
        club_images = prepare_club_images(club_names) if club_names else []

        try:
            create_pdf_report(
                pdf_path,
                player_name=player_label,
                analysis_text=response,
                player_image=player_image,
                club_images=club_images,
            )
        except OSError as exc:
            parser.error(f"Failed to generate PDF report: {exc}")
            return 2

        print(f"Saved PDF report to {pdf_path}")

    if args.slack_channel:
        slack_token = args.slack_token or os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            parser.error(
                "--slack-channel requires a token provided via --slack-token or the SLACK_BOT_TOKEN environment variable."
            )
            return 2

        try:
            send_slack_message(
                display_response,
                token=slack_token,
                channel=args.slack_channel,
            )
        except SlackPostError as exc:
            parser.error(f"Failed to send Slack message: {exc}")
            return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
