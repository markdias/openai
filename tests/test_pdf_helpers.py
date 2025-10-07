"""Tests for PDF helper utilities in footballer_app."""

from footballer_app import _default_pdf_path, extract_club_names


def test_default_pdf_path_sanitises_query():
    assert _default_pdf_path("Kylian Mbappé!") == "Kylian_Mbappé.pdf"


def test_extract_club_names_from_markdown_table():
    markdown = (
        "| Years | Team | Competition/League | Apps | Goals | Notes |\n"
        "| 2015–2017 | FC Example | Example League | 40 | 10 | - |\n"
        "| 2017–2020 | Another Town | Example League | 82 | 19 | Player of the year |\n"
        "| colspan=6 style=font-weight:bold | National team |\n"
        "| 2018– | Exampleland | World Cup | 25 | 5 | Captain |\n"
    )

    clubs = extract_club_names(markdown)

    assert clubs == ["FC Example", "Another Town"]

