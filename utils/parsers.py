"""Substrata document parsers - convert PDF, HTML, Markdown, TXT to plain text."""

from pathlib import Path
import re

import fitz
from bs4 import BeautifulSoup
from loguru import logger
from markdownify import markdownify as to_markdown


def _normalize_whitespace(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def parse_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    logger.info("Parsing PDF: {}", path)
    try:
        with fitz.open(path) as document:
            if document.is_encrypted:
                logger.warning("Skipping encrypted PDF: {}", path)
                return ""
            pages = [page.get_text("text").strip() for page in document]
        return _normalize_whitespace("\n\n".join(page for page in pages if page))
    except Exception as exc:
        logger.warning("Failed to parse PDF {}: {}", path, exc)
        return ""


def parse_html(path: Path) -> str:
    """Convert HTML to Markdown-like plain text."""
    logger.info("Parsing HTML: {}", path)
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        markdown = to_markdown(str(soup))
        return _normalize_whitespace(markdown)
    except Exception as exc:
        logger.warning("Failed to parse HTML {}: {}", path, exc)
        return ""


def parse_markdown(path: Path) -> str:
    """Read Markdown content and remove YAML frontmatter when present."""
    logger.info("Parsing Markdown: {}", path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.startswith("---"):
            text = re.sub(r"\A---\s*\n.*?\n---\s*\n?", "", text, count=1, flags=re.DOTALL)
        return _normalize_whitespace(text)
    except Exception as exc:
        logger.warning("Failed to parse Markdown {}: {}", path, exc)
        return ""


def parse_txt(path: Path) -> str:
    """Read plain text content."""
    logger.info("Parsing text file: {}", path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return _normalize_whitespace(text)
    except Exception as exc:
        logger.warning("Failed to parse text file {}: {}", path, exc)
        return ""


def parse_file(path: Path) -> str:
    """Dispatch parsing based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    if suffix == ".md":
        return parse_markdown(path)
    if suffix == ".txt":
        return parse_txt(path)

    logger.warning("Unsupported file extension for {}: {}", path, suffix or "<none>")
    return ""
