"""Substrata text chunking - splits documents into overlapping chunks for embedding."""

from __future__ import annotations

from pathlib import Path
import re

from config import get_settings


_SETTINGS = get_settings()
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def estimate_tokens(text: str) -> int:
    """Estimate token count from whitespace-separated words."""
    return int(len(text.split()) * 0.75)


def _split_paragraph_sentences(paragraph: str) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(paragraph) if sentence.strip()]
    return sentences or [paragraph.strip()]


def _split_text(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(_split_paragraph_sentences(paragraph))
    return sentences


def _build_overlap(sentences: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or not sentences:
        return []

    overlap_sentences: list[str] = []
    for sentence in reversed(sentences):
        overlap_sentences.insert(0, sentence)
        if estimate_tokens(" ".join(overlap_sentences)) >= overlap:
            break
    return overlap_sentences


def _join_sentences(sentences: list[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()


def chunk_text(
    text: str,
    chunk_size: int = _SETTINGS.CHUNK_SIZE,
    overlap: int = _SETTINGS.CHUNK_OVERLAP,
    doc_id: str = "",
    source_file: str = "",
) -> list[dict]:
    """Split text into overlapping sentence-aware chunks."""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    sentences = _split_text(stripped_text)
    if not sentences:
        sentences = [stripped_text]

    chunks: list[dict[str, str | int]] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        candidate_sentences = current_sentences + [sentence]
        candidate_text = _join_sentences(candidate_sentences)
        if current_sentences and estimate_tokens(candidate_text) >= chunk_size:
            chunk_text_value = _join_sentences(current_sentences)
            if chunk_text_value:
                chunks.append(
                    {
                        "text": chunk_text_value,
                        "doc_id": doc_id,
                        "chunk_index": len(chunks),
                        "source_file": source_file,
                        "token_estimate": estimate_tokens(chunk_text_value),
                    }
                )
            current_sentences = _build_overlap(current_sentences, overlap)
            current_sentences.append(sentence)
        else:
            current_sentences = candidate_sentences

    final_text = _join_sentences(current_sentences)
    if final_text:
        chunks.append(
            {
                "text": final_text,
                "doc_id": doc_id,
                "chunk_index": len(chunks),
                "source_file": source_file,
                "token_estimate": estimate_tokens(final_text),
            }
        )

    return [chunk for chunk in chunks if str(chunk["text"]).strip()]
