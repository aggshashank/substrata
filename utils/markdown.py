"""Substrata Markdown generator - creates Obsidian-compatible wiki entries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from loguru import logger


def slugify(text: str) -> str:
    """Create a stable slug for wiki file names and wikilinks."""
    slug = text.strip().lower().replace("_", "-").replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _yaml_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(f'"{value}"' for value in values) + "]"


def _wikilink(name: str) -> str:
    slug = slugify(name)
    return f"[[{slug}]]" if slug else ""


def create_paper_entry(
    title: str,
    summary: str,
    concepts: list[str],
    source_path: str,
    tags: list[str] | None = None,
) -> str:
    """Generate a paper wiki entry."""
    tag_values = tags or ["paper"]
    concept_links = [link for link in (_wikilink(concept) for concept in concepts) if link]
    concept_section = "\n".join(f"- {link}" for link in concept_links) or "None yet."

    return "\n".join(
        [
            "---",
            f'title: "{title}"',
            f"date: {_today_iso()}",
            f"tags: {_yaml_list(tag_values)}",
            f'source: "{source_path}"',
            'type: "paper"',
            "---",
            "",
            "## Summary",
            summary.strip() or "No summary yet.",
            "",
            "## Key Concepts",
            concept_section,
            "",
            "## Backlinks",
            "No backlinks yet.",
            "",
            "## See Also",
            "",
        ]
    ).strip() + "\n"


def create_concept_entry(
    concept_name: str,
    description: str,
    related_papers: list[str] | None = None,
    related_concepts: list[str] | None = None,
) -> str:
    """Generate a concept wiki entry."""
    paper_links = [link for link in (_wikilink(paper) for paper in (related_papers or [])) if link]
    concept_links = [link for link in (_wikilink(concept) for concept in (related_concepts or [])) if link]

    return "\n".join(
        [
            "---",
            f'title: "{concept_name}"',
            f"date: {_today_iso()}",
            'tags: ["concept"]',
            'type: "concept"',
            "---",
            "",
            "## Description",
            description.strip() or "No description yet.",
            "",
            "## Related Papers",
            "\n".join(f"- {link}" for link in paper_links) or "None yet.",
            "",
            "## Related Concepts",
            "\n".join(f"- {link}" for link in concept_links) or "None yet.",
            "",
            "## Backlinks",
            "No backlinks yet.",
            "",
        ]
    ).strip() + "\n"


def safe_write_file(path: Path, content: str) -> None:
    """Write Markdown content atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".md.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
    logger.info("Wrote file atomically: {}", path)


def update_index(wiki_dir: Path) -> None:
    """Rebuild the main wiki index from paper and concept entries."""
    papers_dir = wiki_dir / "papers"
    concepts_dir = wiki_dir / "concepts"
    paper_links = [f"- [[{file.stem}]]" for file in sorted(papers_dir.glob("*.md"))]
    concept_links = [f"- [[{file.stem}]]" for file in sorted(concepts_dir.glob("*.md"))]

    content = "\n".join(
        [
            "# Substrata Wiki Index",
            "",
            f"Updated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Papers",
            "\n".join(paper_links) or "No papers yet.",
            "",
            "## Concepts",
            "\n".join(concept_links) or "No concepts yet.",
            "",
        ]
    )
    safe_write_file(wiki_dir / "index.md", content)


def append_backlink(file_path: Path, link_target: str, link_label: str) -> None:
    """Append a backlink entry under the Backlinks section when missing."""
    if not file_path.exists():
        logger.warning("Cannot append backlink to missing file: {}", file_path)
        return

    content = file_path.read_text(encoding="utf-8", errors="replace")
    backlink_entry = (
        f"- [[{link_target}|{link_label}]]"
        if link_label and link_label != link_target
        else f"- [[{link_target}]]"
    )
    target_pattern = re.compile(rf"\[\[{re.escape(link_target)}(?:\|[^\]]+)?\]\]")
    if target_pattern.search(content):
        return

    marker = "## Backlinks"
    if marker not in content:
        content = content.rstrip() + f"\n\n{marker}\n{backlink_entry}\n"
        safe_write_file(file_path, content)
        return

    before, after = content.split(marker, 1)
    after = after.lstrip("\n")
    after_lines = after.splitlines()

    insert_index = len(after_lines)
    for index, line in enumerate(after_lines):
        if line.startswith("## ") and line != "## Backlinks":
            insert_index = index
            break

    backlink_lines = after_lines[:insert_index]
    remaining_lines = after_lines[insert_index:]

    if backlink_lines == ["No backlinks yet."]:
        backlink_lines = [backlink_entry]
    else:
        backlink_lines = [line for line in backlink_lines if line.strip()]
        backlink_lines.append(backlink_entry)

    rebuilt_sections = [before.rstrip(), marker]
    rebuilt_sections.extend(backlink_lines or [backlink_entry])
    if remaining_lines:
        rebuilt_sections.append("")
        rebuilt_sections.extend(remaining_lines)

    safe_write_file(file_path, "\n".join(rebuilt_sections).rstrip() + "\n")
