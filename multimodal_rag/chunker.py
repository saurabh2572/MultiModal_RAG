"""Create retrieval chunks without mixing text, table, and image content."""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator


def token_estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _metadata(element: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source", "source_path", "source_sha256", "page", "page_label",
        "page_width", "page_height", "section", "subsection", "modality", "bbox",
        "extraction_method", "heading", "table_index", "image_path", "caption",
        "ocr_confidence", "context_above", "element_ids",
    )
    return {key: element[key] for key in keys if key in element and element[key] is not None}


def _make_chunk(element: dict[str, Any], index: int, text: str, **extra: Any) -> dict[str, Any]:
    metadata = _metadata(element) | extra
    metadata["estimated_tokens"] = token_estimate(text)
    return {"id": f"{element['id']}-chunk{index:03d}", "element_id": element["id"], "text": text, "metadata": metadata}


def _split_long_text(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries, then words when a sentence is oversized."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    units: list[str] = []
    for sentence in sentences:
        if token_estimate(sentence) <= limit:
            units.append(sentence)
            continue
        current: list[str] = []
        for word in sentence.split():
            if current and token_estimate(" ".join(current + [word])) > limit:
                units.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            units.append(" ".join(current))
    return [unit for unit in units if unit.strip()]


def _text_chunks(element: dict[str, Any], max_tokens: int, overlap_tokens: int) -> Iterator[dict[str, Any]]:
    text = element["text"].strip()
    if not text:
        return
    units = _split_long_text(text, max_tokens)
    current: list[str] = []
    index = 1
    for unit in units:
        if current and token_estimate(" ".join(current + [unit])) > max_tokens:
            yield _make_chunk(element, index, " ".join(current), chunk_strategy="sentence")
            index += 1
            tail: list[str] = []
            for prior in reversed(current):
                if token_estimate(" ".join([prior] + tail)) > overlap_tokens:
                    break
                tail.insert(0, prior)
            current = tail
        current.append(unit)
    if current:
        yield _make_chunk(element, index, " ".join(current), chunk_strategy="sentence")


def _table_chunks(element: dict[str, Any], max_tokens: int) -> Iterator[dict[str, Any]]:
    rows = element["rows"]
    if not rows:
        return
    width = max(len(row) for row in rows)

    def line(row: list[str]) -> str:
        return " | ".join(row + [""] * (width - len(row)))

    context = element.get("context_above", "")
    prefix = f"Section: {element['section']}\nTable on page {element['page']}\n{context}\n" if context else f"Section: {element['section']}\nTable on page {element['page']}\n"
    current: list[str] = []
    start_row = 1
    index = 1
    for row_number, row in enumerate(rows, 1):
        row_text = line(row)
        if current and token_estimate(prefix + "\n".join(current + [row_text])) > max_tokens:
            yield _make_chunk(element, index, prefix + "\n".join(current), chunk_strategy="table_rows", row_start=start_row, row_end=row_number - 1, column_count=width)
            index += 1
            current = []
            start_row = row_number
        current.append(row_text)
    if current:
        yield _make_chunk(element, index, prefix + "\n".join(current), chunk_strategy="table_rows", row_start=start_row, row_end=len(rows), column_count=width)


def chunk_elements(elements: Iterable[dict[str, Any]], *, max_tokens: int = 700, overlap_tokens: int = 100) -> Iterator[dict[str, Any]]:
    if max_tokens < 100 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("max_tokens must be >= 100 and overlap_tokens smaller than max_tokens")
    pending: list[dict[str, Any]] = []

    def flush_text() -> Iterator[dict[str, Any]]:
        if not pending:
            return
        first = pending[0]
        combined = first.copy()
        combined["id"] = f"{first['id']}-to-{pending[-1]['id']}"
        combined["element_ids"] = [item["id"] for item in pending]
        combined["text"] = "\n\n".join(item["text"] for item in pending)
        combined["bbox"] = [min(item["bbox"][0] for item in pending), min(item["bbox"][1] for item in pending),
                            max(item["bbox"][2] for item in pending), max(item["bbox"][3] for item in pending)]
        yield from _text_chunks(combined, max_tokens, overlap_tokens)
        pending.clear()

    for element in elements:
        modality = element["modality"]
        if modality == "text":
            if pending and (element["page"], element["section"], element.get("subsection"), element["extraction_method"]) != (pending[-1]["page"], pending[-1]["section"], pending[-1].get("subsection"), pending[-1]["extraction_method"]):
                yield from flush_text()
            pending.append(element)
            continue
        yield from flush_text()
        if modality == "table":
            yield from _table_chunks(element, max_tokens)
        elif modality == "image":
            description = "\n".join(part for part in (
                f"Image on page {element['page']} in {element['section']}.",
                f"Caption: {element['caption']}" if element.get("caption") else "",
                f"OCR: {element['ocr_text']}" if element.get("ocr_text") else "",
            ) if part)
            yield _make_chunk(element, 1, description, chunk_strategy="image_with_ocr")
        else:
            raise ValueError(f"Unknown modality: {modality}")
    yield from flush_text()
