"""Extract positioned text, tables, raster figures, and OCR from PDFs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterator

import pymupdf


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\ufffd", "")
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _bbox(value: Any) -> list[float]:
    return [round(float(x), 2) for x in value]


def _overlap_fraction(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    if a.is_empty or a.get_area() == 0:
        return 0.0
    return (a & b).get_area() / a.get_area()


def _ocr(image: bytes, engine: Any) -> tuple[str, float | None]:
    result = engine(image)
    lines = list(result.txts or ())
    scores = list(result.scores or ())
    return clean_text("\n".join(lines)), (round(sum(scores) / len(scores), 4) if scores else None)


def _render(page: pymupdf.Page, rect: pymupdf.Rect, dpi: int) -> bytes:
    return page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False).tobytes("png")


def _tables(page: pymupdf.Page) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for strategy in ("lines", "text"):
        if found:
            break
        for table in page.find_tables(strategy=strategy).tables:
            rows = [[clean_text(cell or "") for cell in row] for row in table.extract()]
            if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
                continue
            if sum(bool(cell) for row in rows for cell in row) < 4:
                continue
            rect = pymupdf.Rect(table.bbox)
            if any(_overlap_fraction(rect, pymupdf.Rect(item["bbox"])) > 0.7 for item in found):
                continue
            found.append({"bbox": _bbox(rect), "rows": rows, "method": f"pymupdf_{strategy}"})
    return found


def _heading(block: dict[str, Any], baseline: float) -> str | None:
    lines = block.get("lines", [])
    text = clean_text("\n".join("".join(span.get("text", "") for span in line.get("spans", [])) for line in lines))
    if not text or len(text) > 110 or len(text.split()) > 14 or not re.match(r"^[A-Za-z]", text):
        return None
    sizes = [span.get("size", 0) for line in lines for span in line.get("spans", [])]
    if sizes and max(sizes) >= max(12, baseline * 1.24):
        return text
    return None


def _outline_sections(doc: pymupdf.Document) -> dict[int, str]:
    try:
        toc = doc.get_toc()
    except Exception:
        return {}
    depth = 2 if any(level == 2 for level, _, _ in toc) else 1
    labels = {
        "Board Report": "Board Report", "BRSR": "Business Responsibility and Sustainability Report",
        "MDA": "Management Discussion and Analysis", "CG": "Corporate Governance",
        "Consolidated": "Consolidated Financial Statements", "Standalone": "Standalone Financial Statements",
        "statistics": "Financial Statistics", "Notice": "Notice",
    }
    return {
        page: next((label for marker, label in labels.items() if marker.lower() in title.lower()), clean_text(title))
        for level, title, page in toc if level == depth and 1 <= page <= doc.page_count
    }


def parse_pdf(
    pdf_path: Path,
    image_dir: Path,
    *,
    start_page: int = 1,
    max_pages: int | None = None,
    ocr_mode: str = "auto",
    dpi: int = 150,
    min_image_side: int = 80,
) -> Iterator[dict[str, Any]]:
    """Yield elements in page order. OCR runs on visible figures and sparse pages."""
    if ocr_mode not in {"auto", "all", "off"}:
        raise ValueError("ocr_mode must be auto, all, or off")
    if start_page < 1:
        raise ValueError("start_page must be positive")
    pdf_path = pdf_path.resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    engine = None

    def get_engine() -> Any:
        nonlocal engine
        if engine is None:
            from rapidocr import RapidOCR

            engine = RapidOCR(params={"Rec.lang_type": "en"})
        return engine

    with pymupdf.open(pdf_path) as doc:
        if start_page > doc.page_count:
            raise ValueError(f"start_page {start_page} exceeds PDF page count {doc.page_count}")
        section_starts = _outline_sections(doc)
        section = "Overview"
        end_page = min(doc.page_count, start_page - 1 + max_pages) if max_pages else doc.page_count
        for page_index in range(start_page - 1, end_page):
            page = doc[page_index]
            page_number = page_index + 1
            section = next((title for number, title in sorted(section_starts.items(), reverse=True) if number <= page_number), section)
            page_rect = page.rect
            tables = _tables(page)
            layout = page.get_text("dict", sort=True)
            text_blocks = [block for block in layout["blocks"] if block.get("type") == 0]
            sizes = sorted(
                span.get("size", 0)
                for block in text_blocks
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            )
            baseline = sizes[len(sizes) // 2] if sizes else 10
            native_text = clean_text(page.get_text())
            common = {
                "source": pdf_path.name,
                "source_path": str(pdf_path),
                "source_sha256": source_hash,
                "page": page_number,
                "page_label": page.get_label() or str(page_number),
                "page_width": round(page_rect.width, 2),
                "page_height": round(page_rect.height, 2),
            }
            page_elements: list[dict[str, Any]] = []
            for index, table in enumerate(tables, 1):
                page_elements.append({
                    **common, "id": f"{source_hash[:12]}-p{page_number:04d}-table{index:03d}",
                    "modality": "table", "section": section, "table_index": index,
                    "bbox": table["bbox"], "rows": table["rows"],
                    "extraction_method": table["method"],
                })
            text_index = 0
            for block in text_blocks:
                rect = pymupdf.Rect(block["bbox"])
                if any(_overlap_fraction(rect, pymupdf.Rect(table["bbox"])) > 0.5 for table in tables):
                    continue
                value = clean_text("\n".join("".join(span.get("text", "") for span in line.get("spans", [])) for line in block.get("lines", [])))
                if not value:
                    continue
                detected_heading = _heading(block, baseline)
                text_index += 1
                page_elements.append({
                    **common, "id": f"{source_hash[:12]}-p{page_number:04d}-text{text_index:03d}",
                    "modality": "text", "section": section, "text": value,
                    "bbox": _bbox(rect), "heading": detected_heading,
                    "extraction_method": "pymupdf_native",
                })
            image_index = 0
            image_rects: list[pymupdf.Rect] = []
            for block in layout["blocks"]:
                if block.get("type") != 1:
                    continue
                rect = pymupdf.Rect(block["bbox"]) & page_rect
                if rect.width < min_image_side or rect.height < min_image_side:
                    continue
                if any(_overlap_fraction(rect, previous) > 0.9 for previous in image_rects):
                    continue
                image_rects.append(rect)
                image_index += 1
                image_path = image_dir / f"{pdf_path.stem}_p{page_number:04d}_image{image_index:03d}.png"
                image_bytes = _render(page, rect, dpi)
                image_path.write_bytes(image_bytes)
                full_page_overlay = rect.get_area() / page_rect.get_area() > 0.8 and len(native_text) > 100
                ocr_text, ocr_confidence = ("", None)
                if ocr_mode != "off" and not full_page_overlay:
                    ocr_text, ocr_confidence = _ocr(image_bytes, get_engine())
                nearby = [item for item in page_elements if item["modality"] == "text"
                          and 0 <= pymupdf.Rect(item["bbox"]).y0 - rect.y1 <= 65]
                caption = min(nearby, key=lambda item: pymupdf.Rect(item["bbox"]).y0)["text"] if nearby else ""
                page_elements.append({
                    **common, "id": f"{source_hash[:12]}-p{page_number:04d}-image{image_index:03d}",
                    "modality": "image", "section": section, "bbox": _bbox(rect),
                    "image_path": str(image_path.resolve()), "caption": caption[:500],
                    "ocr_text": ocr_text, "ocr_confidence": ocr_confidence,
                    "extraction_method": "pymupdf_render+rapidocr" if ocr_text else "pymupdf_render",
                })
            if ocr_mode == "all" or (ocr_mode == "auto" and len(native_text) < 40):
                page_image = _render(page, page_rect, dpi)
                ocr_text, ocr_confidence = _ocr(page_image, get_engine())
                if ocr_text and (ocr_mode == "all" or not any(item["modality"] in {"text", "table"} for item in page_elements)):
                    page_elements.append({
                        **common, "id": f"{source_hash[:12]}-p{page_number:04d}-ocr001",
                        "modality": "text", "section": section, "text": ocr_text,
                        "bbox": _bbox(page_rect), "heading": None,
                        "ocr_confidence": ocr_confidence, "extraction_method": "rapidocr_page",
                    })
            for item in page_elements:
                if item["modality"] == "table":
                    table_rect = pymupdf.Rect(item["bbox"])
                    above = [candidate for candidate in page_elements if candidate["modality"] == "text"
                             and len(candidate["text"]) <= 90
                             and 0 <= table_rect.y0 - pymupdf.Rect(candidate["bbox"]).y1 <= 55
                             and _overlap_fraction(pymupdf.Rect(candidate["bbox"]), table_rect + (0, -55, 0, 0)) > 0]
                    item["context_above"] = " | ".join(candidate["text"].replace("\n", " ") for candidate in sorted(above, key=lambda candidate: (candidate["bbox"][1], candidate["bbox"][0])))
            subsection = None
            for item in sorted(page_elements, key=lambda item: (item["bbox"][1], item["bbox"][0], item["id"])):
                if item.get("heading"):
                    subsection = item["heading"]
                if subsection:
                    item["subsection"] = subsection
                yield item
