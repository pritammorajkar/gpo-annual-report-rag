import logging
from dataclasses import dataclass
from pathlib import Path
import pymupdf

logger = logging.getLogger(__name__)

@dataclass
class PageText:
    page_number: int  # 1-based physical page in the PDF (what a PDF viewer shows), not the printed folio
    text: str

def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. Download the GPO FY 2018 Annual Report and save it there."
        )
    try:
        document = pymupdf.open(pdf_path)
    except RuntimeError as exc:  # pymupdf.FileDataError / EmptyFileError for corrupt, empty, or non-PDF files
        raise ValueError(f"Could not read {pdf_path} as a PDF: {exc}") from exc
    with document:
        if document.needs_pass:
            raise ValueError(f"{pdf_path} is password-protected. Provide an unencrypted copy.")
        all_pages = [PageText(index + 1, page.get_text("text").strip()) for index, page in enumerate(document)]
    pages = [page for page in all_pages if page.text]
    if not pages:
        raise ValueError(
            f"No selectable text found in {pdf_path} ({len(all_pages)} pages). "
            "This may be a scanned PDF; OCR is needed."
        )
    empty = [page.page_number for page in all_pages if not page.text]
    if empty:
        logger.warning(
            "%d of %d PDF pages have no extractable text (blank or scanned) and were skipped: %s",
            len(empty), len(all_pages), empty,
        )
    return pages
