import tempfile
import unittest
from pathlib import Path
import pymupdf
from src.chunking import chunk_pages
from src.config import PDF_PATH
from src.document import extract_pdf_pages

def make_pdf(path: Path, page_texts: list[str], **save_options) -> Path:
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path, **save_options)
    return path

class ExtractPdfPagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_keeps_physical_page_numbers_and_skips_blank_pages(self):
        pdf = make_pdf(self.dir / "doc.pdf", ["Alpha page", "", "Gamma page"])
        with self.assertLogs("src.document", level="WARNING") as logs:
            pages = extract_pdf_pages(pdf)
        self.assertEqual([(p.page_number, p.text) for p in pages], [(1, "Alpha page"), (3, "Gamma page")])
        self.assertIn("1 of 3 PDF pages", logs.output[0])

    def test_missing_pdf_raises_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "PDF not found"):
            extract_pdf_pages(self.dir / "missing.pdf")

    def test_pdf_without_text_reports_ocr_needed(self):
        pdf = make_pdf(self.dir / "scanned.pdf", ["", ""])
        with self.assertRaisesRegex(ValueError, "OCR"):
            extract_pdf_pages(pdf)

    def test_corrupt_and_empty_files_raise_value_error(self):
        for name, data in [("corrupt.pdf", b"not a pdf"), ("empty.pdf", b"")]:
            path = self.dir / name
            path.write_bytes(data)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Could not read"):
                extract_pdf_pages(path)

    def test_password_protected_pdf_raises_value_error(self):
        pdf = make_pdf(self.dir / "locked.pdf", ["Secret"], encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
        with self.assertRaisesRegex(ValueError, "password"):
            extract_pdf_pages(pdf)

@unittest.skipUnless(PDF_PATH.exists(), "GPO report PDF not present")
class RealReportTests(unittest.TestCase):
    def test_report_extracts_with_valid_page_metadata(self):
        with pymupdf.open(PDF_PATH) as document:
            page_count = document.page_count
        pages = extract_pdf_pages(PDF_PATH)
        numbers = [page.page_number for page in pages]
        self.assertEqual(numbers, sorted(set(numbers)))
        self.assertTrue(1 <= numbers[0] and numbers[-1] <= page_count)
        chunks = chunk_pages(pages, chunk_size=1000, overlap=180)
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))
        self.assertTrue(all(len(chunk.text) <= 1000 for chunk in chunks))
        self.assertTrue({chunk.page_number for chunk in chunks} <= set(numbers))

if __name__ == "__main__":
    unittest.main()
