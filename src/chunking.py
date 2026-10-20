from dataclasses import dataclass
from .document import PageText

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int

def chunk_pages(pages: list[PageText], chunk_size: int = 1000, overlap: int = 180) -> list[Chunk]:
    """Split each page into chunks of at most `chunk_size` characters.

    Chunks never span pages, so every chunk has exactly one source page. Consecutive chunks
    share at least `overlap` characters. Boundaries snap to spaces so words and numbers are
    not cut in half; a single word longer than the window falls back to a hard cut.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    chunks: list[Chunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        start, part = 0, 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + 1, end + 1)
                if boundary - overlap > start:
                    end = boundary
            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(f"page-{page.page_number}-chunk-{part}", content, page.page_number))
            if end == len(text):
                break
            next_start = end - overlap
            word_start = text.rfind(" ", 0, next_start) + 1
            start = word_start if word_start > start else next_start
            part += 1
    return chunks
