import logging
import sys
from .config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, PROVIDER_NAME
from .document import extract_pdf_pages
from .chunking import chunk_pages
from .vector_store import RAGError, embedding_id, replace_index

def main() -> int:
    logging.basicConfig(format="%(levelname)s: %(message)s")
    try:
        pages = extract_pdf_pages(PDF_PATH)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        print(f"Extracted text from {len(pages)} pages; created {len(chunks)} chunks "
              f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}). Embedding with {PROVIDER_NAME} {embedding_id()}...")
        count = replace_index(chunks)
    except (FileNotFoundError, ValueError, RAGError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Indexed {count} chunks.")
    print(f"Persistent vector store: {CHROMA_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
