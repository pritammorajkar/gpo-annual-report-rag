# GPO Annual Report RAG

A Python Retrieval-Augmented Generation (RAG) application for Q&A over the U.S. Government Publishing Office FY 2018 Annual Report.

## Acceptance criteria covered
- PDF ingestion and page-aware text extraction
- Overlapping chunking with page metadata
- Embedding generation
- Persistent ChromaDB vector storage
- Similarity retrieval
- Grounded response generation with PDF page citations
- Evaluation against the three provided sample questions

## Architecture
`PDF -> PyMuPDF -> overlapping chunks -> OpenAI or Gemini embeddings -> ChromaDB -> top-k retrieval -> grounded answer + page citations`

## Setup
Requires Python 3.10+ (tested with 3.12) and an OpenAI or Gemini API key.

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

The report PDF is expected at `data/gpo_2018_annual_report.pdf` (official GPO publication or the course-supplied copy).

Open `.env` in an editor and set `OPENAI_API_KEY=...`, or set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`. `.env` is git-ignored; never commit it or paste the key into chats or logs. Then ingest:

```bash
python -m src.ingest
```

Ingestion embeds every chunk first and only then replaces the Chroma collection in `storage/chroma`, so re-running it never leaves duplicate or stale chunks, and an API failure leaves the previous index intact. Re-run it whenever you change the PDF, chunk settings, provider, or embedding model/dimensions.

Ask one question or use interactive mode:

```bash
python -m src.ask "How many depository libraries did GPO staff visit in FY 2018?"
python -m src.ask
```

Run the acceptance-question evaluation:

```bash
python -m src.evaluate
```

Results (answer, retrieved pages, cited pages, citations to pages that were not retrieved, and the full retrieved passages) are written to `evaluation_results.json`. This is a smoke test, not a guarantee of correctness: compare answers to the cited PDF pages before marking the AC passed.

Citations such as `[PDF p. 5]` are physical PDF page numbers (the page number a PDF viewer shows), not the printed page numbers. In the narrative part of this report the printed number is 2 lower (PDF p. 5 is printed page 3).

Run unit tests (no API key or network needed; API calls are mocked):

```bash
python -m unittest discover -s tests -v
```

## Configuration
`LLM_PROVIDER` selects `openai` (default) or `gemini`. Gemini is called through its OpenAI-compatible endpoint, so both use the `openai` SDK.

- OpenAI: `OPENAI_API_KEY`, `CHAT_MODEL` (default `gpt-4o-mini`), `EMBEDDING_MODEL` (default `text-embedding-3-small`).
- Gemini: `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL` (default `gemini-3.5-flash`), `GEMINI_EMBEDDING_MODEL` (default `gemini-embedding-2`).
- Both: `EMBEDDING_DIMENSIONS` (optional shorter vectors, e.g. 1536; blank = model default), `CHUNK_SIZE` (default 1000 characters), `CHUNK_OVERLAP` (default 180 characters), and `TOP_K` (default 5).

Changing the provider, embedding model, or `EMBEDDING_DIMENSIONS` requires re-running `python -m src.ingest`; querying a mismatched index fails with a clear message.

## Current scope and future extensions
This version targets Use Case 1 (plain-text GPO report). For scanned PDFs, add OCR. For the World Bank report, preserve chapter/section headings in metadata. For the iMac report, add table/chart extraction and OCR or vision-based extraction so visual facts and footnotes are indexed too.
