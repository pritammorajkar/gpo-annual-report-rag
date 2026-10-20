# Implementation Report — GPO Annual Report RAG

**Date:** 2026-10-18
**Status: End-to-end run complete.** The real PDF was indexed with Gemini embeddings and all three sample questions were answered from retrieved evidence. Every claim in every answer was checked against the PDF text and is supported. Unit tests: 39/39 pass.

## Environment
- macOS. The system `python3` is 3.9.6, below the README's minimum, so I used a project `.venv` with Python 3.12.14.
- Installed versions: openai 3.35.1, chromadb 1.5.9, pymupdf 1.28.2, python-dotenv 1.2.4. No new dependencies were added.
- Provider: **Gemini**, called through Google's OpenAI-compatible endpoint with the existing `openai` SDK.
  - Chat model: `gemini-3.5-flash`.
  - Embedding model: `gemini-embedding-2` with `EMBEDDING_DIMENSIONS=1536`.
  - I confirmed both models exist via the models list endpoint.
- The key lives only in `.env`. That file is git-ignored and set to `chmod 600`. A scan of every project file except `.env` and `.venv` found no copy of it, including the Chroma store and `evaluation_results.json`.
- PDF: `data/gpo_2018_annual_report.pdf` (82 pages, selectable text). It was not modified; its SHA-256 is unchanged.

## What was already working
- The pipeline shape: PyMuPDF → per-page overlapping chunks → embeddings → persistent ChromaDB with cosine distance → top-k retrieval → grounded prompt with `[PDF p. N]` labels.
- Page-aware chunking (chunks never span pages), unique chunk IDs, batched embedding calls, an empty-index check, and the CLI entry points.

## What I changed and why
| Area | Change | Why |
|---|---|---|
| Error handling (`vector_store.py`) | Added `RAGError` and an `openai_errors()` wrapper. It maps 401, 403, 404, 429, 400, connection and other errors to short, provider-named messages, raised `from None`. It also redacts the key from 400 messages. | **Secret leak.** OpenAI's 401 message echoes part of the key (verified). The old `ask.py` printed it, and `evaluate.py` would have shown it in a traceback. Gemini reports a bad key as 400 "Invalid Auth key." |
| Re-ingestion (`vector_store.py`) | Every chunk is embedded **before** Chroma is touched; then the collection is dropped and recreated. | The old code deleted the index first, so an API failure left it empty. It also broke when the embedding dimension changed. |
| Index consistency (`vector_store.py`) | The collection stores `embedding_model` as `model@dims`, and `retrieve` refuses to query a mismatched index. `source` and `chunk_id` are added to metadata. `embedding_function=None` is passed explicitly. | Prevents silent garbage retrieval. Makes results traceable. |
| Gemini support (`config.py`, `vector_store.py`) | `LLM_PROVIDER=openai\|gemini`, provider-specific key and model variables, optional `EMBEDDING_DIMENSIONS`, and a configurable base URL. Embeddings are re-ordered with `item.index or 0`. | You asked for the Gemini key to be used. Gemini omits `index` on the first embedding (verified); the old sort would have crashed comparing `None` with `int`. |
| Chunking (`chunking.py`) | Chunk edges snap to spaces. Chunks are still ≤ `chunk_size`, overlap is ≥ `overlap`, and there is a hard-cut fallback for over-long words. | Hard character cuts could split figures such as `$375 mil│lion`. |
| Extraction (`document.py`) | `fitz` → `pymupdf`. Clear errors for missing, corrupt, empty and encrypted files. Warns about blank or scanned pages. | The `fitz` name is deprecated and printed a warning on every run. Unreadable files gave raw errors, and skipped pages went unreported. |
| Grounding check (`qa.py`) | Returns `cited_pages` and `unsupported_citations` (pages cited but not retrieved). | Checks that citations point to evidence that was actually retrieved. |
| Config (`config.py`) | `.env` is loaded from the project root. Blank values fall back to defaults. Clear errors for non-integer numbers and an unknown provider. | `CHAT_MODEL=` used to send `""` to the API, and `CHUNK_SIZE=abc` gave a cryptic `int()` error. |
| CLIs | Clean errors with non-zero exit codes. Ingest reports pages, chunks, provider and model. Evaluate saves cited pages, warnings, the full passages and the run config (no secrets). | Usable diagnostics, and results you can check by hand. |
| Docs and setup | README, `data/README.md` and `.env.example` now cover both providers. `pymupdf>=1.24.3`. | The docs were out of step with the code. |

## Commands executed and outcomes
| Command | Outcome |
|---|---|
| `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` | Succeeded. |
| `python -m unittest discover -s tests -v` (before changes) | 2 tests, OK. |
| `python -m unittest discover -s tests -v` (final, with `.env` present, and again with `.env` temporarily moved aside) | **39 tests, OK** both times. |
| `python -m src.ingest` / `src.ask` / `src.evaluate` with no key | Exit 1 with clear messages (`OPENAI_API_KEY is missing…` / `Vector index is empty…`). |
| `OPENAI_API_KEY=<fake> python -m src.ingest` | Exit 1: `OpenAI rejected the API key (401)…`. **0 key fragments in the output.** |
| `CHUNK_SIZE=big …` / `CHUNK_SIZE=100 CHUNK_OVERLAP=100 …` | Clear validation errors. |
| Gemini probe (scratch script) | Key valid. `gemini-3.5-flash` and `gemini-embedding-2` listed. `dimensions=1536` honoured (default 3072). `temperature=0` accepted. |
| **`python -m src.ingest`** (Gemini) | Exit 0: `Extracted text from 81 pages; created 237 chunks … Embedding with Gemini gemini-embedding-2@1536 … Indexed 237 chunks.` (12 s). |
| Reopened the store from a new process | 237 records, 237 unique IDs, 1536-dimension vectors, 81 pages covered, metadata `{'hnsw:space': 'cosine', 'embedding_model': 'gemini-embedding-2@1536'}`. |
| **`python -m src.evaluate`** | Exit 0. All 3 answered with citations, 0 unsupported citations (16 s). Results are in `evaluation_results.json`. |
| `python -m src.ask "What was GPO's total revenue in FY 2025?"` (out of scope) | `I couldn't verify that from the retrieved report excerpts.` The grounded refusal works. |

## Unit-test results — 39/39 pass
- **Chunking (8):** size limit, minimum overlap, word boundaries, full coverage, page metadata, chunks never spanning pages, unique IDs, blank pages, the long-word fallback, and invalid parameters.
- **Document (6):** physical page numbers with blank pages skipped, missing, corrupt, empty and encrypted PDFs, the "OCR needed" case, and a page-metadata check on the real report.
- **Vector store (17):** uses a temporary Chroma directory and a fake embedder. Covers persistence across a reopen, metadata, ranking and page numbers, the `top_k` cap, re-ingest without duplicates or stale records, the index surviving an embedding failure, a dimension change, model mismatch, the empty index, batching and ordering, Gemini's missing `index`, `dimensions` pass-through, the provider-specific missing-key message, the key never being echoed (401), and key redaction (400).
- **QA and config (8):** page-labelled prompt, cited and unsupported-citation detection, citation parsing, chat-error translation, and env parsing.

## Ingestion results
- **Pages:** 82 in the PDF, 81 with text. PDF page 2 is a blank inside cover and is reported as skipped.
- **Chunks:** 237 at `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=180`. Lengths are 93 / 994 / 1000 characters (minimum / median / maximum).
- **Index:** `storage/chroma` holds 237 records, persisted and queryable.

## Sample-question results
Page numbers are physical PDF pages. In the narrative section, the printed page number is 2 lower.

### 1. How many depository libraries did GPO staff visit in FY 2018?
- **Answer:** "In FY 2018, GPO staff visited 166 depository libraries [PDF p. 5] [PDF p. 21]."
- **Retrieved pages:** 5, 19, 21. **Cited:** 5, 21.
- **Supported: Yes.** p. 5: "in FY 2018 GPO staff visited 166 depository libraries in 25 states". p. 21 repeats the figure.
- **Remaining issue:** none. (The answer leaves out the "25 states" detail, which the question didn't ask for.)

### 2. What new digital system replaced FDsys, and when was it fully deployed?
- **Answer:** govinfo [p. 4, 22], with these milestones:
  - introduced in beta in early 2016 [p. 22];
  - moved out of beta in December 2017 [p. 22];
  - full-scale deployment during FY 2018 [p. 4];
  - scheduled to completely retire and replace FDsys in December 2018 [p. 4, 22].
- **Retrieved pages:** 4, 13, 22. **Cited:** 4, 22.
- **Supported: Yes**, every milestone matches the text verbatim:
  - p. 4: "full-scale deployment of govinfo … the successor to our FDsys website" and "goal of retiring FDsys in December 2018".
  - p. 22: "In early 2016 … introduction of govinfo. Rolled out initially in beta", "moved out of beta in December 2017", and "scheduled to be completely retired and replaced by govinfo in December 2018".
- **Remaining issue:** none. The report gives no single "fully deployed" date, and the answer correctly presents the timeline instead of inventing one.

### 3. How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?
- **Answer:** "more than $375 million" [p. 5], with the work valued at "approximately $375.7 million" [p. 17].
- **Retrieved pages:** 5, 14, 17, 23. **Cited:** 5, 17.
- **Supported: Yes.**
  - p. 5: "On behalf of our Federal agency partners, GPO procured more than $375 million worth of printing and other information products in FY 2018."
  - p. 17: "In FY 2018, this work was valued at approximately $375.7 million and represented 84,111 orders."
- **Remaining issue:** none.

## How to run in the future
```bash
cd gpo_annual_report_rag
source .venv/bin/activate          # first time: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# .env must contain LLM_PROVIDER=gemini and GEMINI_API_KEY=...   (or LLM_PROVIDER=openai and OPENAI_API_KEY=...)
python -m unittest discover -s tests -v
python -m src.ingest               # 81 pages, 237 chunks; re-run after changing PDF/chunking/provider/embedding settings
python -m src.evaluate             # 3 sample questions -> evaluation_results.json
python -m src.ask "How many depository libraries did GPO staff visit in FY 2018?"
python -m src.ask                  # interactive mode
```

## Remaining notes and blockers
- **No blockers.**
- **Rotate the Gemini key.** It was shared in a chat transcript, so rotate it once testing is done, then update `.env`.
- **OpenAI path partly untested.** It is unit-tested and its 401 path was tested live with a fake key, but there was no valid OpenAI key to run it end to end.
- **Temperature on Gemini 3.x.** Google recommends the default temperature for Gemini 3.x models. The app keeps `temperature=0`, which was accepted and gave correct, grounded answers here.
- **Printed page numbers are not tracked.** Citations use physical PDF pages, because the PDF's own page labels are malformed in the financial section.
- **Minor:** an invalid integer in `.env` is reported at import time with a traceback (the message is still clear).
