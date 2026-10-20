from contextlib import contextmanager
import chromadb
import openai
from openai import OpenAI
from .chunking import Chunk
from .config import (
    API_BASE_URL, API_KEY, API_KEY_NAME, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL, LLM_PROVIDER, PDF_PATH, PROVIDER_NAME,
)

class RAGError(RuntimeError):
    """A configuration, index, or API problem whose message is safe to show to users."""

def get_openai_client() -> OpenAI:
    """OpenAI SDK client for the configured provider (Gemini uses its OpenAI-compatible endpoint)."""
    if not API_KEY:
        raise RAGError(
            f"{API_KEY_NAME} is missing (LLM_PROVIDER={LLM_PROVIDER}). "
            f"Copy .env.example to .env and set {API_KEY_NAME} there."
        )
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def _redact(message: str) -> str:
    return message.replace(API_KEY, "<redacted>") if API_KEY else message

@contextmanager
def openai_errors(model: str):
    """Turn API errors into short messages. `from None` drops the original error,
    whose text can echo part of the API key (e.g. OpenAI's 401)."""
    try:
        yield
    except openai.AuthenticationError:
        raise RAGError(f"{PROVIDER_NAME} rejected the API key (401). Check {API_KEY_NAME} in your .env file.") from None
    except openai.PermissionDeniedError:
        raise RAGError(f"{PROVIDER_NAME} denied access to model '{model}' (403). Check the key's permissions.") from None
    except openai.NotFoundError:
        raise RAGError(f"{PROVIDER_NAME} model '{model}' was not found (404). Check the model name in your .env file.") from None
    except openai.RateLimitError:
        raise RAGError(f"{PROVIDER_NAME} rate limit or quota exceeded (429). Check your plan and billing, or retry later.") from None
    except openai.BadRequestError as exc:
        # Gemini reports an invalid key as 400 "Invalid Auth key."
        raise RAGError(f"{PROVIDER_NAME} rejected the request for model '{model}' (400): {_redact(exc.message)}") from None
    except openai.APIConnectionError:
        raise RAGError(f"Could not reach the {PROVIDER_NAME} API. Check your network connection.") from None
    except openai.APIStatusError as exc:
        raise RAGError(f"{PROVIDER_NAME} API error ({exc.status_code}) while using model '{model}'.") from None

def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    client = get_openai_client()
    options = {"dimensions": EMBEDDING_DIMENSIONS} if EMBEDDING_DIMENSIONS else {}
    vectors: list[list[float]] = []
    with openai_errors(EMBEDDING_MODEL):
        for start in range(0, len(texts), batch_size):
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[start:start + batch_size], **options)
            # Gemini omits `index` for the first item (protobuf default 0).
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index or 0))
    if len(vectors) != len(texts):
        raise RAGError(f"Expected {len(texts)} embeddings from {PROVIDER_NAME} but received {len(vectors)}.")
    return vectors

def embedding_id() -> str:
    """Identifies the vector space; an index can only be queried with the same model and size."""
    return f"{EMBEDDING_MODEL}@{EMBEDDING_DIMENSIONS}" if EMBEDDING_DIMENSIONS else EMBEDDING_MODEL

def _chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_collection():
    # Embeddings always come from OpenAI, so Chroma's own embedding function is disabled.
    return _chroma_client().get_or_create_collection(name=COLLECTION_NAME, embedding_function=None)

def replace_index(chunks: list[Chunk], source: str = PDF_PATH.name) -> int:
    if not chunks:
        raise ValueError("No chunks to index.")
    # Embed first so an API failure leaves the existing index untouched.
    vectors = embed_texts([chunk.text for chunk in chunks])
    client = _chroma_client()
    # Recreate the collection so stale chunks, or vectors from a different embedding model, never survive.
    client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=None)
    client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_id()},
        embedding_function=None,
    )
    for start in range(0, len(chunks), 500):
        batch = chunks[start:start + 500]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=vectors[start:start + 500],
            metadatas=[{"page_number": chunk.page_number, "source": source} for chunk in batch],
        )
    return collection.count()

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise RAGError("Vector index is empty. Run `python -m src.ingest` first.")
    indexed_model = (collection.metadata or {}).get("embedding_model")
    if indexed_model and indexed_model != embedding_id():
        raise RAGError(
            f"The index was built with '{indexed_model}' but the configured embedding model is '{embedding_id()}'. "
            "Re-run `python -m src.ingest`."
        )
    vector = embed_texts([question])[0]
    result = collection.query(query_embeddings=[vector], n_results=min(top_k, count), include=["documents", "metadatas", "distances"])
    hits = []
    for i, document in enumerate(result["documents"][0]):
        hits.append({
            "chunk_id": result["ids"][0][i],
            "text": document,
            "page_number": result["metadatas"][0][i]["page_number"],
            "distance": result["distances"][0][i],
        })
    return hits
