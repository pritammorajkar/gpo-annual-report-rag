import math
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import httpx
import openai
from chromadb.api.client import SharedSystemClient
from src import vector_store
from src.chunking import Chunk
from src.vector_store import RAGError, openai_errors

def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words vectors so similar texts are close, without calling OpenAI."""
    vectors = []
    for text in texts:
        vector = [0.0] * 64
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors

CHUNKS = [
    Chunk("page-3-chunk-0", "libraries visited by staff across many states", 3),
    Chunk("page-8-chunk-0", "new website replaced the old system in december", 8),
    Chunk("page-12-chunk-0", "printing procured on behalf of agencies worth millions", 12),
]
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")

def api_error(cls, status: int, message: str):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)

class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.object(vector_store, "CHROMA_DIR", Path(self.tmp.name) / "chroma"),
            mock.patch.object(vector_store, "embed_texts", side_effect=fake_embed),
            mock.patch.object(vector_store, "EMBEDDING_MODEL", "test-embedding-model"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        SharedSystemClient.clear_system_cache()
        self.tmp.cleanup()

    def test_empty_index_raises_helpful_error(self):
        with self.assertRaisesRegex(RAGError, "python -m src.ingest"):
            vector_store.retrieve("anything")

    def test_index_persists_chunks_with_page_metadata(self):
        self.assertEqual(vector_store.replace_index(CHUNKS, source="report.pdf"), 3)
        SharedSystemClient.clear_system_cache()  # reopen from disk like a new process would
        stored = vector_store.get_collection().get(include=["metadatas", "documents"])
        by_id = dict(zip(stored["ids"], stored["metadatas"]))
        self.assertEqual(by_id["page-12-chunk-0"], {"page_number": 12, "source": "report.pdf"})
        self.assertEqual(set(stored["documents"]), {chunk.text for chunk in CHUNKS})

    def test_retrieve_returns_most_relevant_passage_with_page_number(self):
        vector_store.replace_index(CHUNKS)
        hits = vector_store.retrieve("how much printing was procured for agencies", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["page_number"], 12)
        self.assertEqual(hits[0]["chunk_id"], "page-12-chunk-0")
        self.assertLessEqual(hits[0]["distance"], hits[1]["distance"])
        self.assertEqual(set(hits[0]), {"chunk_id", "text", "page_number", "distance"})

    def test_top_k_larger_than_index_is_capped(self):
        vector_store.replace_index(CHUNKS)
        self.assertEqual(len(vector_store.retrieve("libraries", top_k=50)), 3)

    def test_reingest_replaces_records_without_duplicates_or_stale_chunks(self):
        vector_store.replace_index(CHUNKS)
        vector_store.replace_index(CHUNKS)
        self.assertEqual(vector_store.get_collection().count(), 3)
        new_chunks = [Chunk("page-1-chunk-0", "completely new content", 1)]
        self.assertEqual(vector_store.replace_index(new_chunks), 1)
        self.assertEqual(vector_store.get_collection().get()["ids"], ["page-1-chunk-0"])

    def test_embedding_failure_leaves_existing_index_intact(self):
        vector_store.replace_index(CHUNKS)
        with mock.patch.object(vector_store, "embed_texts", side_effect=RAGError("API down")):
            with self.assertRaises(RAGError):
                vector_store.replace_index([Chunk("page-1-chunk-0", "new", 1)])
        self.assertEqual(vector_store.get_collection().count(), 3)

    def test_reingest_with_different_embedding_dimension_succeeds(self):
        vector_store.replace_index(CHUNKS)
        with mock.patch.object(vector_store, "embed_texts", side_effect=lambda texts: [[1.0, 0.0, 0.0] for _ in texts]):
            self.assertEqual(vector_store.replace_index(CHUNKS), 3)

    def test_embedding_model_mismatch_is_reported(self):
        vector_store.replace_index(CHUNKS)
        with mock.patch.object(vector_store, "EMBEDDING_MODEL", "other-model"):
            with self.assertRaisesRegex(RAGError, "Re-run"):
                vector_store.retrieve("libraries")

    def test_empty_question_and_chunk_list_are_rejected(self):
        with self.assertRaises(ValueError):
            vector_store.retrieve("   ")
        with self.assertRaises(ValueError):
            vector_store.replace_index([])

def fake_embeddings_client(first_index=0):
    client = mock.MagicMock()
    def create(model, input, **options):
        data = [SimpleNamespace(index=i, embedding=[float(len(text))]) for i, text in enumerate(input)]
        data[0].index = first_index
        return SimpleNamespace(data=list(reversed(data)))  # API order is not relied on
    client.embeddings.create.side_effect = create
    return client

class EmbedTextsTests(unittest.TestCase):
    def embed(self, client, texts, dimensions=None, batch_size=2):
        with mock.patch.object(vector_store, "get_openai_client", return_value=client), \
                mock.patch.object(vector_store, "EMBEDDING_DIMENSIONS", dimensions):
            return vector_store.embed_texts(texts, batch_size=batch_size)

    def test_batches_requests_and_keeps_input_order(self):
        client = fake_embeddings_client()
        vectors = self.embed(client, ["a", "bb", "ccc", "dddd", "eeeee"])
        self.assertEqual(client.embeddings.create.call_count, 3)
        self.assertEqual(vectors, [[1.0], [2.0], [3.0], [4.0], [5.0]])

    def test_handles_gemini_omitting_index_zero(self):
        vectors = self.embed(fake_embeddings_client(first_index=None), ["a", "bb", "ccc"], batch_size=3)
        self.assertEqual(vectors, [[1.0], [2.0], [3.0]])

    def test_dimensions_are_sent_only_when_configured(self):
        client = fake_embeddings_client()
        self.embed(client, ["a"], dimensions=1536)
        self.assertEqual(client.embeddings.create.call_args.kwargs["dimensions"], 1536)
        client = fake_embeddings_client()
        self.embed(client, ["a"])
        self.assertNotIn("dimensions", client.embeddings.create.call_args.kwargs)

    def test_missing_api_key_names_the_provider_variable(self):
        with mock.patch.object(vector_store, "API_KEY", ""), \
                mock.patch.object(vector_store, "API_KEY_NAME", "GEMINI_API_KEY"):
            with self.assertRaisesRegex(RAGError, "GEMINI_API_KEY is missing"):
                vector_store.embed_texts(["question"])

    def test_embedding_id_includes_dimensions(self):
        with mock.patch.object(vector_store, "EMBEDDING_MODEL", "m"), \
                mock.patch.object(vector_store, "EMBEDDING_DIMENSIONS", 1536):
            self.assertEqual(vector_store.embedding_id(), "m@1536")
        with mock.patch.object(vector_store, "EMBEDDING_MODEL", "m"), \
                mock.patch.object(vector_store, "EMBEDDING_DIMENSIONS", None):
            self.assertEqual(vector_store.embedding_id(), "m")

class OpenAIErrorTests(unittest.TestCase):
    def raise_inside(self, error):
        with self.assertRaises(RAGError) as caught:
            with openai_errors("some-model"):
                raise error
        return caught.exception

    def test_authentication_error_never_echoes_the_key(self):
        error = api_error(openai.AuthenticationError, 401, "Incorrect API key provided: sk-proj-****wxyz.")
        exc = self.raise_inside(error)
        self.assertIn("401", str(exc))
        self.assertNotIn("wxyz", str(exc))
        self.assertIsNone(exc.__cause__)
        self.assertTrue(exc.__suppress_context__)

    def test_bad_request_message_redacts_the_key(self):
        with mock.patch.object(vector_store, "API_KEY", "AQ.secret-test-key"):
            exc = self.raise_inside(api_error(openai.BadRequestError, 400, "Invalid Auth key: AQ.secret-test-key"))
        self.assertNotIn("secret-test-key", str(exc))
        self.assertIn("<redacted>", str(exc))

    def test_other_errors_get_actionable_messages(self):
        cases = [
            (api_error(openai.RateLimitError, 429, "quota"), "429"),
            (api_error(openai.NotFoundError, 404, "no model"), "some-model"),
            (api_error(openai.BadRequestError, 400, "temperature unsupported"), "temperature unsupported"),
            (openai.APIConnectionError(request=REQUEST), "network"),
            (api_error(openai.InternalServerError, 500, "boom"), "500"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertIn(expected, str(self.raise_inside(error)))

if __name__ == "__main__":
    unittest.main()
