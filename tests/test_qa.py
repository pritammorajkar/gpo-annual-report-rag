import os
import unittest
from types import SimpleNamespace
from unittest import mock
import httpx
import openai
from src import config, qa
from src.vector_store import RAGError

HITS = [
    {"chunk_id": "page-5-chunk-0", "text": "The office counted 42 widgets.", "page_number": 5, "distance": 0.1},
    {"chunk_id": "page-21-chunk-1", "text": "Widgets were counted again later.", "page_number": 21, "distance": 0.2},
]

def fake_client(answer: str):
    client = mock.MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    return client

class AnswerQuestionTests(unittest.TestCase):
    def ask(self, answer: str):
        client = fake_client(answer)
        with mock.patch.object(qa, "retrieve", return_value=HITS), \
                mock.patch.object(qa, "get_openai_client", return_value=client):
            return qa.answer_question("How many widgets?", top_k=2), client

    def test_prompt_contains_page_labelled_excerpts(self):
        _, client = self.ask("42 widgets [PDF p. 5].")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], qa.SYSTEM_PROMPT)
        self.assertIn("[PDF p. 5]\nThe office counted 42 widgets.", messages[1]["content"])
        self.assertIn("[PDF p. 21]\nWidgets were counted again later.", messages[1]["content"])

    def test_result_reports_sources_and_valid_citations(self):
        result, _ = self.ask("42 widgets [PDF p. 5] [PDF p. 21].")
        self.assertEqual(result["cited_pages"], [5, 21])
        self.assertEqual(result["unsupported_citations"], [])
        self.assertEqual([s["page_number"] for s in result["sources"]], [5, 21])

    def test_citations_to_unretrieved_pages_are_flagged(self):
        result, _ = self.ask("42 widgets [PDF p. 5] [PDF p. 99].")
        self.assertEqual(result["unsupported_citations"], [99])

    def test_chat_api_errors_are_translated(self):
        client = mock.MagicMock()
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        client.chat.completions.create.side_effect = openai.AuthenticationError(
            "Incorrect API key provided: sk-****abcd", response=httpx.Response(401, request=request), body=None
        )
        with mock.patch.object(qa, "retrieve", return_value=HITS), \
                mock.patch.object(qa, "get_openai_client", return_value=client):
            with self.assertRaises(RAGError) as caught:
                qa.answer_question("How many widgets?")
        self.assertNotIn("abcd", str(caught.exception))

class CitedPagesTests(unittest.TestCase):
    def test_parses_single_and_multi_page_citations(self):
        answer = "A [PDF p. 5]. B [PDF pp. 7, 9]. C [PDF p.12]. Not a citation: p. 40."
        self.assertEqual(qa.cited_pages(answer), [5, 7, 9, 12])

    def test_no_citations(self):
        self.assertEqual(qa.cited_pages("I couldn't verify that from the retrieved report excerpts."), [])

class ConfigTests(unittest.TestCase):
    def test_int_env_reports_invalid_values(self):
        with mock.patch.dict(os.environ, {"CHUNK_SIZE": "big"}):
            with self.assertRaisesRegex(ValueError, "CHUNK_SIZE must be a whole number"):
                config._int_env("CHUNK_SIZE", 1000)

    def test_blank_values_fall_back_to_defaults(self):
        with mock.patch.dict(os.environ, {"CHUNK_SIZE": " ", "CHAT_MODEL": ""}):
            self.assertEqual(config._int_env("CHUNK_SIZE", 1000), 1000)
            self.assertEqual(config._str_env("CHAT_MODEL", "gpt-4o-mini"), "gpt-4o-mini")

if __name__ == "__main__":
    unittest.main()
