import unittest
from src.chunking import chunk_pages
from src.document import PageText

WORDS = " ".join(f"word{i}" for i in range(300))

class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_page_metadata_and_overlap(self):
        chunks = chunk_pages([PageText(7, "abcdefghijklmnopqrstuvwxyz")], chunk_size=10, overlap=3)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_number == 7 for chunk in chunks))
        self.assertEqual(chunks[0].text[-3:], chunks[1].text[:3])

    def test_invalid_overlap_fails(self):
        with self.assertRaises(ValueError):
            chunk_pages([PageText(1, "text")], chunk_size=10, overlap=10)

    def test_invalid_chunk_size_fails(self):
        with self.assertRaises(ValueError):
            chunk_pages([PageText(1, "text")], chunk_size=0, overlap=0)

    def test_chunks_respect_size_and_overlap_on_word_boundaries(self):
        chunk_size, overlap = 100, 30
        chunks = chunk_pages([PageText(1, WORDS)], chunk_size=chunk_size, overlap=overlap)
        self.assertGreater(len(chunks), 5)
        words = set(WORDS.split())
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), chunk_size)
            # No word is cut in half at either edge.
            self.assertTrue(set(chunk.text.split()) <= words, chunk.text)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertIn(previous.text[-overlap:], current.text)
        self.assertTrue(chunks[-1].text.endswith("word299"))

    def test_all_text_is_covered(self):
        chunks = chunk_pages([PageText(1, WORDS)], chunk_size=120, overlap=20)
        covered = set(" ".join(chunk.text for chunk in chunks).split())
        self.assertEqual(covered, set(WORDS.split()))

    def test_chunks_never_span_pages_and_ids_are_unique(self):
        pages = [PageText(3, WORDS), PageText(4, "short page"), PageText(9, WORDS)]
        chunks = chunk_pages(pages, chunk_size=200, overlap=40)
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))
        self.assertEqual({chunk.page_number for chunk in chunks}, {3, 4, 9})
        short = [chunk for chunk in chunks if chunk.page_number == 4]
        self.assertEqual([(c.chunk_id, c.text) for c in short], [("page-4-chunk-0", "short page")])

    def test_whitespace_is_normalised_and_blank_pages_produce_no_chunks(self):
        chunks = chunk_pages([PageText(1, "  a\n\nb\tc  "), PageText(2, "   \n ")], chunk_size=50, overlap=5)
        self.assertEqual([(c.page_number, c.text) for c in chunks], [(1, "a b c")])

    def test_word_longer_than_chunk_falls_back_to_hard_cut(self):
        text = "x" * 250 + " tail"
        chunks = chunk_pages([PageText(1, text)], chunk_size=100, overlap=10)
        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))
        self.assertTrue(chunks[-1].text.endswith("tail"))

if __name__ == "__main__":
    unittest.main()
