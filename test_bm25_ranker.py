"""
Unit tests for BM25 RAG Chunk Ranker.

Run with:
    python3 -m pytest -q
    python3 -m unittest discover
"""

import unittest
import math
from collections import Counter

from bm25_ranker import (
    bm25_score,
    rank_chunks,
    score_chunks,
    _tokenize,
    _compute_term_frequencies,
    STOPWORDS,
)


class TestTokenize(unittest.TestCase):
    """Test the tokenization function."""

    def test_lowercase_conversion(self):
        """Tokens should be lowercased."""
        tokens = _tokenize("Hello WORLD")
        self.assertEqual(tokens, ['hello', 'world'])

    def test_stopword_filtering(self):
        """Stopwords should be filtered out."""
        tokens = _tokenize("the quick brown fox")
        self.assertEqual(tokens, ['quick', 'brown', 'fox'])

    def test_non_alpha_filtering(self):
        """Non-alphabetic tokens should be filtered."""
        tokens = _tokenize("hello 123 world!")
        self.assertEqual(tokens, ['hello', 'world'])

    def test_empty_result_for_stopwords_only(self):
        """Query with only stopwords should return empty list."""
        tokens = _tokenize("the an and from")
        self.assertEqual(tokens, [])


class TestBM25Score(unittest.TestCase):
    """Test BM25 score computation."""

    def test_exact_term_match_higher_score(self):
        """Shorter document with same term frequency should score higher."""
        doc1_counter = Counter({'python': 1})  # Length 1
        doc2_counter = Counter({'python': 1})  # Length 1

        avg_len = 1.0
        doc_len1 = 1   # Shorter - should score higher
        doc_len2 = 5   # Longer - should score lower

        score1 = bm25_score(['python'], doc1_counter, avg_len, doc_len1)
        score2 = bm25_score(['python'], doc2_counter, avg_len, doc_len2)

        # Shorter document scores higher with same term frequency
        self.assertGreater(score1, score2)

    def test_term_presence_matters(self):
        """Document with query term should score higher than without."""
        doc1_counter = Counter({'python': 1, 'java': 1})
        doc2_counter = Counter({'java': 1, 'go': 1})

        avg_len = 2.0
        doc_len = 2

        score1 = bm25_score(['python'], doc1_counter, avg_len, doc_len)
        score2 = bm25_score(['python'], doc2_counter, avg_len, doc_len)

        # Document with term scores higher
        self.assertGreater(score1, score2)

    def test_higher_tf_higher_score(self):
        """Higher term frequency should yield higher score."""
        doc1_counter = Counter({'python': 1})
        doc2_counter = Counter({'python': 3})

        avg_len = 3.0
        doc_len = 3

        score1 = bm25_score(['python'], doc1_counter, avg_len, doc_len)
        score2 = bm25_score(['python'], doc2_counter, avg_len, doc_len)

        self.assertGreater(score2, score1)

    def test_zero_score_for_missing_term(self):
        """Document without query term should have zero score."""
        doc_counter = Counter({'java': 1, 'go': 1})
        score = bm25_score(['python'], doc_counter, 2.0, 2)
        self.assertEqual(score, 0.0)

    def test_score_is_positive_for_match(self):
        """Score should be positive when term is present."""
        doc_counter = Counter({'python': 1})
        score = bm25_score(['python'], doc_counter, 1.0, 1)
        self.assertGreater(score, 0.0)


class TestRankChunks(unittest.TestCase):
    """Test the main rank_chunks function."""

    def test_exact_term_match_ranks_higher(self):
        """Chunks with exact term matches should rank higher."""
        chunks = [
            "python programming language",
            "python snake in the wild",
            "java programming language",
            "programming basics",
        ]
        ranked = rank_chunks("python", chunks, k=2)

        # First result should contain 'python'
        self.assertIn('python', ranked[0][2].lower())
        # Should get exactly 2 results
        self.assertEqual(len(ranked), 2)

    def test_stopword_only_query_returns_empty(self):
        """Query containing only stopwords should return empty list."""
        chunks = [
            "python programming language",
            "java programming language",
        ]
        ranked = rank_chunks("the and a", chunks)
        self.assertEqual(ranked, [])

    def test_empty_query_returns_empty(self):
        """Empty query string should return empty list."""
        chunks = ["python programming"]
        ranked = rank_chunks("", chunks)
        self.assertEqual(ranked, [])

    def test_empty_chunks_returns_empty(self):
        """Empty chunks list should return empty list."""
        ranked = rank_chunks("python", [])
        self.assertEqual(ranked, [])

    def test_deterministic_tie_breaking(self):
        """Tie-breaking should be deterministic by index."""
        chunks = [
            "python programming",  # index 0
            "python coding",        # index 1
            "python script",        # index 2
        ]
        # All chunks have same score for 'python'
        ranked1 = rank_chunks("python", chunks, k=3)
        ranked2 = rank_chunks("python", chunks, k=3)

        # Results should be identical across calls
        self.assertEqual(ranked1, ranked2)
        # Should be ordered by index ascending (deterministic)
        self.assertEqual(ranked1[0][0], 0)
        self.assertEqual(ranked1[1][0], 1)
        self.assertEqual(ranked1[2][0], 2)

    def test_k_limits_results(self):
        """Should return at most k results."""
        chunks = [
            "python", "java", "go", "rust", "c++",
            "javascript", "typescript", "swift", "kotlin", "scala"
        ]
        ranked = rank_chunks("python", chunks, k=3)
        self.assertLessEqual(len(ranked), 3)

    def test_scores_ordered_descending(self):
        """Results should be ordered by score descending."""
        chunks = [
            "python basics",                     # 1 match
            "python programming tutorial",       # 2 matches
            "python python python guide",        # 3 matches
        ]
        ranked = rank_chunks("python", chunks, k=3)

        scores = [r[1] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_returns_correct_index(self):
        """Returned indices should match original chunk positions."""
        chunks = [
            "first chunk python",
            "second chunk",
            "third chunk python python",
        ]
        ranked = rank_chunks("python", chunks, k=3)

        indices = [r[0] for r in ranked]
        self.assertIn(0, indices)
        self.assertIn(2, indices)
        self.assertIn(1, indices)  # May appear if there are non-python matches


class TestScoreChunks(unittest.TestCase):
    """Test the score_chunks convenience function."""

    def test_same_order_as_input(self):
        """Scores should correspond to input chunk order."""
        chunks = ["apple fruit", "banana fruit", "cherry fruit"]
        scores = score_chunks("fruit", chunks)
        self.assertEqual(len(scores), len(chunks))

    def test_zero_for_no_match(self):
        """No match chunks should have zero score."""
        chunks = ["apple", "banana", "cherry"]
        scores = score_chunks("python", chunks)
        self.assertEqual(scores, [0.0, 0.0, 0.0])


class TestIntegration(unittest.TestCase):
    """Integration tests for realistic RAG scenarios."""

    def test_rag_chunk_ranking(self):
        """Test realistic document retrieval scenario."""
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text and speech.",
            "Computer vision enables machines to interpret images.",
            "Reinforcement learning trains agents through rewards.",
        ]

        # Query about deep learning should rank relevant docs higher
        ranked = rank_chunks("deep learning neural networks", docs, k=5)

        # The deep learning document should be first or near top
        self.assertIn('deep learning', ranked[0][2].lower())

    def test_multi_term_query(self):
        """Multi-term queries should score documents with all terms higher."""
        chunks = [
            "python programming tutorial",
            "python only",
            "java programming guide",
            "programming python java",
        ]

        ranked = rank_chunks("python programming", chunks, k=4)

        # Chunk with both terms should rank higher than single-term chunks
        scores = {r[0]: r[1] for r in ranked}

        # Document with both terms (index 0) should score higher than single-term docs
        self.assertGreater(scores[0], scores[1])  # python programming > python only
        self.assertGreater(scores[0], scores[2])  # python programming > java programming


if __name__ == '__main__':
    unittest.main()
