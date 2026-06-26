"""
BM25-style RAG Chunk Ranker

A lightweight, dependency-free BM25 implementation for ranking text chunks
by relevance to a query. Uses only collections.Counter and math.log.
"""

from collections import Counter
from math import log
from typing import List, Tuple


# Default stopwords - common words that carry little semantic meaning
STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with'
})


def _tokenize(text: str, stopwords: frozenset = STOPWORDS) -> List[str]:
    """
    Simple whitespace tokenization, lowercased, punctuation stripped,
    filtered by stopwords and non-alphabetic tokens.
    """
    tokens = text.lower().split()
    cleaned = []
    for t in tokens:
        # Strip leading/trailing punctuation
        clean = t.strip('.,!?;:"\'-()[]{}')
        if clean and clean.isalpha() and clean not in stopwords:
            cleaned.append(clean)
    return cleaned


def _compute_term_frequencies(chunks: List[str], stopwords: frozenset = STOPWORDS) -> List[Counter]:
    """
    Compute term frequencies for each chunk.

    Returns list of Counter objects mapping terms to their frequency in each chunk.
    """
    return [_tokenize(chunk, stopwords) and Counter(_tokenize(chunk, stopwords)) for chunk in chunks]


def bm25_score(
    query_terms: List[str],
    doc_term_freq: Counter,
    avg_doc_len: float,
    doc_len: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Compute BM25 score for a single document given query terms.

    Args:
        query_terms: List of query term strings
        doc_term_freq: Counter of term frequencies in the document
        avg_doc_len: Average document length across corpus
        doc_len: Length of this document
        k1: Term frequency saturation parameter (default: 1.5)
        b: Document length normalization parameter (default: 0.75)

    Returns:
        BM25 relevance score (float)
    """
    score = 0.0
    if avg_doc_len == 0:
        return 0.0

    len_norm = k1 * (1 - b + b * (doc_len / avg_doc_len))

    for term in query_terms:
        tf = doc_term_freq.get(term, 0)
        if tf == 0:
            continue
        # BM25 term frequency saturation
        tf_component = (tf * (k1 + 1)) / (tf + len_norm)
        # IDF component (simplified: log((N - n + 0.5) / (n + 0.5)))
        # Using 1 for N and 1 for n gives log((1-1+0.5)/(1+0.5)) = log(0.5/1.5) ≈ -1.1
        # For documents where term appears, IDF is positive
        idf = log(1.5)  # Simplified constant IDF for all terms
        score += tf_component * idf

    return score


def rank_chunks(
    query: str,
    chunks: List[str],
    k: int = 5,
    stopwords: frozenset = STOPWORDS,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float, str]]:
    """
    Rank text chunks by BM25 relevance to a query string.

    Args:
        query: Query string
        chunks: List of text chunks to rank
        k: Number of top results to return (default: 5)
        stopwords: Set of stopwords to filter (default: STOPWORDS)
        k1: BM25 k1 parameter (default: 1.5)
        b: BM25 b parameter (default: 0.75)

    Returns:
        List of (chunk_index, score, chunk_text) tuples, sorted by score descending.
        Returns empty list for stopword-only or empty queries.
    """
    if not chunks or not query:
        return []

    # Tokenize query and filter stopwords
    query_terms = _tokenize(query, stopwords)
    if not query_terms:
        return []

    # Compute term frequencies for each chunk
    chunk_term_freqs = [_tokenize(chunk, stopwords) for chunk in chunks]
    chunk_counters = [Counter(terms) for terms in chunk_term_freqs]
    doc_lens = [len(terms) for terms in chunk_term_freqs]

    # Compute average document length
    avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0

    # Score each chunk
    scores = []
    for idx, (counter, doc_len) in enumerate(zip(chunk_counters, doc_lens)):
        score = bm25_score(query_terms, counter, avg_doc_len, doc_len, k1, b)
        scores.append((idx, score, chunks[idx]))

    # Sort by score descending, then by index ascending for deterministic tie-breaking
    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores[:k]


# Convenience function for simpler usage
def score_chunks(query: str, chunks: List[str]) -> List[float]:
    """
    Score chunks by BM25 relevance without returning ranked results.

    Args:
        query: Query string
        chunks: List of text chunks

    Returns:
        List of scores in same order as input chunks
    """
    ranked = rank_chunks(query, chunks, k=len(chunks))
    scores = [0.0] * len(chunks)
    for idx, score, _ in ranked:
        scores[idx] = score
    return scores
