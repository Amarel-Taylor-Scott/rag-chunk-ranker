# BM25 RAG Chunk Ranker

A lightweight, dependency-free BM25-style relevance ranker for Retrieval-Augmented Generation (RAG) systems. Ranks text chunks by relevance to a query using the BM25 algorithm.

## Features

- **Pure Python**: Only uses `collections.Counter` and `math.log` from the standard library
- **No dependencies**: Zero external packages required
- **BM25 algorithm**: Well-established relevance scoring from information retrieval
- **Stopword filtering**: Common words are filtered by default
- **Deterministic**: Tie-breaking by document index ensures reproducible results

## Installation

```bash
# Clone or download the project
git clone https://github.com/yourusername/bm25-rag-ranker.git
cd bm25-rag-ranker

# No installation needed - just import the module
python3 -c "from bm25_ranker import rank_chunks; print(rank_chunks('query', ['chunk1', 'chunk2']))"
```

## Quick Start

```python
from bm25_ranker import rank_chunks

# Define your document chunks
chunks = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing deals with text data.",
    "Computer vision enables machines to interpret images.",
]

# Query and rank
query = "deep learning neural networks"
results = rank_chunks(query, chunks, k=3)

# Results are (index, score, chunk_text) tuples
for idx, score, text in results:
    print(f"[{score:.4f}] {text}")
```

## API Reference

### `rank_chunks(query, chunks, k=5, stopwords=None, k1=1.5, b=0.75)`

Rank text chunks by BM25 relevance to a query.

**Parameters:**
- `query` (str): Query string
- `chunks` (List[str]): List of text chunks to rank
- `k` (int): Number of top results to return (default: 5)
- `stopwords` (frozenset): Set of stopwords to filter (default: built-in STOPWORDS)
- `k1` (float): BM25 term frequency saturation (default: 1.5)
- `b` (float): BM25 document length normalization (default: 0.75)

**Returns:**
- List of `(chunk_index, score, chunk_text)` tuples, sorted by score descending

### `score_chunks(query, chunks)`

Score chunks without ranking (returns scores in original order).

### `bm25_score(query_terms, doc_term_freq, avg_doc_len, doc_len, k1=1.5, b=0.75)`

Compute BM25 score for a single document.

## Algorithm

BM25 (Best Matching 25) is a probabilistic ranking function used in information retrieval:

```
score(D, Q) = Σ IDF(qi) × (tf(qi, D) × (k1 + 1)) / (tf(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

Where:
- `tf` = term frequency in document
- `IDF` = inverse document frequency  
- `k1` = term frequency saturation parameter
- `b` = document length normalization parameter
- `|D|` = document length
- `avgdl` = average document length

## Testing

```bash
# Run tests with pytest
python3 -m pytest -q

# Run tests with unittest
python3 -m unittest discover

# Syntax check all files
python3 -m py_compile bm25_ranker.py test_bm25_ranker.py
```

## Examples

### Exact Term Match Ranking

```python
from bm25_ranker import rank_chunks

chunks = [
    "python programming tutorial",
    "python python python guide",
    "java programming basics",
]

results = rank_chunks("python", chunks)
# Chunk with "python python python" ranks highest due to higher term frequency
```

### Stopword-Only Query

```python
from bm25_ranker import rank_chunks

results = rank_chunks("the and a", ["some text"])
# Returns empty list - no meaningful terms in query
```

### Custom Stopwords

```python
from bm25_ranker import rank_chunks

custom_stops = frozenset({"python", "java"})  # Treat these as stopwords
chunks = ["python code", "java code", "go code"]

results = rank_chunks("python java", chunks, stopwords=custom_stops)
# Both terms are stopwords, so returns empty
```

## License

MIT License - see LICENSE file for details.
