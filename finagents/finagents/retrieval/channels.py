"""Retrieval channels: lexical (BM25), semantic (vectors), and metadata.

Each channel scores documents independently and returns a *rank* list rather
than a raw score. Fusion happens in :mod:`finagents.retrieval.hybrid` via
reciprocal rank fusion, so the channels need not be on a comparable scale --
which matters because BM25 scores and cosine similarities are not.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..schemas import Document

_WORD = re.compile(r"[a-z0-9][a-z0-9'\.\-]*")

# Domain terms that carry meaning in finance but would be split or ignored by a
# naive tokenizer. Kept explicit rather than clever.
_PHRASES = (
    "free cash flow",
    "gross margin",
    "operating margin",
    "price target",
    "data center",
    "risk factor",
    "supply chain",
    "export control",
    "year over year",
    "moving average",
    "customer concentration",
    "share repurchase",
    "greater china",
)

STOPWORDS = frozenset("""
    a an and are as at be been but by for from has have how i if in into is it its of on or
    that the their then there these they this to was were what when where which who whom why will
    with would should could about above after again against all also am any because before being
    below between both can did do does doing down during each few further here him his how its
    just more most no nor not now off once only other our out over own same so some such than
    too under until up very we you your
    """.split())


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    lowered = text.lower()
    tokens: List[str] = []
    for phrase in _PHRASES:
        if phrase in lowered:
            tokens.append(phrase.replace(" ", "_"))
    tokens.extend(_WORD.findall(lowered))
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------
# Lexical channel
# --------------------------------------------------------------------------


@dataclass
class BM25Index:
    """Okapi BM25 with a title boost."""

    k1: float = 1.5
    b: float = 0.75
    title_boost: float = 2.5
    _postings: Dict[str, List[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _doc_len: List[int] = field(default_factory=list)
    _doc_ids: List[str] = field(default_factory=list)
    _avg_len: float = 0.0
    _idf: Dict[str, float] = field(default_factory=dict)

    def build(self, documents: Sequence[Document]) -> "BM25Index":
        self._postings = defaultdict(list)
        self._doc_len = []
        self._doc_ids = []
        for idx, doc in enumerate(documents):
            body = tokenize(doc.text)
            title = tokenize(doc.title)
            counts: Counter[str] = Counter()
            counts.update(body)
            # Title terms counted with weight, not duplicated textually.
            for term in title:
                counts[term] += int(self.title_boost)
            length = sum(counts.values()) or 1
            self._doc_len.append(length)
            self._doc_ids.append(doc.doc_id)
            for term, freq in counts.items():
                self._postings[term].append((idx, freq))
        self._avg_len = (
            (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        )

        n_docs = len(documents)
        self._idf = {}
        for term, posting in self._postings.items():
            df = len(posting)
            self._idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        return self

    def score_all(self, query: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if not self._doc_len or not self._avg_len:
            return scores
        query_terms = tokenize(query)
        if not query_terms:
            return scores
        for term in set(query_terms):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf[term]
            for idx, freq in posting:
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (self._doc_len[idx] / self._avg_len)
                )
                scores[self._doc_ids[idx]] = scores.get(
                    self._doc_ids[idx], 0.0
                ) + idf * (freq * (self.k1 + 1.0) / denom)
        return scores


# --------------------------------------------------------------------------
# Semantic channel
# --------------------------------------------------------------------------


class Embedder:
    """Vector interface. Cosine similarity over L2-normalised vectors."""

    name = "base"
    dim = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def similarity(self, query: str, documents: Sequence[Document]) -> Dict[str, float]:
        if not documents:
            return {}
        matrix = self.embed([query, *[(d.title + ". " + d.text) for d in documents]])
        query_vec = matrix[0]
        doc_matrix = matrix[1:]
        sims = doc_matrix @ query_vec
        return {doc.doc_id: float(sim) for doc, sim in zip(documents, sims)}


class LexicalSemanticEmbedder(Embedder):
    """IDF-weighted feature-hashing vectors over unigrams and bigrams.

    Honest limitation: this is a distributional-lite representation, not a
    learned sentence embedding. It captures term overlap with sublinear term
    weighting and is genuinely useful for paraphrase-lite matching, but it will
    not match "profit" to "earnings" the way a trained encoder would. When an
    Ollama embedding model is configured, :class:`OllamaEmbedder` is used
    instead, which is why this class is a fallback rather than the default.
    """

    name = "hashing-idf"
    # Large enough that 17 long documents do not collide heavily. Collisions
    # matter because they add noise to the *ordering*, which is the only thing
    # rank fusion consumes.
    n_features = 32768

    def __init__(self, documents: Sequence[Document] = ()) -> None:
        self._idf: Dict[str, float] = {}
        if documents:
            self.fit(documents)

    def fit(self, documents: Sequence[Document]) -> "LexicalSemanticEmbedder":
        n_docs = max(1, len(documents))
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(tokenize(doc.text)))
        self._idf = {
            term: math.log(1.0 + n_docs / (1.0 + count)) for term, count in df.items()
        }
        self._default_idf = math.log(1.0 + n_docs)
        return self

    def _features(self, text: str) -> List[str]:
        tokens = tokenize(text)
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.n_features), dtype=np.float32)
        for row, text in enumerate(texts):
            features = self._features(text)
            if not features:
                continue
            weights: Counter[str] = Counter()
            for feature in features:
                weights[feature] += self._idf.get(
                    feature, getattr(self, "_default_idf", 1.0)
                )
            for feature, weight in weights.items():
                idx = hash(feature) % self.n_features
                # Signed hashing keeps collisions from systematically inflating
                # similarity.
                sign = 1.0 if (hash(feature) // self.n_features) % 2 == 0 else -1.0
                matrix[row, idx] += sign * (1.0 + math.log(weight))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @property
    def dim(self) -> int:
        return self.n_features


class OllamaEmbedder(Embedder):
    """Embeds via a local Ollama embedding model."""

    name = "ollama"

    def __init__(self, client, dim: int = 768) -> None:
        self.client = client
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.client.embed(list(texts))
        if not vectors:
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        self._dim = len(vectors[0])
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


# --------------------------------------------------------------------------
# Metadata channel
# --------------------------------------------------------------------------


def metadata_score(
    document: Document,
    *,
    as_of=None,
    target_types: Sequence = (),
    wanted_tickers: Sequence[str] = (),
    recency_weight: float = 0.6,
) -> float:
    """Blend of freshness, source reliability, type relevance and ticker match."""
    recency = document.recency(as_of)
    score = recency_weight * recency + (1.0 - recency_weight) * document.reliability
    if target_types and document.doc_type in target_types:
        score += 0.15
    if wanted_tickers:
        overlap = len(set(wanted_tickers) & set(document.metadata.get("tickers", [])))
        if overlap:
            score += min(0.2, 0.1 * overlap)
    return float(min(1.0, score))


def to_ranks(scores: Dict[str, float], descending: bool = True) -> Dict[str, int]:
    """Convert a score map to 0-based ranks (0 = best)."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=descending)
    return {doc_id: rank for rank, (doc_id, _) in enumerate(ordered)}


def rrf_fuse(
    rank_maps: Iterable[Dict[str, int]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Reciprocal rank fusion across channels.

    RRF is used instead of score averaging because the channels are not
    commensurable: a BM25 score of 8.0 and a cosine of 0.4 mean nothing to each
    other, but their rank positions do.
    """
    rank_maps = list(rank_maps)
    if not rank_maps:
        return {}
    if weights is None:
        weights = [1.0] * len(rank_maps)
    fused: Dict[str, float] = defaultdict(float)
    for weight, rank_map in zip(weights, rank_maps):
        for doc_id, rank in rank_map.items():
            fused[doc_id] += weight / (k + rank + 1.0)
    return dict(fused)


__all__ = [
    "BM25Index",
    "Embedder",
    "LexicalSemanticEmbedder",
    "OllamaEmbedder",
    "metadata_score",
    "rrf_fuse",
    "to_ranks",
    "tokenize",
]
