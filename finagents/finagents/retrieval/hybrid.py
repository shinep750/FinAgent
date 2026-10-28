"""Hybrid retriever.

Four channels, each producing a score for every candidate document:

* **lexical**   -- BM25 over unigrams, bigrams and finance phrases
* **semantic**  -- vector cosine similarity (Ollama embeddings, or IDF-hashing)
* **metadata**  -- freshness, source reliability, document-type relevance
* **graph**     -- knowledge-graph distance from the query's entities

Lexical and semantic are fused with reciprocal rank fusion (they are not on a
comparable scale). Metadata and graph are then applied as bounded additive
bonuses, which keeps them from dominating a genuinely good text match while
still letting a fresher or better-connected document outrank a stale one.

Every returned document carries its per-channel scores and a plain-English
explanation of why it was selected -- the explainability requirement starts at
retrieval, not at the report.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence


from ..schemas import Document, MetadataFilters, ScoredDocument, Settings
from .channels import (
    BM25Index,
    Embedder,
    LexicalSemanticEmbedder,
    metadata_score,
    rrf_fuse,
    to_ranks,
)
from .knowledge_graph import FinancialKnowledgeGraph

logger = logging.getLogger(__name__)

# Weights form a convex combination, so scores stay in [0, 1] without clipping.
# An earlier additive-bonus formulation saturated every candidate at 1.0, which
# silently collapsed the ranking into document order -- so these sum to 1.0 by
# construction, and the text channels keep the dominant share.
_W_TEXT = 0.68
_W_META = 0.20
_W_GRAPH = 0.12
_DIVERSITY_PENALTY = 0.05
# Applied to documents beyond the per-collection cap. Large enough to sink them
# below the cap boundary, small enough that they remain retrievable.
_OVERFLOW_FACTOR = 0.4


class HybridRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        knowledge_graph: Optional[FinancialKnowledgeGraph] = None,
        embedder: Optional[Embedder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.documents: List[Document] = list(documents)
        self.by_id: Dict[str, Document] = {d.doc_id: d for d in self.documents}
        self.bm25 = BM25Index().build(self.documents)
        self.embedder = embedder or LexicalSemanticEmbedder(self.documents)
        self.kg = knowledge_graph or FinancialKnowledgeGraph.from_documents(
            self.documents
        )

    # -- public API -------------------------------------------------------

    def search(
        self,
        query: str,
        filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = None,
        query_entities: Optional[Sequence[str]] = None,
    ) -> List[ScoredDocument]:
        top_k = top_k or self.settings.retrieval_top_k
        filters = filters or MetadataFilters()
        candidates = [d for d in self.documents if filters.matches(d)]
        if not candidates:
            return []

        entities = list(
            query_entities or filters.tickers or self._infer_entities(query)
        )

        lexical = self.bm25.score_all(query)
        semantic = self.embedder.similarity(query, candidates)
        graph = self.kg.score_documents(
            candidates, entities, max_hops=self.settings.kg_max_hops
        )
        meta = {
            d.doc_id: metadata_score(
                d,
                as_of=filters.date_to,
                target_types=filters.doc_types,
                wanted_tickers=filters.tickers,
            )
            for d in candidates
        }

        # Restrict fusion inputs to the candidate set so a metadata filter
        # genuinely excludes a document rather than merely demoting it.
        candidate_ids = {d.doc_id for d in candidates}
        lexical = {k: v for k, v in lexical.items() if k in candidate_ids}
        fused = rrf_fuse(
            [to_ranks(lexical), to_ranks(semantic)],
            k=self.settings.fusion_rrf_k,
        )

        max_fused = max(fused.values()) if fused else 1.0
        scored: List[ScoredDocument] = []
        for doc in candidates:
            base = fused.get(doc.doc_id, 0.0) / max_fused if max_fused else 0.0
            meta_component = meta.get(doc.doc_id, 0.0)
            graph_component = graph.get(doc.doc_id, 0.0)
            score = (
                _W_TEXT * base + _W_META * meta_component + _W_GRAPH * graph_component
            )
            scored.append(
                ScoredDocument(
                    document=doc,
                    score=round(min(1.0, score), 6),
                    keyword_score=round(lexical.get(doc.doc_id, 0.0), 6),
                    semantic_score=round(max(0.0, semantic.get(doc.doc_id, 0.0)), 6),
                    graph_score=round(graph_component, 6),
                    metadata_score=round(meta_component, 6),
                    rank_signals={
                        "rrf_fused": round(base, 6),
                        "recency": round(doc.recency(filters.date_to), 6),
                        "reliability": round(doc.reliability, 6),
                    },
                )
            )

        scored = self._apply_diversity(scored, query)
        scored.sort(key=lambda s: s.score, reverse=True)
        results = scored[:top_k]
        kg_context = self.kg.context_for(entities, max_hops=self.settings.kg_max_hops)
        for item in results:
            item.kg_context = kg_context
            item.explanation = self._explain(item, query, filters)
        return results

    def search_for_task(
        self, task, top_k: Optional[int] = None
    ) -> List[ScoredDocument]:
        """Run every retrieval query attached to a task and merge the results.

        Multi-query retrieval is what lets one task pull filings for the numbers
        and news for the narrative without the planner having to know that.
        """
        queries = list(task.retrieval_queries) or [task.description]
        merged: Dict[str, ScoredDocument] = {}
        for query in queries:
            hits = self.search(
                query,
                filters=task.metadata_filters,
                top_k=top_k or self.settings.retrieval_per_task_k,
            )
            for hit in hits:
                existing = merged.get(hit.document.doc_id)
                if existing is None:
                    merged[hit.document.doc_id] = hit
                else:
                    # Reciprocal combination: a document found by two of a
                    # task's queries is more likely to be on-topic than one
                    # found by a single query with a high score.
                    existing.score = round(
                        min(1.0, existing.score + 0.5 * hit.score), 6
                    )
                    existing.rank_signals["multi_query_hits"] = (
                        existing.rank_signals.get("multi_query_hits", 1.0) + 1.0
                    )
        results = sorted(merged.values(), key=lambda s: s.score, reverse=True)
        limit = top_k or self.settings.retrieval_per_task_k
        return results[:limit]

    # -- internals --------------------------------------------------------

    def _apply_diversity(
        self, scored: List[ScoredDocument], query: str
    ) -> List[ScoredDocument]:
        """Discourage a single filing, or a single publisher, from filling the list.

        Two different problems, so two mechanisms. The soft penalty is per
        *source* string: a wire service that republishes the same story under
        several headlines is redundant even though each document is distinct.
        The hard cap is per *collection*: online, one 10-Q is 20-odd chunks that
        all report `source="SEC EDGAR"`, so a per-source penalty alone leaves the
        filing free to occupy every slot and crowd out the news that would
        corroborate or contradict it. Capping per collection keeps the evidence
        base genuinely mixed.
        """
        cap = self.settings.max_docs_per_collection
        by_source: Dict[str, int] = {}
        for item in sorted(scored, key=lambda s: s.score, reverse=True):
            source = item.document.source
            by_source[source] = by_source.get(source, 0) + 1
            if by_source[source] > 2:
                item.score = round(
                    max(0.0, item.score - _DIVERSITY_PENALTY * (by_source[source] - 2)),
                    6,
                )
        return self._cap_per_collection(scored, cap)

    @staticmethod
    def _cap_per_collection(
        scored: List[ScoredDocument], cap: int
    ) -> List[ScoredDocument]:
        if cap <= 0:
            return scored
        seen: Dict[str, int] = {}
        for item in sorted(scored, key=lambda s: s.score, reverse=True):
            collection = str(
                item.document.metadata.get("collection") or item.document.source
            )
            seen[collection] = seen.get(collection, 0) + 1
            if seen[collection] > cap:
                # Demote, never delete. The caller re-sorts by score, so merely
                # returning the surplus last would put it straight back at the
                # top; lowering the score is what actually enforces the cap. It
                # still stays retrievable, which matters when a narrow query only
                # matches one filing and dropping the best evidence would be worse
                # than the duplication the cap exists to prevent.
                item.score = round(item.score * _OVERFLOW_FACTOR, 6)
        return scored

    def _infer_entities(self, query: str) -> List[str]:
        """Find graph entities named in the query.

        Ticker-style entities are returned first so they seed the traversal;
        multi-word concept entities follow. Ordering matters because context
        statements are ranked by the earliest seed they touch.
        """
        upper = query.upper()
        tickers = [
            t
            for t in self.kg.entities
            if t.isupper() and 1 < len(t) <= 5 and t in upper
        ]
        lowered = query.lower()
        concepts = [
            name
            for name in self.kg.entities
            if name not in tickers and len(name) > 5 and name.lower() in lowered
        ]
        return tickers + concepts

    def _explain(
        self, item: ScoredDocument, query: str, filters: MetadataFilters
    ) -> str:
        doc = item.document
        reasons: List[str] = []
        if item.keyword_score > 0:
            reasons.append(f"strong keyword match ({item.keyword_score:.2f})")
        if item.semantic_score > 0.3:
            reasons.append(f"high semantic similarity ({item.semantic_score:.2f})")
        if item.graph_score > 0:
            reasons.append(f"knowledge-graph connected ({item.graph_score:.2f})")
        recency = item.rank_signals.get("recency", 0.0)
        if recency > 0.6:
            reasons.append("recent")
        if doc.reliability >= 0.9:
            reasons.append("high-reliability source")
        if not reasons:
            reasons.append("retrieved by channel fusion")
        # Signals only: the document's own citation is rendered alongside this
        # text everywhere it is shown, and repeating it reads as a stutter.
        return ", ".join(reasons) + (
            f"; fusion score {item.score:.3f} (filters: {filters.describe()})"
        )


__all__ = ["HybridRetriever"]
