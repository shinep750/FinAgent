"""Retrieval must be explainable and must respect its metadata filters.

An assertion like "the retriever returns the risk factors first" is only
meaningful if the scores are derived from the text, so these tests check the
signals rather than just the ordering.

The corpus is the fixture in ``conftest.py``. That keeps the suite offline while
still exercising the real retriever, and it also pins the diversity behaviour
that matters online: with live SEC data a single filing can produce 20 chunks
from one `collection`, so per-source caps alone would let one filing fill the
result list.
"""

from __future__ import annotations

import pytest

from finagents.retrieval import FinancialKnowledgeGraph, HybridRetriever
from finagents.retrieval.channels import tokenize
from finagents.schemas import MetadataFilters, UserQuery


@pytest.fixture(scope="module")
def kg(documents):
    return FinancialKnowledgeGraph.from_documents(documents)


@pytest.fixture(scope="module")
def retriever(documents, kg, settings):
    return HybridRetriever(documents, knowledge_graph=kg, settings=settings)


class TestCorpus:
    def test_documents_load(self, documents):
        assert len(documents) >= 5

    def test_doc_ids_are_unique(self, documents):
        ids = [d.doc_id for d in documents]
        assert len(ids) == len(set(ids))

    def test_reliability_in_range(self, documents):
        assert all(0.0 <= d.reliability <= 1.0 for d in documents)

    def test_every_document_names_its_source(self, documents):
        """An unsourced document cannot be audited, so it is not admissible."""
        assert all(d.source for d in documents)

    def test_no_synthetic_flag_survives_in_the_corpus(self, documents):
        """The product has no synthetic corpus; nothing may claim to be one."""
        assert not any(d.metadata.get("synthetic") for d in documents)

    def test_every_document_declares_its_tickers(self, documents):
        assert all(d.metadata.get("tickers") for d in documents)


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self):
        assert "revenue" in tokenize("Revenue, up 12%.")

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the revenue")
        assert "and" not in tokenize("revenue and margin")

    def test_keeps_numbers(self):
        assert "75.0" in tokenize("gross margin was 75.0%")


class TestHybridRetriever:
    def test_returns_scored_documents(self, retriever):
        hits = retriever.search("NVDA revenue growth", top_k=5)
        assert hits and all(0.0 <= h.score <= 1.0 for h in hits)

    def test_sorted_descending(self, retriever):
        scores = [h.score for h in retriever.search("NVDA", top_k=6)]
        assert scores == sorted(scores, reverse=True)

    def test_keyword_query_finds_filing(self, retriever):
        hits = retriever.search("gross margin fiscal year revenue", top_k=5)
        assert any("xbrl" in h.document.doc_id for h in hits)

    def test_export_controls_query_finds_risk_factors(self, retriever):
        hits = retriever.search("export controls restrict sales regions", top_k=5)
        assert any("risks" in h.document.doc_id for h in hits)

    def test_every_hit_explains_itself(self, retriever):
        for hit in retriever.search("NVDA revenue", top_k=5):
            assert hit.explanation
            assert "fusion score" in hit.explanation

    def test_explanation_does_not_repeat_the_title(self, retriever):
        """The renderer prints the citation, so a stutter is noise."""
        for hit in retriever.search("NVDA revenue", top_k=5):
            assert not hit.explanation.startswith(hit.document.title)

    def test_metadata_filter_restricts_tickers(self, retriever):
        hits = retriever.search(
            "revenue margin", filters=MetadataFilters(tickers=["AMD"]), top_k=8
        )
        assert hits
        assert all("AMD" in h.document.metadata.get("tickers", []) for h in hits)

    def test_metadata_filter_restricts_types(self, retriever):
        hits = retriever.search(
            "NVDA",
            filters=MetadataFilters(tickers=["NVDA"], doc_types=["news"]),
            top_k=8,
        )
        assert all(h.document.doc_type.value == "news" for h in hits)

    def test_scores_are_not_all_identical(self, retriever):
        """Uniform scores mean the channels are not actually discriminating."""
        scores = {round(h.score, 4) for h in retriever.search("NVDA", top_k=8)}
        assert len(scores) > 1

    def test_search_for_task_merges_queries(self, workflow, retriever):
        plan = workflow.planner.plan(
            UserQuery(raw_text="Analyse NVDA fundamentals and news")
        )
        task = next(t for t in plan.tasks if t.perspective.value == "fundamental")
        hits = retriever.search_for_task(task, top_k=6)
        assert hits
        assert len(hits) <= 6

    def test_diversity_limits_one_source(self, retriever):
        hits = retriever.search("NVDA revenue margin growth outlook", top_k=8)
        by_source: dict = {}
        for hit in hits:
            by_source[hit.document.source] = by_source.get(hit.document.source, 0) + 1
        assert max(by_source.values()) <= 3

    def test_diversity_limits_one_filing(self, retriever, documents):
        """Live filings chunk into many pieces of one document.

        Capping per *source* is not enough when one 10-Q produces 20 chunks that
        all share a source string, so the retriever also caps per collection.
        """
        same_filing = [
            d for d in documents if d.metadata.get("collection") == "sec-nvda-10q"
        ]
        assert len(same_filing) >= 2
        hits = retriever.search(
            "risk factors customer concentration export controls", top_k=8
        )
        by_collection: dict = {}
        for hit in hits:
            collection = hit.document.metadata.get("collection")
            by_collection[collection] = by_collection.get(collection, 0) + 1
        assert max(by_collection.values(), default=0) <= 3

    def test_one_verbose_filing_cannot_take_every_slot(self, settings):
        """Six chunks of one filing, plus two other documents.

        Every chunk repeats the filing's vocabulary, so lexical scoring alone
        would rank all six first. The per-collection cap must demote the surplus
        so the other documents are actually seen.
        """
        from finagents.schemas import Document, EvidenceType

        verbose = [
            Document(
                doc_id=f"sec-chunky-10q-{i}",
                title=(
                    "NVIDIA CORP 10-Q - Item 1A. Risk Factors"
                    if i % 2
                    else "NVIDIA CORP 10-Q - Item 2. Management's Discussion"
                ),
                text=(
                    "Customer concentration and export controls remain the principal risks. "
                    "Demand depends on a small number of hyperscale customers, and export "
                    "controls restrict sales of advanced products to certain regions. "
                    f"This is section {i} of the same filing and repeats the same vocabulary."
                ),
                doc_type=EvidenceType.REGULATORY,
                source="SEC EDGAR",
                reliability=0.78,
                metadata={
                    "tickers": ["NVDA"],
                    "collection": "sec-chunky-10q",
                    "source_kind": "regulatory_filing",
                },
            )
            for i in range(6)
        ]
        others = [
            Document(
                doc_id="news-chunky-1",
                title="NVIDIA CORP announces a new datacenter platform",
                text=(
                    "NVIDIA CORP unveiled a new datacenter platform and said customer "
                    "export controls and hyperscale demand shape its roadmap this year."
                ),
                doc_type=EvidenceType.NEWS,
                source="Reuters (via Google News)",
                reliability=0.80,
                metadata={"tickers": ["NVDA"], "collection": "news-chunky"},
            ),
            Document(
                doc_id="market-chunky-1",
                title="NVIDIA CORP price statistics",
                text=(
                    "NVIDIA CORP last traded at 181.20 with a 50-day moving average of "
                    "169.40, a 200-day average of 151.05 and a 14-day RSI of 58.2."
                ),
                doc_type=EvidenceType.MARKET_DATA,
                source="Yahoo Finance (via yfinance)",
                reliability=0.85,
                metadata={"tickers": ["NVDA"], "collection": "market-chunky"},
            ),
        ]
        retriever = HybridRetriever([*verbose, *others], settings=settings)
        hits = retriever.search(
            "customer concentration export controls hyperscale demand", top_k=4
        )
        top_four = hits[:4]
        filing_chunks = [
            h for h in top_four if h.document.metadata["collection"] == "sec-chunky-10q"
        ]
        assert len(filing_chunks) <= settings.max_docs_per_collection
        assert any(
            h.document.metadata["collection"] != "sec-chunky-10q" for h in top_four
        )


class TestKnowledgeGraph:
    def test_built_from_documents(self, kg):
        assert kg.entities
        assert kg.triples

    def test_every_triple_resolves(self, kg):
        for triple in kg.triples:
            assert triple.subject in kg.entities
            assert triple.object in kg.entities

    def test_every_entity_edge_is_reachable_from_a_cited_document(self, kg, documents):
        """An edge no document supports is exactly the hallucination risk.

        Edges are derived from document metadata, so the check is that every
        entity the graph reaches is linked to a document that was actually
        retrieved.
        """
        known = {d.doc_id for d in documents}
        counts = kg.docs_for_entities(list(kg.entities))
        assert counts, "the graph must cite the documents it was built from"
        for name, count in counts.items():
            linked = set(kg.entities[name].doc_ids)
            assert linked <= known, f"{name} cites a document that was not retrieved"
            assert count == len(linked)

    def test_filing_chunks_share_one_collection_node(self, kg, documents):
        """20 chunks of one filing must not look like 20 unrelated sources."""
        collections = {
            d.metadata.get("collection")
            for d in documents
            if d.metadata.get("source_kind") == "regulatory_filing"
        }
        for collection in collections:
            assert any(
                t.subject == "NVDA" and t.object == collection for t in kg.triples
            ), collection

    def test_metric_edges_come_from_document_metadata(self, kg, documents):
        market = next(d for d in documents if d.doc_id == "market-nvda-snapshot")
        for metric in market.metadata["indicators"]:
            assert kg.knows(metric), metric

    def test_context_for_ticker(self, kg):
        context = kg.context_for(["NVDA"])
        assert context
        assert any("NVDA" in line for line in context)

    def test_context_is_bounded(self, kg):
        assert len(kg.context_for(["NVDA"], max_statements=3)) <= 3

    def test_unknown_entity_yields_nothing(self, kg):
        assert kg.context_for(["ZZZZ"]) == []

    def test_ticker_expands_to_related_entities(self, kg):
        expanded = kg.expand(["NVDA"])
        assert expanded["NVDA"] == 1.0
