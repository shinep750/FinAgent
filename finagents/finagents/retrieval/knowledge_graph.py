"""Financial knowledge graph.

Retrieval over text alone misses relations: a risk factor is linked to the
regulatory event that caused it, a metric to the filing that discloses it, a
company to the peers that move it. The graph supplies that structure, and it
does two concrete jobs in the pipeline:

* **Scoring** -- documents are ranked partly by how many graph entities they
  mention, and partly by the relation between those entities and the query.
* **Context expansion** -- adjacent entities are injected into the agent
  prompt as graph context, so an agent reasoning about export controls is also
  shown the affected metrics and the mitigation events.

Implemented directly (adjacency + BFS) rather than pulling in a graph library:
the graph is small, the traversal depth is bounded, and this keeps the
dependency surface minimal.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..schemas import Document

# Relations that should not be traversed during context expansion: they are
# bookkeeping edges and would pull in noise.
_TRAVERSAL_BLOCKLIST = frozenset({"IS_A"})


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    weight: float = 1.0


@dataclass
class Entity:
    name: str
    entity_type: str = "Concept"
    mentions: int = 0
    doc_ids: Set[str] = field(default_factory=set)

    def describe(self) -> str:
        return f"{self.name} ({self.entity_type}, mentioned in {self.mentions} document(s))"


def _metric_names(metadata: Dict[str, object]) -> List[str]:
    """Metric identifiers a document carries, from XBRL tags or computed indicators."""
    names: List[str] = []
    for key in ("indicators", "metrics", "xbrl_tags"):
        value = metadata.get(key)
        if isinstance(value, dict):
            names.extend(str(k) for k in value)
        elif isinstance(value, (list, tuple)):
            names.extend(str(v) for v in value)
    return [n for n in dict.fromkeys(names) if n and n not in ("sessions", "as_of")]


class FinancialKnowledgeGraph:
    def __init__(self) -> None:
        self._triples: List[Triple] = []
        self._out: Dict[str, List[Triple]] = defaultdict(list)
        self._in: Dict[str, List[Triple]] = defaultdict(list)
        self._entities: Dict[str, Entity] = {}
        # How many triples share a (relation, object) shape. A high count means
        # the edge is boilerplate ("X REPORTS_METRIC Revenue") rather than
        # something specific to this query.
        self._edge_shape_frequency: Dict[Tuple[str, str], int] = defaultdict(int)

    # -- construction -----------------------------------------------------

    def add_triple(
        self, subject: str, relation: str, obj: str, weight: float = 1.0
    ) -> None:
        triple = Triple(subject, relation, obj, weight)
        self._triples.append(triple)
        self._out[subject].append(triple)
        self._in[obj].append(triple)
        self._edge_shape_frequency[(relation, obj)] += 1
        self._touch(subject)
        self._touch(obj)

    def _touch(self, name: str) -> Entity:
        if name not in self._entities:
            self._entities[name] = Entity(name=name)
        return self._entities[name]

    def add_document_links(self, doc_id: str, entity_names: Iterable[str]) -> None:
        for name in entity_names:
            entity = self._touch(name)
            entity.doc_ids.add(doc_id)
            entity.mentions += 1

    @classmethod
    def from_seed(
        cls,
        triples: Sequence[Tuple[str, str, str, float]],
        doc_links: Dict[str, List[str]],
    ) -> "FinancialKnowledgeGraph":
        graph = cls()
        for subject, relation, obj, weight in triples:
            graph.add_triple(subject, relation, obj, weight)
        for doc_id, names in doc_links.items():
            graph.add_document_links(doc_id, names)
        return graph

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        triples: Sequence[Tuple[str, str, str, float]] = (),
        doc_links: Optional[Dict[str, List[str]]] = None,
    ) -> "FinancialKnowledgeGraph":
        """Build a graph from the documents that were actually retrieved.

        Online there is no curated triple list, so the edges have to come from
        the corpus itself. Two kinds are derived:

        * ``(TICKER, DOCUMENTED_IN, collection)`` - clusters the 20-odd chunks of
          one filing under a single node, so a risk question reaches the whole
          filing rather than whichever chunk happened to rank first.
        * ``(TICKER, HAS_METRIC, name)`` for every indicator or XBRL tag, which
          lets a query that names a metric reach the issuer even when the metric
          word appears in no title.

        Weights come from document reliability, so a figure pulled from the XBRL
        API outranks the same claim scraped from a press release. Every edge
        still names the document it came from, which is the point: an edge with
        no citable source would be exactly the kind of invented fact this
        project refuses to emit.
        """
        graph = cls.from_seed(triples, doc_links or {})
        for doc in documents:
            names: List[str] = [str(t) for t in doc.metadata.get("tickers", [])]
            concepts: List[str] = []
            for key in ("entities", "risks", "metrics", "sector", "outlet"):
                value = doc.metadata.get(key)
                if isinstance(value, (list, tuple)):
                    concepts.extend(str(v) for v in value)
            company = doc.metadata.get("company")
            weight = max(0.05, min(1.0, float(doc.reliability)))

            graph.add_document_links(doc.doc_id, names + concepts)
            if company:
                graph.add_document_links(doc.doc_id, [str(company)])

            # Chunk identity lives on `collection`; a filing split into sections
            # must not look like 20 unrelated documents.
            collection = str(doc.metadata.get("collection") or doc.doc_id)
            for ticker in names:
                graph.add_triple(ticker, "DOCUMENTED_IN", collection, weight)
            if company and names:
                graph.add_triple(names[0], "COMPANY", str(company), weight)
            for metric in _metric_names(doc.metadata):
                for ticker in names:
                    graph.add_triple(ticker, "HAS_METRIC", metric, weight)
        return graph

    # -- queries ----------------------------------------------------------

    @property
    def entities(self) -> Dict[str, Entity]:
        return self._entities

    @property
    def triples(self) -> List[Triple]:
        return self._triples

    def knows(self, name: str) -> bool:
        return name in self._entities

    def degrees(self) -> Dict[str, float]:
        """Normalised degree, used as a centrality prior in scoring."""
        if not self._triples:
            return {}
        raw = {
            name: len(self._out.get(name, [])) + len(self._in.get(name, []))
            for name in self._entities
        }
        top = max(raw.values()) or 1
        return {name: degree / top for name, degree in raw.items()}

    def neighbours(self, name: str) -> List[Triple]:
        return [*self._out.get(name, []), *self._in.get(name, [])]

    def expand(self, seeds: Sequence[str], max_hops: int = 2) -> Dict[str, float]:
        """Breadth-first expansion from seed entities.

        Returns a relevance score per reachable entity, decaying with distance.
        Traversal is bidirectional (an entity both affects and is affected by a
        risk is relevant) and relation weights are honoured.
        """
        relevance: Dict[str, float] = {}
        queue: deque[Tuple[str, int, float]] = deque()
        for seed in seeds:
            if seed in self._entities:
                queue.append((seed, 0, 1.0))
                relevance[seed] = max(relevance.get(seed, 0.0), 1.0)
        decay = 0.55
        while queue:
            name, hops, score = queue.popleft()
            if hops >= max_hops:
                continue
            for triple in self.neighbours(name):
                if triple.relation in _TRAVERSAL_BLOCKLIST:
                    continue
                other = triple.object if triple.subject == name else triple.subject
                if other not in self._entities:
                    continue
                candidate = score * decay * triple.weight
                if candidate <= relevance.get(other, 0.0):
                    continue
                relevance[other] = candidate
                queue.append((other, hops + 1, candidate))
        return relevance

    def context_for(
        self, seeds: Sequence[str], max_hops: int = 2, max_statements: int = 12
    ) -> List[str]:
        """Human/model-readable context statements around the seed entities.

        Statements are ordered by relevance rather than by insertion: an edge
        that touches a seed outranks one that only touches a high-degree hub
        concept. Without this, a query mentioning "revenue" leads with every
        issuer's REPORTS_METRIC edge, which is true but useless.
        """
        seed_set = list(dict.fromkeys(seeds))
        if not seed_set:
            return []
        seed_rank = {name: i for i, name in enumerate(seed_set)}

        candidates: List[Tuple[Tuple[int, float], str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for name in seed_set:
            for triple in self.neighbours(name):
                key = (triple.subject, triple.relation, triple.object)
                if key in seen:
                    continue
                seen.add(key)
                endpoints = (triple.subject, triple.object)
                touching = [seed_rank[e] for e in endpoints if e in seed_rank]
                if touching:
                    # Best (lowest) seed rank this edge touches.
                    rank = min(touching)
                elif all(e in self._entities for e in endpoints):
                    rank = len(seed_rank) + 1
                else:
                    continue
                # Generic type edges are the least informative thing to show.
                is_type_edge = triple.relation in _TRAVERSAL_BLOCKLIST
                # ...as are edges whose shape is shared by several entities
                # (e.g. every issuer's REPORTS_METRIC-->Revenue).
                is_boilerplate = (
                    self._edge_shape_frequency.get((triple.relation, triple.object), 1)
                    > 1
                )
                specificity = 1 if (is_type_edge or is_boilerplate) else 0
                statement = f"{triple.subject} --{triple.relation}--> {triple.object}"
                candidates.append(((rank, specificity, -triple.weight), statement))

        ordered = sorted(candidates, key=lambda item: item[0])
        return [statement for _, statement in ordered[:max_statements]]

    def docs_for_entities(self, names: Iterable[str]) -> Dict[str, int]:
        """Count of graph-linked documents per entity, for the graph channel."""
        counts: Dict[str, int] = {}
        for name in names:
            entity = self._entities.get(name)
            if entity:
                counts[name] = len(entity.doc_ids)
        return counts

    def score_documents(
        self,
        documents: Sequence[Document],
        query_entities: Sequence[str],
        max_hops: int = 2,
    ) -> Dict[str, float]:
        """Graph relevance of each document to the query.

        A document scores if it is linked to any entity reachable from the query
        seeds, weighted by graph distance and the entity's centrality. Documents
        with no graph link are omitted, so this channel is a *bonus* signal and
        never punishes an otherwise relevant document.
        """
        relevance = self.expand(query_entities, max_hops=max_hops)
        if not relevance:
            return {}
        degrees = self.degrees()
        scores: Dict[str, float] = {}
        for doc in documents:
            total = 0.0
            for entity in doc.metadata.get("tickers", []):
                total = max(total, relevance.get(entity, 0.0))
            for key in ("entities", "risks", "metrics"):
                for entity in doc.metadata.get(key, []) or []:
                    rel = relevance.get(str(entity), 0.0)
                    if rel:
                        total += rel * (0.5 + 0.5 * degrees.get(str(entity), 0.0))
            if total:
                scores[doc.doc_id] = float(min(1.0, total / 2.0))
        return scores

    def summary(self) -> Dict[str, int]:
        return {
            "entities": len(self._entities),
            "triples": len(self._triples),
            "linked_documents": len(
                {d for e in self._entities.values() for d in e.doc_ids}
            ),
        }


__all__ = ["Entity", "FinancialKnowledgeGraph", "Triple"]
