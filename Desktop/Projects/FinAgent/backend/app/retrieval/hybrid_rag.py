from typing import List, Dict, Any
from app.retrieval.vector_store import vector_db
from app.retrieval.graph_store import graph_db
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self):
        self.vector_store = vector_db
        self.graph_store = graph_db
        # In a real app, BM25 index would be loaded from disk/cache
        self.corpus = ["Apple revenue growth", "TSMC chip production", "Market volatility indices"]
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Dense Search (Semantic)
        # Note: In reality, query needs embedding first
        vector_results = await self.vector_store.search([0.1]*1536, top_k=top_k)
        
        # 2. Sparse Search (Keyword)
        tokenized_query = query.split()
        bm25_results = []
        for doc, score in zip(self.corpus, self.bm25.get_scores(tokenized_query)):
            bm25_results.append({"text": doc, "score": score, "type": "keyword"})
            
        # 3. Structural Search (Knowledge Graph)
        kg_results = await self.graph_store.get_entity_relationships(query)
        
        # Blend and Rank (Simplified RRF)
        all_evidence = []
        for r in vector_results: all_evidence.append({**r, "type": "semantic"})
        for r in bm25_results: all_evidence.append(r)
        for r in kg_results: all_evidence.append({**r, "type": "graph"})
        
        return all_evidence[:top_k]

hybrid_retriever = HybridRetriever()
