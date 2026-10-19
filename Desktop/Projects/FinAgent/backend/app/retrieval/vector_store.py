from qdrant_client import QdrantClient
from app.core.config import settings
from typing import List, Dict, Any

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        self.collection_name = "financial_evidence"

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        # Mock implementation for structure - would use self.client.search in real
        return [
            {"id": 1, "text": "Apple Q3 revenue increased by 5%", "score": 0.92, "metadata": {"source": "SEC-10Q"}},
            {"id": 2, "text": "TSMC reports supply chain constraints", "score": 0.88, "metadata": {"source": "News-Reuters"}}
        ]

vector_db = VectorStore()
