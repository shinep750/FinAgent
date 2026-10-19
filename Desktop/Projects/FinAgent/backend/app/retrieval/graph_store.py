from neo4j import GraphDatabase
from app.core.config import settings
from typing import List, Dict, Any

class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )

    async def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        # Cypher query to find related entities and their relationships
        query = "MATCH (e:Entity {name: $name})-[r]->(related) RETURN e.name, type(r) as rel, related.name as target"
        
        # Mock return for structure
        return [
            {"source": "Apple", "relationship": "SUPPLIED_BY", "target": "TSMC"},
            {"source": "Apple", "relationship": "COMPETES_WITH", "target": "Samsung"}
        ]

graph_db = GraphStore()
