import asyncio
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.core.config import settings

# --- MOCK DATA ---
# Knowledge Graph Data: (Source, Relationship, Target)
KG_RELATIONS = [
    ("Nvidia", "SUPPLIED_BY", "TSMC"),
    ("Nvidia", "COMPETES_WITH", "AMD"),
    ("TSMC", "SUPPLIED_BY", "ASML"),
    ("Apple", "SUPPLIED_BY", "TSMC"),
    ("AMD", "COMPETES_WITH", "Nvidia"),
    ("AI_Boom", "AFFECTS", "Nvidia"),
    ("US_Export_Ban", "AFFECTS", "Nvidia"),
    ("US_Export_Ban", "AFFECTS", "TSMC"),
]

# Vector Data: (Text, Metadata)
VECTOR_DATA = [
    {
        "text": "Nvidia Q3 revenue grew 206% year-over-year, driven by H100 demand.",
        "meta": {"source": "SEC-10Q", "category": "financial"},
        "vector": [0.1] * 1536
    },
    {
        "text": "TSMC reports capacity constraints for 3nm chips, potentially delaying AI GPU shipments.",
        "meta": {"source": "Reuters", "category": "news"},
        "vector": [0.2] * 1536
    },
    {
        "text": "The US Department of Commerce is considering tighter export controls on H100 chips to China.",
        "meta": {"source": "Bloomberg", "category": "risk"},
        "vector": [0.3] * 1536
    },
    {
        "text": "AMD is launching the MI300X to compete directly with Nvidia in the data center space.",
        "meta": {"source": "Analyst Report", "category": "technical"},
        "vector": [0.4] * 1536
    }
]

async def seed_neo4j():
    print("🌱 Seeding Neo4j Knowledge Graph...")
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD))
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create relationships
        for src, rel, tgt in KG_RELATIONS:
            session.run(
                f"MERGE (s:Entity {{name: $src}}) MERGE (t:Entity {{name: $tgt}}) MERGE (s)-[:{rel}]->(t)",
                src=src, tgt=tgt
            )
    driver.close()
    print("✅ Neo4j seeded.")

async def seed_qdrant():
    print("🌱 Seeding Qdrant Vector Store...")
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    collection_name = "financial_evidence"
    
    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except Exception:
        pass # Collection already exists

    # Upload points
    points = [
        PointStruct(
            id=idx,
            vector=item["vector"],
            payload={"text": item["text"], **item["meta"]}
        )
        for idx, item in enumerate(VECTOR_DATA)
    ]
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Qdrant seeded.")

async def main():
    try:
        await seed_neo4j()
        await seed_qdrant()
        print("\n🚀 All systems seeded! You can now try the application.")
    except Exception as e:
        print(f"❌ Seeding failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
