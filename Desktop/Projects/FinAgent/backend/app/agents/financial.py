from app.graph.state import AgentState
from app.retrieval.hybrid_rag import hybrid_retriever

async def analysis_node(state: AgentState):
    query = state["query"]
    evidence = await hybrid_retriever.retrieve(f"financial fundamentals for {query}")
    return {
        "evidence": [{"agent": "financial", "data": evidence}],
        "current_node": "financial_analysis"
    }

async def news_node(state: AgentState):
    query = state["query"]
    evidence = await hybrid_retriever.retrieve(f"latest news for {query}")
    return {
        "evidence": [{"agent": "news", "data": evidence}],
        "current_node": "news_analysis"
    }

async def tech_node(state: AgentState):
    query = state["query"]
    evidence = await hybrid_retriever.retrieve(f"technical indicators for {query}")
    return {
        "evidence": [{"agent": "technical", "data": evidence}],
        "current_node": "technical_analysis"
    }

async def risk_node(state: AgentState):
    query = state["query"]
    evidence = await hybrid_retriever.retrieve(f"risk factors for {query}")
    return {
        "evidence": [{"agent": "risk", "data": evidence}],
        "current_node": "risk_analysis"
    }
