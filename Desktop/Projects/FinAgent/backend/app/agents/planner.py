import json
from typing import List, Dict
from app.graph.state import AgentState

def planner_node(state: AgentState):
    query = state["query"]
    # In a real implementation, this would be an LLM call with a specific prompt
    # to decompose the query into tasks.
    
    # Mocking the LLM decomposition logic
    plan = []
    if "valuation" in query or "fundamental" in query:
        plan.append("fundamental")
    if "news" in query or "recent" in query:
        plan.append("news")
    if "trend" in query or "technical" in query:
        plan.append("technical")
    if "risk" in query or "volatility" in query:
        plan.append("risk")
    
    if not plan:
        plan = ["fundamental", "news"] # Default baseline
        
    return {
        "plan": plan,
        "current_node": "planner"
    }
