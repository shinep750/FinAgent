from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.agents.planner import planner_node
from app.agents.financial import analysis_node, news_node, tech_node, risk_node
from app.agents.verifier import verify_node, synthesize_node

workflow = StateGraph(AgentState)

# Define Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("financial_analysis", analysis_node)
workflow.add_node("news_analysis", news_node)
workflow.add_node("technical_analysis", tech_node)
workflow.add_node("risk_analysis", risk_node)
workflow.add_node("verifier", verify_node)
workflow.add_node("synthesizer", synthesize_node)

# Define Edges
workflow.set_entry_point("planner")

def route_planner(state: AgentState):
    return state["plan"]

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "fundamental": "financial_analysis",
        "news": "news_analysis",
        "technical": "technical_analysis",
        "risk": "risk_analysis"
    }
)

# Fan-in to verifier
workflow.add_edge("financial_analysis", "verifier")
workflow.add_edge("news_analysis", "verifier")
workflow.add_edge("technical_analysis", "verifier")
workflow.add_edge("risk_analysis", "verifier")

workflow.add_edge("verifier", "synthesizer")
workflow.add_edge("synthesizer", END)

app_graph = workflow.compile()
