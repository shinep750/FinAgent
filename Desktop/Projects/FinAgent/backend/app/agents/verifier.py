from app.graph.state import AgentState

def verify_node(state: AgentState):
    evidence = state["evidence"]
    conflicts = []
    confidence = 0.85
    return {
        "verification": {"conflicts": conflicts, "confidence": confidence},
        "current_node": "verifier"
    }

def synthesize_node(state: AgentState):
    evidence_summary = str(state["evidence"])
    report = f"FINAL FINANCIAL REPORT\nQuery: {state[`"query`"]}\n\nEvidence:\n{evidence_summary}\n\nVerdict: High Confidence Investment."
    return {
        "final_report": report,
        "current_node": "synthesizer"
    }
