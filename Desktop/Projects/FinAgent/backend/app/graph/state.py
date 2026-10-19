from typing import Annotated, List, TypedDict, Dict
import operator

class AgentState(TypedDict):
    query: str
    plan: List[str]
    evidence: Annotated[List[Dict], operator.add]
    verification: Dict
    final_report: str
    current_node: str
