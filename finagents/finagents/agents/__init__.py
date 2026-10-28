"""Agent implementations and the registry."""

from __future__ import annotations

from typing import Dict, Optional, Type

from ..data.providers import MarketDataProvider
from ..llm.base import LLMClient
from ..schemas import Settings
from .base import AgentRegistry, FinancialAgent
from .financial import FinancialAnalysisAgent
from .news import NewsAnalysisAgent
from .planner import PlannerAgent
from .risk import RiskAnalysisAgent
from .technical import TechnicalAnalysisAgent

AGENT_CLASSES: Dict[str, Type[FinancialAgent]] = {
    FinancialAnalysisAgent.name: FinancialAnalysisAgent,
    NewsAnalysisAgent.name: NewsAnalysisAgent,
    TechnicalAnalysisAgent.name: TechnicalAnalysisAgent,
    RiskAnalysisAgent.name: RiskAnalysisAgent,
}


def build_registry(provider: Optional[MarketDataProvider] = None) -> AgentRegistry:
    """Registry of the specialised agents.

    Agents that need market data receive the provider here rather than reaching
    for a global, which is what keeps them unit-testable with a fake provider.
    """
    registry = AgentRegistry()
    for name, cls in AGENT_CLASSES.items():
        if cls in (TechnicalAnalysisAgent, RiskAnalysisAgent):
            registry.register(name, _bind(cls, provider))
        else:
            registry.register(name, cls)
    return registry


def _bind(cls: Type[FinancialAgent], provider: Optional[MarketDataProvider]):
    class _Bound(cls):  # type: ignore[misc, valid-type]
        def __init__(self, llm: LLMClient, settings: Optional[Settings] = None) -> None:
            super().__init__(llm, settings, provider=provider)

    _Bound.__name__ = cls.__name__
    _Bound.name = cls.name
    _Bound.perspective = cls.perspective
    return _Bound


__all__ = [
    "AGENT_CLASSES",
    "AgentRegistry",
    "FinancialAgent",
    "FinancialAnalysisAgent",
    "NewsAnalysisAgent",
    "PlannerAgent",
    "RiskAnalysisAgent",
    "TechnicalAnalysisAgent",
    "build_registry",
]
