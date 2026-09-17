from langgraph.graph import END, START, StateGraph

from jevgraph.nodes import (
    blocked,
    call_model,
    call_tools,
    clarify,
    classify_request,
    escalate,
    evaluate_answer,
    route_after_classify,
    route_after_model,
)
from jevgraph.utils import AnalystState


def build_graph():
    builder = StateGraph(AnalystState)
    builder.add_node("classify", classify_request)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("blocked", blocked)
    builder.add_node("clarify", clarify)
    builder.add_node("escalate", escalate)
    builder.add_node("evaluate", evaluate_answer)

    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "model",
            "search": "model",
            "clarify": "clarify",
            "escalate": "escalate",
            "blocked": "blocked",
        },
    )
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "tools", "finish": "evaluate"},
    )
    builder.add_edge("tools", "model")
    builder.add_edge("clarify", "evaluate")
    builder.add_edge("escalate", "evaluate")
    builder.add_edge("evaluate", END)
    builder.add_edge("blocked", END)
    return builder.compile()


agent = build_graph().with_config({"recursion_limit": 20})

__all__ = ["AnalystState", "agent", "build_graph"]
