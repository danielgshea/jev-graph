import json
import os
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith import traceable
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
from typing_extensions import TypedDict

from tools import search_web

load_dotenv()
os.environ["LANGSMITH_GATEWAY"] = "true"

analyst_instructions = """You are an analytical assistant.

Use search_web when current or uncertain information is needed. Be clear about
uncertainty, ground claims in tool results, and never invent sources or facts.
"""


class AnalystState(MessagesState):
    analyst: dict[str, Any]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _request(state: AnalystState) -> str:
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            return _text(message.content)
    return ""


def _snapshot(state: AnalystState, **extra: Any) -> dict[str, Any]:
    return {
        "request": _request(state),
        "messages": [_text(message.content) for message in state["messages"]],
        "analyst": state.get("analyst", {}),
        **extra,
    }


def _merge(state: AnalystState, **values: Any) -> dict[str, Any]:
    return {"analyst": {**state.get("analyst", {}), **values}}


def _route_from_jev(state: AnalystState, route: Any) -> str:
    return "blocked" if state["analyst"]["is_safe"] < 0.5 else route.choice


@traceable(name="analyst_jev")
def _ask_jev(state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    with TypeSafeClient() as client:
        return client.system_one(state=state, questions=questions).answers


search_web_tool = tool(search_web)
tools_by_name = {search_web_tool.name: search_web_tool}


@lru_cache
def _model():
    return init_chat_model(os.getenv("WEATHER_AGENT_MODEL", "openai:gpt-5.5"), temperature=0)


def _blocked(_: AnalystState) -> dict[str, Any]:
    return {"messages": [AIMessage(content="I cannot help with that request.")]}


def _clarify(_: AnalystState) -> dict[str, Any]:
    return {"messages": [AIMessage(content="Could you clarify what you would like me to research?")]}


def _escalate(_: AnalystState) -> dict[str, Any]:
    return {"messages": [AIMessage(content="This request should be reviewed by a human or specialist.")]}


def classify_request(state: AnalystState) -> dict[str, Any]:
    answers = _ask_jev(
        _snapshot(state),
        {
            "route": Choice(
                instructions="Choose the best next action for this request.",
                criteria={
                    "answer": "The agent can answer from its knowledge.",
                    "search": "Current or missing information requires web search.",
                    "clarify": "The request is ambiguous and needs clarification.",
                    "escalate": "A human or specialist should handle the request.",
                },
            ),
            "is_safe": Noul(instructions="Is this request safe to process without violating guardrails?"),
            "should_remember": Noul(
                instructions="Does this request contain a useful durable preference or fact to remember?"
            ),
        },
    )
    return _merge(
        state,
        is_safe=answers["is_safe"].noul,
        should_remember=answers["should_remember"].noul,
        route=answers["route"].choice,
        route_confidence=answers["route"].confidence,
    )


def call_model(state: AnalystState) -> dict[str, Any]:
    response = _model().bind_tools([search_web_tool]).invoke(
        [SystemMessage(content=analyst_instructions), *state["messages"]]
    )
    return {"messages": [response]}


def call_tools(state: AnalystState) -> dict[str, Any]:
    tool_messages = []
    analyst = state.get("analyst", {})
    for call in state["messages"][-1].tool_calls:
        tool_state = _snapshot(state, tool_name=call["name"], tool_args=call.get("args", {}))
        safety = _ask_jev(
            tool_state,
            {"is_authorized": Noul(instructions="Is this tool call safe and authorized for the user's request?")},
        )["is_authorized"].noul
        if safety < 0.5:
            tool_messages.append(
                ToolMessage(
                    content="Tool call blocked by the analyst safety gate.",
                    tool_call_id=call["id"],
                    status="error",
                )
            )
            analyst = {**analyst, "tool_safe": safety}
            continue

        result = tools_by_name[call["name"]].invoke(call.get("args", {}))
        quality = _ask_jev(
            {**tool_state, "tool_result": result},
            {
                "retrieval_quality": Score(
                    instructions="How useful and relevant is this tool result for answering the request?",
                    criteria=[
                        "Poor: irrelevant, empty, or misleading",
                        "Adequate: partially useful but incomplete",
                        "Excellent: relevant, reliable, and sufficient",
                    ],
                )
            },
        )["retrieval_quality"]
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=call["id"],
            )
        )
        analyst = {
            **analyst,
            "tool_safe": safety,
            "retrieval_quality": quality.score,
            "retrieval_confidence": quality.confidence,
        }
    return {"messages": tool_messages, "analyst": analyst}


def evaluate_answer(state: AnalystState) -> dict[str, Any]:
    message = state["messages"][-1]
    answers = _ask_jev(
        _snapshot(state, final_answer=_text(message.content)),
        {
            "is_complete": Noul(instructions="Does the final answer fully complete the user's request?"),
            "escalation": Choice(
                instructions="Choose the right handoff outcome for this answer.",
                criteria={
                    "continue": "No handoff is needed.",
                    "clarify": "Ask the user for missing information.",
                    "human_review": "A human or specialist should review or take over.",
                },
            ),
            "output_quality": Score(
                instructions="How clear, accurate, and useful is the final answer?",
                criteria=[
                    "Poor: incorrect, unsupported, or unusable",
                    "Adequate: mostly useful with notable omissions",
                    "Excellent: accurate, clear, and complete",
                ],
            ),
            "memory_action": Choice(
                instructions="Choose how to handle the candidate memory from this interaction.",
                criteria={
                    "replace": "Replace conflicting existing memory with the new fact.",
                    "merge": "Merge the new fact with compatible existing memory.",
                    "reject": "Do not persist this information.",
                },
            ),
        },
    )
    return _merge(
        state,
        is_complete=answers["is_complete"].noul,
        escalation=answers["escalation"].choice,
        escalation_confidence=answers["escalation"].confidence,
        output_quality=answers["output_quality"].score,
        output_quality_confidence=answers["output_quality"].confidence,
        memory_action=answers["memory_action"].choice,
        memory_confidence=answers["memory_action"].confidence,
    )


def build_graph():
    builder = StateGraph(AnalystState)
    builder.add_node("classify", classify_request)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("blocked", _blocked)
    builder.add_node("clarify", _clarify)
    builder.add_node("escalate", _escalate)
    builder.add_node("evaluate", evaluate_answer)

    builder.add_conditional_edges(
        "classify",
        lambda state: _route_from_jev(
            state,
            _ask_jev(
                _snapshot(state),
                {
                    "route": Choice(
                        instructions="Choose the best next action for this request.",
                        criteria={
                            "answer": "The agent can answer from its knowledge.",
                            "search": "Current or missing information requires web search.",
                            "clarify": "The request is ambiguous and needs clarification.",
                            "escalate": "A human or specialist should handle the request.",
                        },
                    ),
                },
            )["route"],
        ),
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
        lambda state: (
            "finish"
            if not state["messages"][-1].tool_calls
            else "tools"
            if _ask_jev(
                _snapshot(state, selected_tools=[call["name"] for call in state["messages"][-1].tool_calls]),
                {
                    "tool_selection": Choice(
                        instructions="Classify whether the selected tools fit the user's request.",
                        criteria={
                            "appropriate": "The selected tools are the right next action.",
                            "unnecessary": "The agent should answer without using a tool.",
                            "wrong_tool": "A different tool or action is needed.",
                        },
                    )
                },
            )["tool_selection"].choice
            != "unnecessary"
            else "finish"
        ),
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
