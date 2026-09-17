import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import MessagesState
from langsmith import traceable
from typesafe_sdk import TypeSafeClient

from jevgraph.tools import search_web

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
