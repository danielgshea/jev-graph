import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for current information relevant to the user's request."""
    return tavily.search(query, max_results=max_results, include_raw_content=False)
