"""Tool definitions for LiteLLM function calling."""

from __future__ import annotations

import json

from macrs.catalog import ItemCatalog

# OpenAI-compatible tool schema
SEARCH_MOVIES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_movies",
        "description": (
            "Search the movie catalog. Returns up to 20 matching movies. "
            "Call this to find movies that match the user's preferences. "
            "You can combine filters — all are optional."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search by title (substring match). E.g. 'spider', 'toy'.",
                },
                "genre": {
                    "type": "string",
                    "description": "Filter by genre. E.g. 'Animation', 'Sci-Fi', 'Drama'.",
                },
                "year_min": {
                    "type": "integer",
                    "description": "Minimum release year (inclusive).",
                },
                "year_max": {
                    "type": "integer",
                    "description": "Maximum release year (inclusive).",
                },
            },
            "required": [],
        },
    },
}


def execute_search(catalog: ItemCatalog, arguments: dict) -> str:
    """Run search_movies and return results as text."""
    results = catalog.search(
        keyword=arguments.get("keyword", ""),
        genre=arguments.get("genre", ""),
        year_min=arguments.get("year_min"),
        year_max=arguments.get("year_max"),
    )
    if not results:
        return "No movies found matching those filters. Try broadening your search."

    lines = []
    for item in results:
        attrs = ", ".join(f"{k}: {v}" for k, v in item.attributes.items())
        desc = f" — {item.description}" if item.description else ""
        lines.append(f"- {item.name} ({attrs}){desc}")
    return f"Found {len(results)} movies:\n" + "\n".join(lines)
