"""Pluggable item catalog. Subclass ItemCatalog to swap domains."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    name: str
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"genre": "sci-fi", "year": "2023"}
    description: str = ""


class ItemCatalog:
    """In-memory catalog. Replace with a DB-backed version as needed."""

    def __init__(self, items: list[Item] | None = None) -> None:
        self.items = items or []

    def search(self, **filters: str) -> list[Item]:
        """Return items matching all key=value filters (substring match)."""
        results = []
        for item in self.items:
            if all(v.lower() in item.attributes.get(k, "").lower() for k, v in filters.items()):
                results.append(item)
        return results

    def as_text(self) -> str:
        """Full catalog as text for LLM context."""
        lines = []
        for item in self.items:
            attrs = ", ".join(f"{k}: {v}" for k, v in item.attributes.items())
            desc = f" — {item.description}" if item.description else ""
            lines.append(f"- {item.name} ({attrs}){desc}")
        return "\n".join(lines)
