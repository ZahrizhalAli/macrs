"""Pluggable item catalog. Subclass ItemCatalog to swap domains."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    name: str
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"genre": "sci-fi", "year": "2023"}
    description: str = ""


class ItemCatalog:
    """In-memory catalog. Replace with a DB-backed version as needed."""

    def __init__(self, items: list[Item] | None = None) -> None:
        self.items = items or []

    def search(
        self,
        keyword: str = "",
        genre: str = "",
        year_min: int | None = None,
        year_max: int | None = None,
        limit: int = 20,
    ) -> list[Item]:
        """Search items by keyword (title substring), genre, and year range."""
        results: list[Item] = []
        kw = keyword.lower()
        genre_lower = genre.lower()

        for item in self.items:
            if kw and kw not in item.name.lower():
                continue
            if genre_lower and genre_lower not in item.attributes.get("genre", "").lower():
                continue
            year_str = item.attributes.get("year", "")
            if year_str.isdigit():
                year = int(year_str)
                if year_min and year < year_min:
                    continue
                if year_max and year > year_max:
                    continue
            elif year_min or year_max:
                continue  # skip items with no year when year filter is set
            results.append(item)
            if len(results) >= limit:
                break

        return results

    def as_text(self, limit: int | None = None) -> str:
        """Catalog as text for LLM context. Use limit to cap entries."""
        items = self.items if limit is None else self.items[:limit]
        lines = []
        for item in items:
            attrs = ", ".join(f"{k}: {v}" for k, v in item.attributes.items())
            desc = f" — {item.description}" if item.description else ""
            lines.append(f"- {item.name} ({attrs}){desc}")
        return "\n".join(lines)

    @classmethod
    def from_movielens(cls, movies_csv: str | Path) -> ItemCatalog:
        """Load from a MovieLens movies.csv file.

        Expected format: movieId,title,genres
        Title often contains year like "Toy Story (1995)".
        Genres are pipe-separated like "Adventure|Animation|Children".
        """
        items: list[Item] = []
        path = Path(movies_csv)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_title = row["title"].strip()
                # Extract year from title e.g. "Toy Story (1995)"
                year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
                year = year_match.group(1) if year_match else ""
                title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title)

                genres = row["genres"].replace("|", ", ")
                if genres == "(no genres listed)":
                    genres = ""

                attrs: dict[str, str] = {"genre": genres}
                if year:
                    attrs["year"] = year

                items.append(Item(name=title, attributes=attrs))

        return cls(items)
