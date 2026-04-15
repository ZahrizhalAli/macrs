"""CLI entry point for MACRS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from macrs.catalog import ItemCatalog
from macrs.config import LLMConfig
from macrs.engine import MACRSEngine

# Default path relative to project root
DEFAULT_MOVIELENS = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small" / "movies.csv"


def load_catalog(path: str | None) -> ItemCatalog:
    csv_path = Path(path) if path else DEFAULT_MOVIELENS
    if csv_path.exists():
        catalog = ItemCatalog.from_movielens(csv_path)
        print(f"Loaded {len(catalog.items)} movies from {csv_path.name}")
        return catalog
    print(f"Warning: {csv_path} not found — starting with empty catalog", file=sys.stderr)
    return ItemCatalog()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="MACRS — Multi-Agent Conversational Recommender")
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model string (default: gpt-4o-mini)")
    parser.add_argument("--catalog", default=None, help="Path to MovieLens movies.csv (default: data/ml-latest-small/movies.csv)")
    parser.add_argument("--verbose", action="store_true", help="Show all candidate responses before the chosen one")
    args = parser.parse_args()

    cfg = LLMConfig(model=args.model)
    engine = MACRSEngine(catalog=load_catalog(args.catalog), llm_config=cfg)

    print("MACRS — Multi-Agent Conversational Recommender")
    print(f"Model: {cfg.model}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            result = engine.turn(user_input)
        except Exception as e:
            print(f"[Error] {e}", file=sys.stderr)
            continue

        if args.verbose:
            print("\n--- Candidates ---")
            for c in result.all_candidates:
                print(f"[{c.agent_name}]: {c.content}")
            print("------------------")

        print(f"\nMACRS [{result.chosen.agent_name}]: {result.chosen.content}\n")


if __name__ == "__main__":
    main()
