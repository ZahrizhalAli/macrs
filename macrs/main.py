"""CLI entry point for MACRS."""

from __future__ import annotations

import argparse
import logging
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


def _print_reflection(engine: MACRSEngine) -> None:
    """Print the latest reflection entry."""
    if not engine.reflection_log.entries:
        return
    entry = engine.reflection_log.entries[-1]

    print(f"\n--- Reflection (turn {entry.turn}) ---")
    print(f"  Signal: {entry.feedback.signal.value} — {entry.feedback.evidence}")

    if entry.info_reflection:
        ir = entry.info_reflection
        parts = []
        if ir.new_preferences:
            parts.append(f"prefs={ir.new_preferences}")
        if ir.new_liked:
            parts.append(f"liked={ir.new_liked}")
        if ir.new_disliked:
            parts.append(f"disliked={ir.new_disliked}")
        if parts:
            print(f"  [info] {', '.join(parts)}")
        print(f"  [info] {ir.reasoning}")

    if entry.strategy_reflection:
        sr = entry.strategy_reflection
        print(f"  [strategy] {sr.reasoning}")
        if sr.asking_suggestion:
            print(f"    → asking: {sr.asking_suggestion}")
        if sr.recommending_suggestion:
            print(f"    → recommending: {sr.recommending_suggestion}")
        if sr.chitchat_suggestion:
            print(f"    → chitchat: {sr.chitchat_suggestion}")
        if sr.planner_correction:
            print(f"    → planner: {sr.planner_correction}")

    print(f"  Profile: {engine.user_profile.summary()}")
    print("-------------------------------")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="MACRS — Multi-Agent Conversational Recommender")
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model string (default: gpt-4o-mini)")
    parser.add_argument("--catalog", default=None, help="Path to MovieLens movies.csv (default: data/ml-latest-small/movies.csv)")
    parser.add_argument("--verbose", action="store_true", help="Show all candidate responses and reflection logs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

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
            _print_reflection(engine)
            print("\n--- Candidates ---")
            for c in result.all_candidates:
                print(f"  [{c.agent_name}]: {c.content}")
            print("------------------")

        print(f"\nMACRS [{result.chosen.agent_name}]: {result.chosen.content}\n")


if __name__ == "__main__":
    main()
