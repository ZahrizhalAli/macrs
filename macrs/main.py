"""CLI entry point for MACRS."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from macrs.catalog import Item, ItemCatalog
from macrs.config import LLMConfig
from macrs.engine import MACRSEngine


def sample_movie_catalog() -> ItemCatalog:
    """A small demo catalog. Replace or extend with your own domain."""
    return ItemCatalog([
        Item("Zootopia", {"genre": "animation, comedy", "year": "2016"}, "A bunny cop and a fox con artist uncover a conspiracy."),
        Item("Elemental", {"genre": "animation, romance", "year": "2023"}, "Fire and water elements discover they have a lot in common."),
        Item("Inception", {"genre": "sci-fi, thriller", "year": "2010"}, "A thief steals secrets by entering people's dreams."),
        Item("The Shawshank Redemption", {"genre": "drama", "year": "1994"}, "A banker sentenced to life in prison befriends a smuggler."),
        Item("Spider-Man: Across the Spider-Verse", {"genre": "animation, action", "year": "2023"}, "Miles Morales catapults across the multiverse."),
        Item("Spirited Away", {"genre": "animation, fantasy", "year": "2001"}, "A girl navigates a world of spirits to save her parents."),
        Item("The Grand Budapest Hotel", {"genre": "comedy, drama", "year": "2014"}, "A concierge and his protégé navigate a stolen painting plot."),
        Item("Interstellar", {"genre": "sci-fi, drama", "year": "2014"}, "Astronauts travel through a wormhole to find a new home for humanity."),
    ])


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="MACRS — Multi-Agent Conversational Recommender")
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model string (default: gpt-4o-mini)")
    parser.add_argument("--verbose", action="store_true", help="Show all candidate responses before the chosen one")
    args = parser.parse_args()

    cfg = LLMConfig(model=args.model)
    engine = MACRSEngine(catalog=sample_movie_catalog(), llm_config=cfg)

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
