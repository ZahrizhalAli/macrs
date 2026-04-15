"""Conversation engine — orchestrates one turn of the MACRS Act Planning loop."""

from __future__ import annotations

from dataclasses import dataclass

from macrs.agents.asking import AskingAgent
from macrs.agents.chitchat import ChitChatAgent
from macrs.agents.planner import PlannerAgent
from macrs.agents.recommending import RecommendingAgent
from macrs.catalog import ItemCatalog
from macrs.config import LLMConfig
from macrs.models import CandidateResponse, DialogueHistory, UserProfile


@dataclass
class TurnResult:
    chosen: CandidateResponse
    all_candidates: list[CandidateResponse]


class MACRSEngine:
    def __init__(
        self,
        catalog: ItemCatalog,
        llm_config: LLMConfig | None = None,
    ) -> None:
        cfg = llm_config or LLMConfig()
        self.catalog = catalog
        self.history = DialogueHistory()
        self.user_profile = UserProfile()

        self.asking = AskingAgent(cfg)
        self.recommending = RecommendingAgent(cfg)
        self.chitchat = ChitChatAgent(cfg)
        self.planner = PlannerAgent(cfg)

    def turn(self, user_input: str) -> TurnResult:
        """Process one user message and return the system's chosen response."""
        self.history.add("user", user_input)

        # Step 1: All three responder agents generate candidates.
        # Recommending agent gets the catalog so it can call search_movies.
        # Asking and chitchat don't need the catalog — they don't recommend items.
        candidates = [
            self.asking.generate(self.history, self.user_profile),
            self.recommending.generate(self.history, self.user_profile, self.catalog),
            self.chitchat.generate(self.history, self.user_profile),
        ]

        # Step 2: Planner selects the best candidate
        chosen = self.planner.select(candidates, self.history, self.user_profile)

        # Record the chosen response in dialogue history
        self.history.add("assistant", chosen.content)

        return TurnResult(chosen=chosen, all_candidates=candidates)
