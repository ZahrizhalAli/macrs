"""Conversation engine — orchestrates one turn of the MACRS loop.

Per-turn flow:
1. Detect feedback signal from user's message
2. Run information-level reflection (always) → update user profile
3. Run strategy-level reflection (on failure signals) → update agent hints
4. Three responder agents generate candidates (with strategy hints injected)
5. Planner selects best candidate (with corrective experience injected)
6. Log reflection entry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from macrs.agents.asking import AskingAgent
from macrs.agents.chitchat import ChitChatAgent
from macrs.agents.planner import PlannerAgent
from macrs.agents.recommending import RecommendingAgent
from macrs.catalog import ItemCatalog
from macrs.config import LLMConfig
from macrs.models import CandidateResponse, DialogueHistory, UserProfile
from macrs.reflection import (
    FeedbackSignal,
    ReflectionEntry,
    ReflectionLog,
    apply_info_reflection,
    detect_feedback,
    reflect_information,
    reflect_strategy,
)

logger = logging.getLogger("macrs.engine")

# Signals that trigger strategy-level reflection (the expensive one)
_FAILURE_SIGNALS = {FeedbackSignal.REJECTION, FeedbackSignal.DISENGAGEMENT, FeedbackSignal.REPETITION}


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
        self.reflection_log = ReflectionLog()
        self._turn_count = 0
        self._last_agent: str | None = None

        self.asking = AskingAgent(cfg)
        self.recommending = RecommendingAgent(cfg)
        self.chitchat = ChitChatAgent(cfg)
        self.planner = PlannerAgent(cfg)
        self._llm_config = cfg

    def turn(self, user_input: str) -> TurnResult:
        """Process one user message and return the system's chosen response."""
        self._turn_count += 1
        self.history.add("user", user_input)

        # --- Reflection Phase ---

        # Step 1: Detect feedback signal
        feedback = detect_feedback(user_input, self.history, self._last_agent)
        logger.debug("Feedback: %s — %s", feedback.signal.value, feedback.evidence)

        # Step 2: Information-level reflection (always runs — cheap, keeps profile fresh)
        info_ref = reflect_information(self.history, self.user_profile, self._llm_config)
        apply_info_reflection(self.user_profile, info_ref)

        # Step 3: Strategy-level reflection (only on failure signals)
        strat_ref = None
        if feedback.signal in _FAILURE_SIGNALS:
            strat_ref = reflect_strategy(feedback, self.history, self.user_profile, self._llm_config)

        # Log this reflection event
        entry = ReflectionEntry(
            turn=self._turn_count,
            feedback=feedback,
            info_reflection=info_ref,
            strategy_reflection=strat_ref,
        )
        self.reflection_log.add(entry)

        # --- Generation Phase ---

        # Build per-agent strategy hints from the latest strategy reflection
        latest_strat = self.reflection_log.latest_strategy
        asking_hint = latest_strat.asking_suggestion if latest_strat else ""
        rec_hint = latest_strat.recommending_suggestion if latest_strat else ""
        chat_hint = latest_strat.chitchat_suggestion if latest_strat else ""
        planner_hint = latest_strat.planner_correction if latest_strat else ""

        # Step 4: Responder agents generate candidates
        candidates = [
            self.asking.generate(self.history, self.user_profile, strategy_hint=asking_hint),
            self.recommending.generate(self.history, self.user_profile, self.catalog, strategy_hint=rec_hint),
            self.chitchat.generate(self.history, self.user_profile, strategy_hint=chat_hint),
        ]

        # Step 5: Planner selects best candidate
        chosen = self.planner.select(
            candidates, self.history, self.user_profile, planner_correction=planner_hint,
        )

        self.history.add("assistant", chosen.content)
        self._last_agent = chosen.agent_name

        return TurnResult(chosen=chosen, all_candidates=candidates)
