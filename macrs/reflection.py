"""User Feedback-Aware Reflection Mechanism.

Two levels of reflection:
1. Information-level: Extracts/updates user preferences from behavioral signals.
2. Strategy-level: Detects failures and generates corrective strategy suggestions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from litellm import completion

from macrs.config import LLMConfig
from macrs.models import DialogueHistory, UserProfile

logger = logging.getLogger("macrs.reflection")


# ---------------------------------------------------------------------------
# Feedback signals
# ---------------------------------------------------------------------------

class FeedbackSignal(Enum):
    REJECTION = "rejection"          # user rejected a recommendation
    DISENGAGEMENT = "disengagement"  # short/vague reply, topic change
    REPETITION = "repetition"        # dialogue looping without progress
    POSITIVE = "positive"            # user expressed interest or agreement
    NEUTRAL = "neutral"              # no clear signal


@dataclass
class FeedbackResult:
    signal: FeedbackSignal
    evidence: str  # short explanation of why this signal was detected


# Rule-based rejection patterns
_REJECTION_PATTERNS = re.compile(
    r"\b(no|nah|nope|not really|don'?t like|not into|not what i|wrong|"
    r"skip|pass|something else|different|hate|dislike|boring)\b",
    re.IGNORECASE,
)

_POSITIVE_PATTERNS = re.compile(
    r"\b(yes|yeah|love|great|sounds good|perfect|exactly|that'?s it|"
    r"interesting|cool|awesome|nice|tell me more|more about)\b",
    re.IGNORECASE,
)


def detect_feedback(
    user_message: str,
    history: DialogueHistory,
    last_agent: str | None = None,
) -> FeedbackResult:
    """Detect feedback signal from the user's latest message (rule-based)."""
    msg = user_message.strip()

    # Very short replies suggest disengagement
    if len(msg.split()) <= 2 and not _POSITIVE_PATTERNS.search(msg):
        return FeedbackResult(FeedbackSignal.DISENGAGEMENT, f"Very short reply: '{msg}'")

    # Rejection of a recommendation
    if last_agent == "recommending" and _REJECTION_PATTERNS.search(msg):
        return FeedbackResult(FeedbackSignal.REJECTION, f"Rejected recommendation: '{msg}'")

    # General rejection / negative sentiment
    if _REJECTION_PATTERNS.search(msg):
        return FeedbackResult(FeedbackSignal.REJECTION, f"Negative signal: '{msg}'")

    # Positive engagement
    if _POSITIVE_PATTERNS.search(msg):
        return FeedbackResult(FeedbackSignal.POSITIVE, f"Positive signal: '{msg}'")

    # Repetition detection: check if last 4 assistant messages repeat the same act
    if last_agent and len(history.messages) >= 8:
        recent_assistant = [m.content for m in history.messages[-8:] if m.role == "assistant"]
        if len(recent_assistant) >= 3:
            # Simple heuristic: if responses are very similar in length/structure
            lengths = [len(r) for r in recent_assistant[-3:]]
            if max(lengths) - min(lengths) < 20:
                return FeedbackResult(FeedbackSignal.REPETITION, "Recent responses appear repetitive")

    return FeedbackResult(FeedbackSignal.NEUTRAL, "No strong signal detected")


# ---------------------------------------------------------------------------
# Information-level reflection
# ---------------------------------------------------------------------------

_INFO_REFLECT_SYSTEM = """\
You are an information extraction module in a conversational recommender system.

Given the latest dialogue exchange and the current user profile, extract any NEW preference information.

Return ONLY a JSON object (no markdown fences):
{
  "preferences": {"key": "value", ...},
  "liked": ["item name", ...],
  "disliked": ["item name", ...],
  "reasoning": "one sentence explaining what you inferred"
}

Rules:
- Only include NEW information not already in the profile.
- Keys for preferences should be: genre, year, mood, theme, director, actor, or other relevant attributes.
- If nothing new, return empty lists/dicts with reasoning "No new information."
"""


@dataclass
class InfoReflection:
    new_preferences: dict[str, str]
    new_liked: list[str]
    new_disliked: list[str]
    reasoning: str


def reflect_information(
    history: DialogueHistory,
    user_profile: UserProfile,
    llm_config: LLMConfig,
) -> InfoReflection:
    """Extract new preference information from recent dialogue."""
    user_msg = (
        f"## Current User Profile\n{user_profile.summary()}\n\n"
        f"## Recent Dialogue (last 4 turns)\n{history.format(last_n=4)}"
    )

    resp = completion(
        model=llm_config.model,
        messages=[
            {"role": "system", "content": _INFO_REFLECT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=200,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return InfoReflection(
        new_preferences=data.get("preferences", {}),
        new_liked=data.get("liked", []),
        new_disliked=data.get("disliked", []),
        reasoning=data.get("reasoning", "Parse failed"),
    )


def apply_info_reflection(profile: UserProfile, reflection: InfoReflection) -> None:
    """Merge new information into the user profile."""
    for k, v in reflection.new_preferences.items():
        profile.preferences[k] = v
    for item in reflection.new_liked:
        if item not in profile.liked:
            profile.liked.append(item)
    for item in reflection.new_disliked:
        if item not in profile.disliked:
            profile.disliked.append(item)


# ---------------------------------------------------------------------------
# Strategy-level reflection
# ---------------------------------------------------------------------------

_STRATEGY_REFLECT_SYSTEM = """\
You are a strategy reflection module in a conversational recommender system.

A dialogue failure has been detected. Analyze the situation and produce actionable advice.

Return ONLY a JSON object (no markdown fences):
{
  "asking_suggestion": "specific advice for the asking agent, or empty string",
  "recommending_suggestion": "specific advice for the recommending agent, or empty string",
  "chitchat_suggestion": "specific advice for the chitchat agent, or empty string",
  "planner_correction": "what the planner should avoid or prefer next, or empty string",
  "reasoning": "one sentence explaining the diagnosed problem"
}

Be specific and actionable. Bad: "ask better questions." Good: "ask about preferred decade since the user rejected two recent films."
"""


@dataclass
class StrategyReflection:
    asking_suggestion: str
    recommending_suggestion: str
    chitchat_suggestion: str
    planner_correction: str
    reasoning: str


def reflect_strategy(
    feedback: FeedbackResult,
    history: DialogueHistory,
    user_profile: UserProfile,
    llm_config: LLMConfig,
) -> StrategyReflection:
    """Generate strategy corrections based on a detected failure."""
    user_msg = (
        f"## Detected Failure\nSignal: {feedback.signal.value}\nEvidence: {feedback.evidence}\n\n"
        f"## User Profile\n{user_profile.summary()}\n\n"
        f"## Recent Dialogue (last 6 turns)\n{history.format(last_n=6)}"
    )

    resp = completion(
        model=llm_config.model,
        messages=[
            {"role": "system", "content": _STRATEGY_REFLECT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=250,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return StrategyReflection(
        asking_suggestion=data.get("asking_suggestion", ""),
        recommending_suggestion=data.get("recommending_suggestion", ""),
        chitchat_suggestion=data.get("chitchat_suggestion", ""),
        planner_correction=data.get("planner_correction", ""),
        reasoning=data.get("reasoning", "Parse failed"),
    )


# ---------------------------------------------------------------------------
# Reflection log
# ---------------------------------------------------------------------------

@dataclass
class ReflectionEntry:
    turn: int
    feedback: FeedbackResult
    info_reflection: InfoReflection | None = None
    strategy_reflection: StrategyReflection | None = None


class ReflectionLog:
    """Append-only log of all reflection events during a session."""

    def __init__(self) -> None:
        self.entries: list[ReflectionEntry] = []

    def add(self, entry: ReflectionEntry) -> None:
        self.entries.append(entry)
        # Log to Python logger for visibility
        logger.info(
            "Turn %d | signal=%s | evidence=%s",
            entry.turn, entry.feedback.signal.value, entry.feedback.evidence,
        )
        if entry.info_reflection:
            logger.info(
                "  [info-reflect] prefs=%s liked=%s disliked=%s | %s",
                entry.info_reflection.new_preferences,
                entry.info_reflection.new_liked,
                entry.info_reflection.new_disliked,
                entry.info_reflection.reasoning,
            )
        if entry.strategy_reflection:
            logger.info(
                "  [strategy-reflect] %s",
                entry.strategy_reflection.reasoning,
            )
            if entry.strategy_reflection.asking_suggestion:
                logger.info("    asking: %s", entry.strategy_reflection.asking_suggestion)
            if entry.strategy_reflection.recommending_suggestion:
                logger.info("    recommending: %s", entry.strategy_reflection.recommending_suggestion)
            if entry.strategy_reflection.chitchat_suggestion:
                logger.info("    chitchat: %s", entry.strategy_reflection.chitchat_suggestion)
            if entry.strategy_reflection.planner_correction:
                logger.info("    planner: %s", entry.strategy_reflection.planner_correction)

    @property
    def latest_strategy(self) -> StrategyReflection | None:
        """Return the most recent strategy reflection, if any."""
        for entry in reversed(self.entries):
            if entry.strategy_reflection:
                return entry.strategy_reflection
        return None
