from __future__ import annotations

import json
import re

from litellm import completion

from macrs.config import LLMConfig
from macrs.models import CandidateResponse, DialogueHistory, UserProfile

PLANNER_SYSTEM = """\
You are the Planner Agent in a multi-agent conversational recommender system.

Your job: choose the single best response from three candidates, each representing a different dialogue act.

## Dialogue Acts
- **asking**: Elicits more user preferences through a targeted question.
- **recommending**: Suggests a relevant item based on known preferences.
- **chitchat**: Keeps the conversation engaging and natural.

## Decision Criteria
1. **User profile completeness** — If we know very little, prefer asking.
2. **Dialogue progress** — If enough preferences are known, prefer recommending.
3. **Engagement** — If the user seems disengaged or the conversation is stalling, prefer chitchat.
4. **Naturalness** — The chosen response should feel like a coherent next turn.

## Output Format
Respond with ONLY a JSON object (no markdown fences):
{"choice": "<asking|recommending|chitchat>", "reasoning": "<one sentence>"}
"""


class PlannerAgent:
    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_config = llm_config

    def select(
        self,
        candidates: list[CandidateResponse],
        history: DialogueHistory,
        user_profile: UserProfile,
        planner_correction: str = "",
    ) -> CandidateResponse:
        candidate_text = "\n\n".join(
            f"### {c.agent_name} response\n{c.content}" for c in candidates
        )

        parts = [
            f"## User Profile\n{user_profile.summary()}",
            f"\n## Dialogue History (last 10 turns)\n{history.format(last_n=10)}",
            f"\n## Candidate Responses\n{candidate_text}",
        ]
        if planner_correction:
            parts.append(f"\n## Corrective Experience (from reflection)\n{planner_correction}")
        parts.append("\nChoose the best candidate.")

        user_msg = "\n".join(parts)

        resp = completion(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        raw = resp.choices[0].message.content.strip()
        choice = self._parse_choice(raw)

        # Find the matching candidate; fall back to first if parsing fails
        for c in candidates:
            if c.agent_name == choice:
                return c
        return candidates[0]

    def _parse_choice(self, raw: str) -> str:
        # Try JSON parse first
        try:
            data = json.loads(raw)
            return data.get("choice", "")
        except json.JSONDecodeError:
            pass
        # Fallback: look for the agent name in the text
        match = re.search(r'"choice"\s*:\s*"(asking|recommending|chitchat)"', raw)
        if match:
            return match.group(1)
        for name in ("asking", "recommending", "chitchat"):
            if name in raw.lower():
                return name
        return ""
