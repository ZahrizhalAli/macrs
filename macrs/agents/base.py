from __future__ import annotations

from litellm import completion

from macrs.config import LLMConfig
from macrs.models import CandidateResponse, DialogueHistory, UserProfile


class ResponderAgent:
    """Base class for the three responder agents."""

    agent_name: str = "base"
    system_prompt: str = ""

    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_config = llm_config

    def generate(
        self,
        history: DialogueHistory,
        user_profile: UserProfile,
        catalog_text: str,
    ) -> CandidateResponse:
        system = self._build_system(user_profile, catalog_text)
        messages = [{"role": "system", "content": system}]
        for m in history.messages:
            messages.append({"role": m.role, "content": m.content})

        resp = completion(
            model=self.llm_config.model,
            messages=messages,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )
        content = resp.choices[0].message.content.strip()
        return CandidateResponse(agent_name=self.agent_name, content=content)

    def _build_system(self, user_profile: UserProfile, catalog_text: str) -> str:
        parts = [
            self.system_prompt,
            f"\n## User Profile\n{user_profile.summary()}",
        ]
        if catalog_text:
            parts.append(f"\n## Available Items\n{catalog_text}")
        return "\n".join(parts)
