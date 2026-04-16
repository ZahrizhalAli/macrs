from __future__ import annotations

import json

from litellm import completion

from macrs.catalog import ItemCatalog
from macrs.config import LLMConfig
from macrs.models import CandidateResponse, DialogueHistory, UserProfile
from macrs.tools import SEARCH_MOVIES_TOOL, execute_search


class ResponderAgent:
    """Base class for the three responder agents.

    Agents that set `use_tools = True` get access to the search_movies tool
    and will run a tool-use loop (up to max_tool_calls rounds).
    """

    agent_name: str = "base"
    system_prompt: str = ""
    use_tools: bool = False
    max_tool_calls: int = 3

    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_config = llm_config

    def generate(
        self,
        history: DialogueHistory,
        user_profile: UserProfile,
        catalog: ItemCatalog | None = None,
        strategy_hint: str = "",
    ) -> CandidateResponse:
        system = self._build_system(user_profile, strategy_hint)
        messages: list[dict] = [{"role": "system", "content": system}]
        for m in history.messages:
            messages.append({"role": m.role, "content": m.content})

        # Agents without tools — single call, same as before
        if not self.use_tools or catalog is None:
            resp = completion(
                model=self.llm_config.model,
                messages=messages,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return CandidateResponse(agent_name=self.agent_name, content=content.strip())

        # Tool-use loop: agent can call search_movies, see results, then respond
        tools = [SEARCH_MOVIES_TOOL]
        for _ in range(self.max_tool_calls):
            resp = completion(
                model=self.llm_config.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            msg = resp.choices[0].message

            # No tool call — agent is done, return its text
            if not msg.tool_calls:
                content = msg.content or ""
                return CandidateResponse(agent_name=self.agent_name, content=content.strip())

            # Process tool calls
            messages.append(msg.model_dump())
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "search_movies":
                    args = json.loads(tool_call.function.arguments)
                    result = execute_search(catalog, args)
                else:
                    result = f"Unknown tool: {tool_call.function.name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # Exhausted tool calls — do a final call without tools to force a text response
        resp = completion(
            model=self.llm_config.model,
            messages=messages,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        return CandidateResponse(agent_name=self.agent_name, content=content.strip())

    def _build_system(self, user_profile: UserProfile, strategy_hint: str = "") -> str:
        parts = [
            self.system_prompt,
            f"\n## User Profile\n{user_profile.summary()}",
        ]
        if strategy_hint:
            parts.append(f"\n## Strategy Guidance (from reflection)\n{strategy_hint}")
        return "\n".join(parts)
