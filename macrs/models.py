from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class UserProfile:
    """Accumulated knowledge about the user's preferences."""

    liked: list[str] = field(default_factory=list)
    disliked: list[str] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)  # e.g. {"genre": "sci-fi"}

    def summary(self) -> str:
        parts: list[str] = []
        if self.preferences:
            parts.append("Preferences: " + ", ".join(f"{k}={v}" for k, v in self.preferences.items()))
        if self.liked:
            parts.append("Liked: " + ", ".join(self.liked))
        if self.disliked:
            parts.append("Disliked: " + ", ".join(self.disliked))
        return "\n".join(parts) if parts else "No preferences known yet."


@dataclass
class DialogueHistory:
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def format(self, last_n: int | None = None) -> str:
        msgs = self.messages if last_n is None else self.messages[-last_n:]
        return "\n".join(f"{m.role}: {m.content}" for m in msgs)


@dataclass
class CandidateResponse:
    agent_name: str  # "asking", "recommending", "chitchat"
    content: str
