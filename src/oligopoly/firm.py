"""One firm's append-only conversation.

The message list only ever grows (never rewritten), so every request is a prefix-extension of the
previous one and providers can reuse the cached prefix. Text the firm must see before its next decision
(results, end of part, Part 2 instructions, fallback notices) is queued with `tell` and sent together with
the next prompt in a single user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass
class Firm:
    letter: str
    order: list[str]  # listing order of all firms in this firm's results (fixed per market)
    messages: list[Message] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def start(self, system: str) -> None:
        if self.messages:
            raise RuntimeError("conversation already started")
        self.messages.append({"role": "system", "content": system})

    def tell(self, text: str) -> None:
        """Queue text for the next user message."""
        self.pending.append(text)

    def ask(self, prompt: str) -> list[Message]:
        """Append a user message (queued text + prompt) and return a snapshot of the conversation to send."""
        self._user("\n\n".join([*self.pending, prompt]))
        self.pending.clear()
        return list(self.messages)

    def answer(self, content: str) -> None:
        if self.messages[-1]["role"] != "user":
            raise RuntimeError("an assistant turn must follow a user turn")
        self.messages.append({"role": "assistant", "content": content})

    def flush(self) -> None:
        """Append any queued text as a final user message (end of the experiment; never sent)."""
        if self.pending:
            self._user("\n\n".join(self.pending))
            self.pending.clear()

    def _user(self, text: str) -> None:
        if self.messages[-1]["role"] == "user":
            raise RuntimeError("two user turns in a row")
        self.messages.append({"role": "user", "content": text})
