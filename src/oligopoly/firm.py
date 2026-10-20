"""One firm's transcript and the bounded context built from it for each call (DECISIONS.md §6 item 3, 2026-10-20).

The firm keeps every turn it has had (`transcript`: user and assistant messages exactly as they occurred, each
tagged with the global period it belongs to), but each request is built fresh by `context_messages`:

1. the instructions: Part 1's as the system message; Part 2's as the next (user) message once Part 2 has started;
2. a memory block (user message): the firm's latest private notes and its own cumulative profit;
3. the verbatim turns of the last `window` completed periods (counted across the part boundary) and of the
   current period so far, ending with the new prompt.

Consecutive messages of the same role (Part 2's instructions, the memory block and the window's opening user
turn) are merged into one message, joined by a blank line, so the request's roles alternate system, user,
assistant, user, ... as strict chat templates require. The context record says so (`merged`), and
`llm.reconstruct_request` rebuilds the merged messages; records from before the merge have no `merged` key.

Periods are numbered globally: Part 1 period t is t, Part 2 period t is (Part 1 length) + t, and the post-game
probe is (total periods) + 1. A period's turns start with the user message that opens it, which carries the
results of the period before. So a call in period g sees the turns of periods g - window .. g: the results of
periods g - window - 1 .. g - 1, the chat and answers of periods g - window .. g - 1, and period g so far.

Text the firm must see before its next decision (results, end of part, fallback notices) is queued with `tell`
and sent together with the next prompt in a single user message.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


def merge_roles(messages: Sequence[Message]) -> list[Message]:
    """Merge each run of consecutive same-role messages into one, joining their contents with a blank line."""
    out: list[Message] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def context_messages(instructions: Sequence[str], memory: str, turns: Sequence[Message],
                     merge: bool = True) -> list[Message]:
    """The messages of one request: instructions, memory block, then the windowed turns (role and content
    only), with consecutive same-role messages merged (`merge`; False rebuilds requests made before the
    merge)."""
    if not instructions:
        raise ValueError("a request needs the Part 1 instructions")
    out: list[Message] = [{"role": "system", "content": instructions[0]}]
    out += [{"role": "user", "content": text} for text in instructions[1:]]
    out.append({"role": "user", "content": memory})
    out += [{"role": m["role"], "content": m["content"]} for m in turns]
    return merge_roles(out) if merge else out


def window_start(transcript: Sequence[Message], period: int, window: int) -> int:
    """Index of the first transcript turn of period `period - window` or later."""
    return next((i for i, m in enumerate(transcript) if m["period"] >= period - window), len(transcript))


@dataclass
class Firm:
    letter: str
    order: list[str]  # listing order of all firms in this firm's results (fixed per market)
    window: int = 3  # completed periods shown verbatim
    instructions: list[str] = field(default_factory=list)  # [Part 1] or [Part 1, Part 2]
    transcript: list[Message] = field(default_factory=list)  # {"role", "content", "period"}
    pending: list[str] = field(default_factory=list)
    notes: str | None = None  # the firm's latest private notes (its memory); None until it writes any

    def start(self, instructions: str) -> None:
        if self.instructions:
            raise RuntimeError("instructions already given")
        self.instructions.append(instructions)

    def add_instructions(self, text: str) -> None:
        """Part 2's instructions: shown in every later request."""
        if len(self.instructions) != 1:
            raise RuntimeError("Part 2 instructions follow Part 1's exactly once")
        self.instructions.append(text)

    def tell(self, text: str) -> None:
        """Queue text for the next user message."""
        self.pending.append(text)

    def ask(self, prompt: str, period: int, memory: str) -> tuple[list[Message], dict[str, Any]]:
        """Append a user message (queued text + prompt) in global period `period`; return the request's messages
        and its context record (number of instruction messages, memory text, transcript window [lo, hi), and
        `merged`: same-role messages merged)."""
        if self.transcript and self.transcript[-1]["role"] == "user":
            raise RuntimeError("two user turns in a row")
        if self.transcript and period < self.transcript[-1]["period"]:
            raise RuntimeError("periods go forward")
        self.transcript.append({"role": "user", "content": "\n\n".join([*self.pending, prompt]), "period": period})
        self.pending.clear()
        lo, hi = window_start(self.transcript, period, self.window), len(self.transcript)
        ctx = {"instructions": len(self.instructions), "memory": memory, "window": [lo, hi], "merged": True}
        return context_messages(self.instructions, memory, self.transcript[lo:hi]), ctx

    def answer(self, content: str) -> None:
        if not self.transcript or self.transcript[-1]["role"] != "user":
            raise RuntimeError("an assistant turn must follow a user turn")
        self.transcript.append({"role": "assistant", "content": content, "period": self.transcript[-1]["period"]})

    def flush(self, period: int) -> None:
        """Append any queued text as a final user message (end of the experiment; never sent)."""
        if self.pending:
            self.transcript.append({"role": "user", "content": "\n\n".join(self.pending), "period": period})
            self.pending.clear()
