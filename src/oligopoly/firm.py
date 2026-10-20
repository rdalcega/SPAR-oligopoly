"""One firm's transcript and the request built from it for each call (DECISIONS.md §6 item 3, summaries layout).

The firm keeps everything it has been shown or has answered (`transcript`), each entry tagged with the global
period it belongs to and its `kind`:

- "summary": the system-generated summary of a completed period (its results, any fallback notice, and the
  die or end-of-part line), written when the period ends;
- "chat": one chat round of the current period as this firm sees it (every firm, in its listing order);
- "prompt": the prompt of a decision; "answer": the firm's answer; "reask": a re-ask after an unusable answer.

Each request is built fresh by `section_messages`: the Part 1 instructions as the system message, then one user
message holding, in order,

1. Part 2's instructions, once Part 2 has started;
2. the summaries of the last `window` completed periods (counted across the part boundary), oldest first, and
   the firm's own cumulative profit;
3. the firm's latest private notes (if the run uses notes);
4. in Talk parts, the chat rounds of the current period so far (or a placeholder before any);
5. the prompt.

The firm's earlier answers are not shown again. Within a decision, each unusable answer and its re-ask follow
as assistant and user messages. The context record (`ctx`, layout "sections") holds transcript indices and the
cumulative-profit text, so `llm.reconstruct_request` rebuilds every request; records without a layout used
the earlier verbatim-window layout (`context_messages`).

Periods are numbered globally: Part 1 period t is t, Part 2 period t is (Part 1 length) + t, and the post-game
probe is (total periods) + 1.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import prompts

Message = dict[str, Any]
LAYOUT = "sections"


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
    """Earlier layout (records without `layout`): instructions, memory block, then the windowed turns, with
    consecutive same-role messages merged (`merge`; False rebuilds requests made before the merge)."""
    if not instructions:
        raise ValueError("a request needs the Part 1 instructions")
    out: list[Message] = [{"role": "system", "content": instructions[0]}]
    out += [{"role": "user", "content": text} for text in instructions[1:]]
    out.append({"role": "user", "content": memory})
    out += [{"role": m["role"], "content": m["content"]} for m in turns]
    return merge_roles(out) if merge else out


def _check_context(transcript: Sequence[Message], ctx: dict[str, Any]) -> None:
    """ValueError unless every index of `ctx` points into `transcript` at an entry of the right kind, before the
    decision's turns: summaries, chat rounds, the answer carrying the notes, and turns [lo, hi) opening on a prompt."""
    n = len(transcript)
    lo, hi = ctx["turns"]

    def kinds(ixs: Sequence[int], kind: str, what: str) -> None:
        if any(not (isinstance(i, int) and 0 <= i < lo) or transcript[i].get("kind") != kind for i in ixs):
            raise ValueError(f"context {what} {list(ixs)} do not point at earlier {kind} entries")

    if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi <= n) or transcript[lo].get("kind") != "prompt":
        raise ValueError(f"context turns [{lo}, {hi}) do not fit a transcript of {n} entries opening on a prompt")
    kinds(ctx["summaries"], "summary", "summaries")
    if ctx["chat"] is not None:
        kinds(ctx["chat"], "chat", "chat rounds")
    if ctx["notes"] and ctx["notes_at"] is not None:
        kinds([ctx["notes_at"]], "answer", "notes_at")


def section_messages(instructions: Sequence[str], transcript: Sequence[Message], ctx: dict[str, Any]) -> list[Message]:
    """The messages of one request in the summaries layout, from the firm's instructions, transcript and the
    request's context record."""
    k = ctx["instructions"]
    if not 1 <= k <= len(instructions):
        raise ValueError(f"context has {k} instruction messages, the firm {len(instructions)}")
    _check_context(transcript, ctx)
    body = [prompts.summaries_text([transcript[i]["content"] for i in ctx["summaries"]], ctx["totals"])]
    if ctx["notes"]:
        i = ctx["notes_at"]
        body.append(prompts.notes_text(None if i is None else json.loads(transcript[i]["content"])["notes"]))
    if ctx["chat"] is not None:
        body.append(prompts.chat_text([transcript[i]["content"] for i in ctx["chat"]]))
    lo, hi = ctx["turns"]
    body.append(transcript[lo]["content"])
    out: list[Message] = [{"role": "system", "content": instructions[0]},
                          {"role": "user", "content": "\n\n".join([*instructions[1:k], *body])}]
    out += [{"role": m["role"], "content": m["content"]} for m in transcript[lo + 1:hi]]
    return out


@dataclass
class Firm:
    letter: str
    order: list[str]  # listing order of all firms in this firm's results and chat rounds (fixed per market)
    window: int = 3  # completed periods whose summaries each request shows
    instructions: list[str] = field(default_factory=list)  # [Part 1] or [Part 1, Part 2]
    transcript: list[Message] = field(default_factory=list)  # {"role", "kind", "content", "period"}
    pending: list[str] = field(default_factory=list)  # notices for the summary of the period in progress
    notes: str | None = None  # the firm's latest private notes (its memory); None until it writes any
    notes_at: int | None = None  # transcript index of the answer that carried `notes`
    _ctx: dict[str, Any] | None = field(default=None, repr=False)  # the open decision's context

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
        """Queue a notice for the summary of the period in progress."""
        self.pending.append(text)

    def _append(self, role: str, kind: str, content: str, period: int) -> None:
        if self.transcript and period < self.transcript[-1]["period"]:
            raise RuntimeError("periods go forward")
        self.transcript.append({"role": role, "kind": kind, "content": content, "period": period})

    def ask(self, prompt: str, period: int, totals: str, notes: bool, chat: bool) -> tuple[list[Message],
                                                                                          dict[str, Any]]:
        """Open a decision with `prompt` in global period `period`; return the request's messages and its context
        record: number of instruction messages, transcript indices of the summaries shown, the cumulative-profit
        text `totals`, whether notes are shown and the index of the answer carrying them, the indices of this
        period's chat rounds (None: no chat section) and the decision's turns [lo, hi)."""
        if self._ctx is not None:
            raise RuntimeError("the previous decision is still open")
        summaries = [i for i, m in enumerate(self.transcript) if m["kind"] == "summary"]
        rounds = [i for i, m in enumerate(self.transcript) if m["kind"] == "chat" and m["period"] == period]
        self._append("user", "prompt", prompt, period)
        lo = len(self.transcript) - 1
        self._ctx = {"layout": LAYOUT, "instructions": len(self.instructions),
                     "summaries": summaries[-self.window:],
                     "totals": totals, "notes": notes, "notes_at": self.notes_at if notes else None,
                     "chat": rounds if chat else None, "turns": [lo, lo + 1]}
        return section_messages(self.instructions, self.transcript, self._ctx), dict(self._ctx)

    def reask(self, text: str) -> tuple[list[Message], dict[str, Any]]:
        """Re-ask within the open decision, after an unusable answer."""
        if self._ctx is None or self.transcript[-1]["kind"] != "answer":
            raise RuntimeError("a re-ask follows an answer within an open decision")
        self._append("user", "reask", text, self.transcript[-1]["period"])
        self._ctx = {**self._ctx, "turns": [self._ctx["turns"][0], len(self.transcript)]}
        return section_messages(self.instructions, self.transcript, self._ctx), dict(self._ctx)

    def answer(self, content: str, notes: str | None = None) -> None:
        """Record the firm's answer to the open request; `notes` (the capped notes it carried, if any) become its
        memory. Call `close` once the decision is over."""
        if self._ctx is None or self.transcript[-1]["kind"] not in ("prompt", "reask"):
            raise RuntimeError("an answer must follow a prompt or re-ask")
        self._append("assistant", "answer", content, self.transcript[-1]["period"])
        if notes is not None:
            self.notes, self.notes_at = notes, len(self.transcript) - 1

    def close(self) -> None:
        if self._ctx is None or self.transcript[-1]["kind"] != "answer":
            raise RuntimeError("a decision closes after an answer")
        self._ctx = None

    def add_chat(self, period: int, text: str) -> None:
        """A chat round of global period `period` as this firm sees it (prompts.round_text)."""
        if self._ctx is not None:
            raise RuntimeError("a decision is still open")
        self._append("user", "chat", text, period)

    def summarise(self, period: int, results: str, closing: str | None = None) -> None:
        """The summary of completed global period `period`: its results, the queued notices, then `closing` (the
        die or end-of-part line)."""
        if self._ctx is not None:
            raise RuntimeError("a decision is still open")
        self._append("user", "summary", "\n".join([results, *self.pending, *([closing] if closing else [])]),
                     period)
        self.pending.clear()
