"""Offline stand-in for `OpenRouterClient`: same `chat` interface, deterministic from the seed.

It answers price and chat prompts with a mix of valid JSON in assorted wrappings (code fences, leaked
think blocks, prose, single quotes, numeric strings), occasional malformed or invalid output (to exercise
re-asks and fallbacks), chat messages and passes. Notes and messages carry a `#<seed>` token so tests can
trace where a text travelled.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import ChatResult, Usage

PHRASES = ["Shall we all set 100?", "I will keep my price at 100.", "Let's not undercut.",
           "Prices were low last period.", "I'll match the lowest price.", "Hello everyone."]


class MockClient:
    def __init__(self, log_path: str | Path | None = None, malformed_rate: float = 0.06,
                 record: bool = False, notes_words: int = 0):
        self.notes_words = notes_words
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.malformed_rate = malformed_rate
        self.record = record
        self.calls: list[dict[str, Any]] = []  # tag, seed, messages (only when record=True), content

    async def __aenter__(self) -> MockClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    def respond(self, kind: str, seed: int) -> str:
        """Raw content for a `kind` ("price", "message" or "transition") prompt."""
        rng = random.Random(seed)
        if rng.random() < self.malformed_rate:
            bad = (["I think a high price is best.", '{"price": 150}', '{"price": 57.5}', '{"notes": "x", "price": ',
                    ""] if kind == "price" else ['{"msg": "hi"}', '{"message": ["a"]}', "Sure, I will write.", ""])
            return rng.choice(bad)
        notes = f"note #{seed}: " + rng.choice(["watching rivals", "keep high", "test a cut", "match lowest"])
        if self.notes_words:  # notes at the stated length (dry-run projections)
            notes += " memo" * max(0, self.notes_words - len(notes.split()))
        if kind == "transition":
            obj: dict[str, Any] = {"notes": notes + " (for Part 2)"}
        elif kind == "price":
            value: Any = rng.choice([100, 100, 100, 99, 99, rng.randint(0, 100)])
            if rng.random() < 0.05:
                value = rng.choice([str(value), float(value)])
            obj = {"notes": notes, "price": value}
        else:
            r = rng.random()
            msg = None if r < 0.35 else rng.choice(["pass", ""]) if r < 0.45 else f"#{seed} {rng.choice(PHRASES)}"
            if msg and rng.random() < 0.05:
                msg += " " + "blah " * 70
            obj = {"notes": notes, "message": msg}
        js = json.dumps(obj)
        style = rng.random()
        if style < 0.6:
            return js
        if style < 0.7:
            return f"```json\n{js}\n```"
        if style < 0.8:
            return f"<think>Let me consider {{'price': 1}}.</think>\n{js}"
        if style < 0.9:
            return f"Here is my decision:\n{js}\nThat is final."
        return repr(obj)  # Python-style single quotes

    def respond_probe(self, messages: Sequence[dict[str, Any]], seed: int) -> str:
        """A probe answer naming the firms the probe asks about; sometimes wrong or malformed."""
        rng = random.Random(seed)
        if rng.random() < self.malformed_rate:
            return rng.choice(['{"max_consumer_price": 100}', "I am not sure.", ""])
        return json.dumps(probe_answer(messages, rng))

    async def chat(self, messages: Sequence[dict[str, Any]], *, model: str, provider: str, seed: int | None = None,
                   tag: dict[str, Any] | None = None, **settings: Any) -> ChatResult:
        phase = (tag or {}).get("phase")
        if phase == "probe":
            content = self.respond_probe(messages, seed or 0)
        else:
            content = self.respond(phase if phase in ("price", "transition") else "message", seed or 0)
        prompt_chars = sum(len(m["content"]) for m in messages)
        usage = Usage(prompt_tokens=prompt_chars // 4, completion_tokens=len(content) // 4 + 20,
                      reasoning_tokens=20, total_tokens=prompt_chars // 4 + len(content) // 4 + 20)
        call = {"tag": tag or {}, "seed": seed, "content": content}
        if self.record:
            call["messages"] = [dict(m) for m in messages]
        self.calls.append(call)
        if self.log_path:
            rec = {"ts": datetime.now(timezone.utc).isoformat(), "model_requested": model,
                   "provider_requested": provider, "provider_served": "mock", "seed": seed, "tag": tag or {},
                   "settings": {k: v for k, v in settings.items() if k != "cache_upto"}, "content": content,
                   "usage": usage.__dict__, "cost_usd": 0.0, "error": None}
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return ChatResult(content=content, reasoning="(mock reasoning)", reasoning_details=None,
                          finish_reason="stop", native_finish_reason="stop", provider="mock", model=model,
                          generation_id=f"mock-{seed}", usage=usage, cost_usd=0.0, latency_s=0.0, attempts=1,
                          seed_possibly_ignored=False, provider_mismatch=False)


def probe_answer(messages: Sequence[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """A plausible post-game probe answer for the firms listed in the last message's JSON format (offline
    clients only)."""
    last = messages[-1]["content"] if messages else ""
    text = last if isinstance(last, str) else "".join(b.get("text", "") for b in last)
    letters = re.findall(r'"([A-Z])": <int>', text)
    return {"max_consumer_price": rng.choice([100, 100, 100, 99]), "best_common_price": rng.choice([100, 100, 1]),
            "last_period_prices": {L: rng.choice([100, 99, 50]) for L in letters},
            "deviations": rng.choice(["no chat", "No firm deviated.", "Firm B undercut in period 3."])}
