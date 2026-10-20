"""Extract and validate a firm's JSON answer from raw model content.

Extraction tolerates code fences, leading or trailing prose, leaked `<think>…</think>` blocks and
Python-style single quotes; it takes the last JSON object that contains the required key. Validation
returns a `Parsed` whose `error` (if any) is phrased for the re-ask prompt, and whose `warnings` record
every coercion or truncation so that it can be reported.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_THINK_CLOSE = re.compile(r"</(think|thinking|reasoning)>", re.I)
_INT_STR = re.compile(r"^\s*[+-]?\d+(\.0*)?\s*$")
PASS_WORDS = {"", "pass", "none", "null", "no message", "(pass)"}


@dataclass
class Parsed:
    value: int | str | None = None  # price, or message text (None = pass)
    notes: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    obj: dict[str, Any] | None = None  # the extracted JSON object

    @property
    def ok(self) -> bool:
        return self.error is None


def strip_reasoning(text: str) -> str:
    """Remove leaked reasoning blocks; with an unmatched closing tag, keep only what follows it."""
    text = _THINK.sub("", text)
    if (m := list(_THINK_CLOSE.finditer(text))):
        text = text[m[-1].end():]
    return text.strip()


def _balanced(text: str, start: int) -> str | None:
    """Substring from text[start] == '{' to its matching '}', respecting single- and double-quoted strings."""
    depth, quote, esc = 0, None, False
    for i in range(start, len(text)):
        c = text[i]
        if quote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _load(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    pythonish = re.sub(r"\bnull\b", "None", re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", s)))
    for cand in (s, pythonish):
        try:
            return ast.literal_eval(cand)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    return None


def extract_object(text: str | None, key: str) -> dict[str, Any] | None:
    """The last top-level JSON object in `text` that has `key`; failing that, the last nested one."""
    if not text:
        return None
    text = strip_reasoning(text)
    top = nested = None
    top_end = -1  # end of the last parsed top-level object
    for m in re.finditer(r"\{", text):
        sub = _balanced(text, m.start())
        if sub is None:
            continue
        obj = _load(sub)
        if not isinstance(obj, dict):
            continue
        inside = m.start() < top_end
        if not inside:
            top_end = m.start() + len(sub)
        if key in obj:
            if inside:
                nested = obj
            else:
                top = obj
    return top if top is not None else nested


def _notes(obj: dict[str, Any], p: Parsed) -> None:
    n = obj.get("notes")
    if n is None:
        p.warnings.append("notes missing")
        return
    if not isinstance(n, str):
        p.warnings.append(f"notes coerced from {type(n).__name__}")
        n = json.dumps(n, ensure_ascii=False)
    p.notes = n
    if len(n.split()) > 150:
        p.warnings.append(f"notes long ({len(n.split())} words)")


def _no_object(content: str | None, key: str) -> str:
    if not content or not strip_reasoning(content):
        return "the response was empty"
    return f'no JSON object with a "{key}" field was found'


def parse_price(content: str | None, p_max: int = 100) -> Parsed:
    """Price: an integer 0..p_max. Numeric strings and floats with no fractional part are accepted with a
    warning; anything else (other floats, bools, out of range) is an error."""
    obj = extract_object(content, "price")
    if obj is None:
        return Parsed(error=_no_object(content, "price"))
    p = Parsed(obj=obj)
    v = obj["price"]
    if isinstance(v, bool) or v is None:
        p.error = f"price {json.dumps(v)} is not a whole number"
    elif isinstance(v, int):
        p.value = v
    elif isinstance(v, float) and v.is_integer():
        p.value = int(v)
        p.warnings.append(f"price coerced from float {v!r}")
    elif isinstance(v, str) and _INT_STR.match(v):
        p.value = int(float(v))
        p.warnings.append(f"price coerced from string {v!r}")
    else:
        p.error = f"price {json.dumps(v, ensure_ascii=False)} is not a whole number"
    if p.error is None and not 0 <= p.value <= p_max:
        p.error, p.value = f"price {p.value} is outside 0-{p_max}", None
    _notes(obj, p)
    return p


def parse_message(content: str | None, char_limit: int | None = 300) -> Parsed:
    """Chat message: a string (truncated to char_limit with a warning). null, an empty string or "pass"
    mean no message (value None)."""
    obj = extract_object(content, "message")
    if obj is None:
        return Parsed(error=_no_object(content, "message"))
    p = Parsed(obj=obj)
    v = obj["message"]
    if v is not None and not isinstance(v, str):
        if isinstance(v, (dict, list)):
            p.error = 'the "message" field must be a string or null'
            return p
        p.warnings.append(f"message coerced from {type(v).__name__}")
        v = str(v)
    text = v.strip() if v is not None else None
    if text is not None and text.lower() in PASS_WORDS:
        if text:
            p.warnings.append(f"message {text!r} read as a pass")
        text = None
    if text is not None and char_limit is not None and len(text) > char_limit:
        p.warnings.append(f"message truncated from {len(text)} to {char_limit} characters")
        text = text[:char_limit]
    p.value = text
    _notes(obj, p)
    return p
