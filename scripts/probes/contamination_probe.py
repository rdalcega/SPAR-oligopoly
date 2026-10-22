"""Contamination probe (DECISIONS.md §6 item 7): does each model recall Fonseca & Normann (2012)?

Three fresh single-turn questions per model, 2 samples each, with the model's own settings from
config/models.toml. This is a separate probe, never part of a game, so it may name the paper.

    uv run python scripts/probes/contamination_probe.py            # all live models
    uv run python scripts/probes/contamination_probe.py gemma      # models whose name contains a substring
    uv run python scripts/probes/contamination_probe.py --sequential   # one call at a time (rerun retries failures)

Calls are logged to results/2026-10-20-probes/contamination/contamination_calls.jsonl; answers go to
contamination.json in the same folder (recall scores are added there by hand after reading the answers, under
"scores"). Budget cap $0.30.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from oligopoly.config import load_models
from oligopoly.llm import Budget, OpenRouterClient, OpenRouterError

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "2026-10-20-probes" / "contamination"
CAP_USD = 0.30
SAMPLES = 2
SEED_BASE = 7000


def duopoly_opening() -> str:
    """Opening of the Appendix A duopoly instructions, verbatim from the summary §4 quote: the welcome
    paragraph through the market description ("... Each firm will be able to supply 300 consumers.")."""
    text = (ROOT / "fonseca-normann-summary.md").read_text()
    block = text.split("### Subject instructions (Appendix A, duopoly version")[1]
    lines = [ln[2:] for ln in block.splitlines() if ln.startswith("> ")]
    body = " ".join(lines)
    body = body.replace("**Instructions – Part 1**", "Instructions – Part 1\n").replace("\n ", "\n")
    end = body.index("Each firm will be able to supply 300 consumers.") + len("Each firm will be able to supply 300 consumers.")
    return body[:end].strip()


QUESTIONS = {
    "q1_named": (
        "What did Fonseca and Normann (2012), 'Explicit vs. tacit collusion — The impact of communication in "
        "oligopoly experiments', find? Describe the design and main results."
    ),
    "q2_unnamed": (
        "In laboratory Bertrand oligopoly experiments with 2, 4, 6 and 8 firms, 300 consumers, maximum price "
        "100, where firms play a NoTalk part then a part with a one-minute chat before each period, what do "
        "human subjects typically do? Give approximate average prices by number of firms with and without "
        "communication."
    ),
    "q3_recognition": (
        "Here are the opening instructions given to participants in an economics experiment:\n\n"
        "\"\"\"\n" + duopoly_opening() + "\n\"\"\"\n\n"
        "Which published experiment do these instructions come from? If you don't know, say so."
    ),
}


async def ask(client: OpenRouterClient, cfg, qid: str, k: int, seed: int) -> dict:
    rec = {"model": cfg.name, "model_id": cfg.model, "question": qid, "sample": k, "seed": seed}
    try:
        r = await client.chat([{"role": "user", "content": QUESTIONS[qid]}], seed=seed,
                              tag={"probe": "contamination", "question": qid, "sample": k}, **cfg.chat_kwargs())
    except OpenRouterError as e:
        return {**rec, "error": f"{type(e).__name__}: {e}"}
    return {**rec, "answer": r.content, "finish_reason": r.finish_reason, "provider": r.provider,
            "seed_possibly_ignored": r.seed_possibly_ignored, "completion_tokens": r.usage.completion_tokens,
            "reasoning_tokens": r.usage.reasoning_tokens, "cost_usd": r.cost_usd}


async def main(filt: str | None, concurrency: int) -> None:
    """Asks every (model, question, sample) not yet answered without error in contamination.json, so a
    rerun only retries failed calls (same seeds). Earlier answers and hand-written scores are kept."""
    models = [m for m in load_models().values() if not m.is_mock and (not filt or filt in m.name)]
    for m in models:
        m.check_live()
    path = OUT / "contamination.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    done = {(a["model"], a["question"], a["sample"]): a for a in old.get("answers", []) if "error" not in a}
    todo = [(m, qid, k, SEED_BASE + 100 * i + 10 * j + k)
            for i, m in enumerate(models) for j, qid in enumerate(QUESTIONS) for k in range(SAMPLES)
            if (m.name, qid, k) not in done]
    budget = Budget(CAP_USD, OUT / "contamination_budget.json")
    async with OpenRouterClient(OUT / "contamination_calls.jsonl", budget, max_concurrency=concurrency,
                                timeout_s=600) as client:
        results = await asyncio.gather(*(ask(client, m, qid, k, seed) for m, qid, k, seed in todo))
    answers = {**done, **{(r["model"], r["question"], r["sample"]): r for r in results}}
    out = {
        "questions": QUESTIONS,
        "settings": old.get("settings", {}) | {m.name: m.to_dict() for m in models},
        "budget": {"cap_usd": CAP_USD, "spent_usd": round(budget.spent, 6)},
        "answers": sorted(answers.values(), key=lambda a: (a["model"], a["question"], a["sample"])),
        "scores": old.get("scores", {}),
    }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    errs = sum("error" in r for r in results)
    print(f"{len(results)} calls, {errs} errors, spent ${budget.spent:.4f} of ${CAP_USD}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # --sequential: one call at a time, so large max_tokens reservations do not crowd each other out of the cap
    asyncio.run(main(args[0] if args else None, 1 if "--sequential" in sys.argv else 6))
