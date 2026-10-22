# Qwen3-14B prompt variants, 2026-10-21

One n = 4 NT-T market per prompt variant plus a fresh baseline (DECISIONS.md, "Prompt variants"), `--seed 1`,
code at commit 6089b6f. Variants are an extension: these markets are never compared with the human results.
Single markets, so read them as cases, not cell means.

- `analysis/<variant>/`: `scripts/analyze.py`, one run at a time (the uniform-design check refuses to pool
  different prompt wordings). Regenerate with
  `uv run python scripts/analyze.py --runs-root ~/Documents/code/SPAR-oligopoly/data/runs
  --run-id q14-v-<variant>-2026-10-21 --out results/2026-10-21-variants/analysis/<variant>`.
- `audits/<variant>.txt` and `audits/q14-v-<variant>-2026-10-21_audit.json`: `scripts/audit_pilot.py` (printed summary
  and full report; the JSON reports were copied from the `q14-prompt-variants` worktree before it was removed). No
  engine or call issues; every call was served by the pinned `deepinfra/fp8` endpoint, finish reason `stop`, no seed
  flags.
- `pages/q14-prompt-variants.html`: source of the result page (link below).
- Raw records: `data/runs/q14-v-<variant>-2026-10-21/` in the main checkout (untracked). The first
  `identical_copies` start was stopped after 20 calls to fix its wording
  (`q14-v-identical_copies-2026-10-21.aborted-v1-wording`, $0.01).

Mean selling price (Part 1 = NoTalk, 29 periods; Part 2 = Talk, 24 periods):

| Variant | P1 all | P1 from 6 | P2 all | P2 last 5 | Chat agreements (kept) | Cost |
|---|---|---|---|---|---|---|
| baseline | 0.9 | 1.0 | 1.0 | 1.0 | 23 (20) | $0.23 |
| profit_formula | 1.0 | 1.0 | 1.4 | 1.0 | 22 (15) | $0.26 |
| similar_reasoning | 1.0 | 1.0 | 2.0 | 2.0 | 24 (24) | $0.21 |
| identical_copies | 0.9 | 1.0 | 1.0 | 1.0 | 22 (21) | $0.22 |
| future_consequences | 58.9 | 55.9 | 44.1 | 44.0 | 21 (19) | $0.28 |
| rivals_choices | 1.0 | 1.0 | 2.0 | 2.0 | 24 (24) | $0.21 |

Total $1.42 of the $3 cap. Result page (source `pages/q14-prompt-variants.html`): https://claude.ai/artifact/7USZvwycUwa8ZpXqCcjnRM

Reading:

- Only `future_consequences` moved prices. Part 1 opened at 75/99/99/99 and every firm undercut by one each period
  (all four cut to the same price, so they kept tying): 75 → 44 over 29 periods. The notes show undercutting
  intent ("undercut to 71 and capture the entire market"), not tacit collusion. The difference from the other
  markets is the opening: there no firm opened at 0 or 1. In every other market some firm opened at 0 or 1 in
  period 1 (often citing the Bertrand Nash equilibrium) and prices sat at 1 from period 2. With chat the market
  agreed on 45–50, then held 44 for 22 periods.
- With chat, Qwen3-14B agrees in almost every period and mostly keeps the agreement, but it agrees on the price
  Part 1 ended at (1–2, or 44). In the post-game probe every firm in every market answers that the best common
  price is 100, so the low cartels are not a misunderstanding of the payoffs; `profit_formula` changes nothing.
- `identical_copies`: firms read "identical" as symmetric Bertrand competition ("the Nash equilibrium here is for
  all firms to set the lowest possible price (0 ECU)") and never mention being the same model in chat, though
  the variant allows it.
- The fresh baseline never left 1 in Part 2, unlike the 24 cartel of `q14-n4-nt-t-2026-10-21`: one market per
  prompt can't separate a variant's effect from run-to-run noise, and the `future_consequences` result rests on
  one Part 1 opening.
- `future_consequences` is the least neutral of the five: it points firms at repeated-game reasoning.
