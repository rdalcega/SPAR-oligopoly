# Pilots of 2026-10-20

Pilot markets run on 2026-10-20 before the sweep, to test the engine and prompts and to measure tokens and cost for
the projections (`DECISIONS.md`). Each ran under a design that was later replaced, so none of them is comparable
with the summaries-layout runs of 2026-10-21 and later; they are tests, not results. Only the audits are tracked.

| Run | Models | Markets | Design | Spent |
|---|---|---|---|---|
| `pilot1`, `pilot1-q14` | the 7-model lineup; Qwen3-14B | n = 4 T-NT, one each; Qwen3.5-9B failed before period 1 | full history, notes on every decision, short parts (6 + 4 periods) | $1.41, $0.09 |
| `pilot2-full`, `pilot2-full10` | gpt-oss-20b | n = 2 NT-T; n = 10 T-NT | full history, full-length parts (29 + 24) | $0.13, $2.40 |
| `pilot3-n2`, `pilot3-n10` | gpt-oss-20b, Qwen3-14B | n = 2 NT-T (full length); n = 10 T-NT (6 + 4) | verbatim 3-period rolling window, post-game probe | $0.20, $0.35 |
| `ab-w3`, `ab-w60` | gpt-oss-20b, Qwen3-14B | n = 2 NT-T, full length | memory-window A/B: window 3 vs 60 | $0.88, $1.80 |

What they led to (details in `DECISIONS.md`):

- Pilot 1: Qwen3.5-9B dropped for DeepInfra rate limits; gpt-oss-20b chosen as the one model; the chat-round wording
  after price answers on chat turns (§6 item 1); findings on the chat coder's number parsing ("7,500" read as 7)
  and on retries under rate limits (see `audits/pilot1.md`).
- The rolling window and self-written notes replaced full history (§6 item 3); `pilot3-*` are the "new-design
  pilots" behind the sweep's cost projections, and `pilot3-n10` is the short n = 10 pilot with no forced chat
  passes after the chat-turn wording (§6 item 8).
- `ab-w3` / `ab-w60`: the memory-window A/B check (§6 item 8, "memory-window A/B check"); its config
  (`config/experiment_ab_fullmem.toml`) was removed in the 2026-10-22 cleanup.

## Files

- `audits/pilot1.md`: read-only audit of pilot 1 (engine invariants, call log, findings by severity, behaviour per
  model, tokens and cost); `audits/pilot1_audit.json` is its machine-readable output from `scripts/audit_pilot.py`.
- `audits/pilot3-n2+pilot3-n10_audit.json`: `scripts/audit_pilot.py` on the two `pilot3-*` runs. No engine issues.
  The only flags are 44 "future/same-round message ... visible" hits from the substring leak check in
  `pilot3-n2/gpt-oss-20b__n2__NT-T__m0`, all in Part 2 where both firms post near-identical short sentences every
  round ("We keep price 1 for this period and onward."). They were not investigated further and are not mentioned
  in `DECISIONS.md`; treat them as unresolved rather than as confirmed false positives.

Raw market records and call logs are in the main checkout's `data/runs/<run>/` (untracked).
