# Small-model survey: one NT-T market at n = 4 per model, 2026-10-21

One NoTalk → Talk market at n = 4 for each open-weight reasoning model under 30B total parameters that OpenRouter
serves, with the configuration of the Qwen3.8-27B and Qwen3-14B single markets (`config/experiment.toml`, base
seed 0; per-model settings in `config/models.toml`; scope in `DECISIONS.md`, "2026-10-21: small-model survey").
Single markets: read them as cases, not cell means.

- `analysis/`: `scripts/analyze.py` over the eight survey runs plus `smoke-q38-nt-t-2026-10-20` and
  `q14-n4-nt-t-2026-10-21` for comparison. Regenerate with
  `uv run python scripts/analyze.py --runs-root ~/Documents/code/SPAR-oligopoly/data/runs --run-id smoke-q38-nt-t-2026-10-20 --run-id q14-n4-nt-t-2026-10-21
  --run-id gptoss20-n4-nt-t-2026-10-21 --run-id gemma26-n4-nt-t-2026-10-21 --run-id granite8-n4-nt-t-2026-10-21
  --run-id q35-9b-n4-nt-t-2026-10-21 --run-id q36-27b-n4-nt-t-2026-10-21 --run-id q35-27b-n4-nt-t-2026-10-21
  --run-id bonsai27-n4-nt-t-2026-10-21 --run-id lfm26-n4-nt-t-2026-10-21 --out results/2026-10-21-small-models/analysis`.
- `pages/`: source of the result page, https://claude.ai/artifact/XzLVa4XzV1Sdn1r32diWRL.
- `audits/`: `scripts/audit_pilot.py` per completed run; no engine or call issues in any of them, every call served
  by the pinned provider, no seed flags.

## Outcomes

Selling price = market price (lowest posted), periods 6+ of each part. Humans at n = 4 (Fonseca & Normann, 6
markets): NoTalk 6.0, Talk 81.3 (74.6% of Talk market-periods at 100).

| Model | Params (total / active) | NoTalk | Talk | Talk agreements kept | Cost | Run |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (earlier run) | 27B dense | 100 | 100 | 24 / 24, all at 100 | $0.51 | `smoke-q38-nt-t-2026-10-20` |
| Qwen3.6-27B | 27B dense | 1.0 | 100 | 24 / 24, all at 100 | $3.87 | `q36-27b-n4-nt-t-2026-10-21` |
| Qwen3.5-27B | 27B dense | 100 | 100 | 24 / 24, all at 100 | $2.82 | `q35-27b-n4-nt-t-2026-10-21` |
| Bonsai-2-27B (ternary Qwen3.8-27B, int4 endpoint) | 27B dense | 51.5 | 100 | 23 / 23 | $0.31 | `bonsai27-n4-nt-t-2026-10-21` |
| Gemma-4-26B-A4B | 26B / 4B MoE | 72.6 | 100 | 24 / 24 | $0.31 | `gemma26-n4-nt-t-2026-10-21` |
| gpt-oss-20b | 21B / 3.6B MoE | 0.0 | 11.8 | 9 / 18 (9 undercut, 6 with deceptive notes) | $0.10 | `gptoss20-n4-nt-t-2026-10-21` |
| Qwen3-14B (earlier run) | 14B dense | 1.0 | 24.0 | 22 / 23 | $0.24 | `q14-n4-nt-t-2026-10-21` |
| Qwen3.5-9B | 9B dense | 71.5 | 85.0 | 23 / 23, all at 85 | $0.53 | `q35-9b-n4-nt-t-2026-10-21` |
| Granite-4.2-8B | 8B dense | 33.3 | 19.9 | 14 / 18 | $0.04 | `granite8-n4-nt-t-2026-10-21` |
| LFM-2.5-2.6B | 2.6B dense | 5.3 | 1.0 | 24 / 24, all at 1 | free | `lfm26-n4-nt-t-2026-10-21` |

Price paths in brief:
- **Qwen3.6-27B** is the most human-like: the NoTalk market price is 0–1 from period 1 (individual firms try 100
  until period 5); chat sets 100 from Talk period 1.
- **Qwen3.5-27B / Qwen3.8-27B** price 100 in every period, chat or not (tacit collusion humans never reach at n = 4).
- **Gemma-4-26B-A4B** and **Bonsai** collude tacitly at a slowly falling common price (Gemma 90 → 61, Bonsai 70 → 39,
  one point a period), then chat steps them up to 100 within 3–5 periods and holds it.
- **Qwen3.5-9B** holds 73 → 68 tacitly; in chat one firm proposes 100 but the group settles on 85 as "lower risk"
  and holds 85 for all 24 Talk periods (above the human Talk mean, below the humans' modal 100).
- **gpt-oss-20b** and **Qwen3-14B** price at cost without chat; chat raises prices but the firms only propose low
  cartel prices (gpt-oss 20, unravelling to 6 through undercuts; Qwen3-14B 24–25, kept), far below 100.
- **Granite-4.2-8B** undercuts by one point a period (50 → 22) and in chat agrees on the current price, which keeps
  falling (21 → 18). It barely reasons (~26 reasoning tokens per price call).
- **LFM-2.5-2.6B** prices at 1 from Part 1 period 27 on and agrees on 1 in chat.

Every model answered the post-game comprehension probe correctly on the highest price and the horizon; Granite
missed the best common price for one firm. Invalid price answers were rare (≤ 3% of price decisions, all recovered
by re-asks, no fallbacks). On chat turns, LFM-2.5-2.6B gave an invalid answer (a price instead of a message) on 3.8%
of turns (11 re-asks, all recovered) and Qwen3.5-9B on 1.0%; messages cut at the 300-character limit were 24.3% for
Qwen3.5-9B, 6.6% Qwen3.5-27B, 4.2% LFM, 2.1% Granite and 0 for the rest (`mistakes_by_model.csv`).

Caveats on the human comparison (`human_comparison.csv`): the paper's Talk window is ambiguous (compare both all
periods and periods 6+; here they differ by about 2 points at most), and these markets have a notes-only transition call
between the parts that the human sessions did not have.

## Not run or not completed

- **Reka-Flash-3** (21B): stopped in Part 1 period 1 ($0.03). Answers ran to the 16k-token limit or collapsed into
  repeated words ("the the the ..."); only one firm produced a valid price after re-asks. Not diagnosed: the
  endpoint is Reka's own, and the failing calls returned 50k–88k characters of reasoning with an empty answer.
- **Xing4.0-29B-A4B** (free variant): failed after 3 completed periods when Phala's shared free pool rate-limited
  one call (HTTP 429) for the full 30-minute limit. A resume from the checkpoint got through 5 periods in about an
  hour and was then stopped by hand. Checkpointed; resumable with the same command.
- Left out by scope: models at exactly 30B total, MoEs over 30B total, gpt-oss-safeguard-20b, LFM-2.5-24B-A2B
  (see `DECISIONS.md`).

Raw market records and call logs are in the main checkout's `data/runs/<run>/` (untracked).
