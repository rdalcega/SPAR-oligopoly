# Qwen3.8-27B and Qwen3-14B single markets, 2026-10-20/21

Smoke test of the refactored environment and a first look at two Qwen models, one NT-T market per cell (design in
`DECISIONS.md`). Single markets, not cell means: read them as cases.

- `analysis/`: `scripts/analyze.py` over the three runs below (analysis coders as of branch
  `worktree-fix-analysis-coders`). Regenerate with
  `uv run python scripts/analyze.py --run-id smoke-q38-nt-t-2026-10-20 --run-id q38-n268-nt-t-2026-10-20
  --run-id q14-n4-nt-t-2026-10-21 --out results/2026-10-21/analysis`.
- `audits/`: `scripts/audit_pilot.py` per run; no engine or call issues in any of them.
- `pages/`: sources of the two result pages (links below).

| Run | Model | Markets | Real cost | Outcome |
|---|---|---|---|---|
| `smoke-q38-nt-t-2026-10-20` | qwen3.8-27b | n = 4 | $0.51 | 100 in all 53 periods |
| `q38-n268-nt-t-2026-10-20` | qwen3.8-27b | n = 2, 6, 8 | $3.13 | 100 throughout at n = 2, 6; n = 8 holds 100 for 14 periods, a defection (from a misread horizon) slides it to 81, chat rebuilds 100 |
| `q14-n4-nt-t-2026-10-21` | qwen3-14b | n = 4 | $0.24 | 1 without chat, a cartel at 24 with chat; below the gate (Talk price ≥ 50), so n = 2, 6, 8 were not run |

The q38-n268 run was interrupted by the Mac sleeping (~35 min) and resumed from checkpoints; 11 in-flight calls
were charged their reservation ($0.35, in `budget.json`).

Raw market records and call logs are in `data/runs/<run>/` (untracked; copied on 2026-10-21 from the
`experiments` worktree, which still holds the originals).

Pages:
- Qwen3.8-27B report: https://claude.ai/artifact/ET2twcbviJwmuTUbMNg6Df
- Qwen3-14B vs Qwen3.8-27B: https://claude.ai/artifact/5S2cWC1P4Vr1uUjPdEYPwB
