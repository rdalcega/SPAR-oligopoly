# Sweep of 2026-10-20

Analysis outputs of the overnight sweep (design in `DECISIONS.md`), copied from `data/analysis/` so they are
tracked. Regenerate with `uv run python scripts/analyze.py --run-id <run> --out data/analysis/<run>`.

- `sweep-gptoss/`: gpt-oss-20b, all 90 markets (5 sizes × 3 orderings × 6), $15.25.
- `sweep-qwen14b/`: Qwen3-14B, 62 of 90 markets, $39.00 (hit its cap). Missing: n6 T-NT 1/6, n8 T-NT 0/6,
  n10 NT-T 0/6, n10 T-NT 0/6, n10 NT-NT 1/6. The 28 unfinished markets are checkpointed in `data/runs/`.
- `reading_workflow.json`: qualitative reading of the transcripts by Claude readers, each claim checked by an
  independent GPT verifier (confirmed / partly, with corrections in the verdict note). Qwen3-14B's covers its 36
  human-comparable markets only.
- Raw market records and call logs (~1.5 GB) are in `data/runs/sweep-*` (untracked).
- Results page: https://claude.ai/artifact/E1dyScQJDmnVU9njU6DfqF
