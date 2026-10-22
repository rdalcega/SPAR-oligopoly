# Sweep of 2026-10-20

Analysis outputs of the overnight sweep (design in `DECISIONS.md`), copied from `data/analysis/` so they are
tracked: per model, 9 of the files `scripts/analyze.py` writes plus `reading_workflow.json`.

**Older design and older code.** These markets ran under the verbatim rolling-window prompt layout (the last 3
periods shown verbatim, notes on every decision, no transition call), which `DECISIONS.md` §6 item 3 replaced the
same day with the summaries layout. They are not comparable with the summaries-layout runs of 2026-10-21 and later.
The outputs were made with the analysis code at commit b960aa3, before the later coder fixes (merged in 07eb925),
so a rerun with current code would shift some rates slightly (tie_full and notes_contradict; no agreed price or
kept/broken verdict changes, per `DECISIONS.md` "Analysis — heuristic definitions") and write more files.
To regenerate with current code:
`uv run python scripts/analyze.py --runs-root ~/Documents/code/SPAR-oligopoly/data/runs --run-id <run>
--out results/2026-10-20-sweep/<run>`.

- `sweep-gptoss/`: gpt-oss-20b, all 90 markets (5 sizes × 3 orderings × 6), $15.25.
- `sweep-qwen14b/`: Qwen3-14B, 62 of 90 markets, $39.00 (hit its cap). Missing: n6 T-NT 1/6, n8 T-NT 0/6,
  n10 NT-T 0/6, n10 T-NT 0/6, n10 NT-NT 1/6. The 28 unfinished markets are checkpointed in the raw run.
- `reading_workflow.json` (in each): qualitative reading of the transcripts by Claude readers, each claim checked
  by an independent GPT verifier (confirmed / partly, with corrections in the verdict note). Qwen3-14B's covers its
  36 human-comparable markets only.
- `audits/`: `scripts/audit_pilot.py` on `sweep-gptoss` (text summary and JSON report).
- `dashboard/`: datasets and page source (`index.html`) of the results page below, built by
  `scripts/archive/build_artifact_data.py` (frozen); `dashboard/manual/` holds its hand-maintained inputs.
- Raw market records and call logs (~1.5 GB) are in the main checkout's `data/runs/sweep-*` (untracked).
- Results page: https://claude.ai/artifact/E1dyScQJDmnVU9njU6DfqF
