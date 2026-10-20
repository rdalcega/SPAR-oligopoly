#!/usr/bin/env bash
# Overnight sweep of 2026-10-20 (DECISIONS.md: §6 item 11 update and "sweep budget"). Two runs in parallel, one
# per model, so each run directory has its own lock and budget; the caps add up to $57.
# Qwen3-14B runs in the recorded priority order: human-comparable cells first, then the rest (resume skips
# finished markets, so the last invocation only plays what is still missing).
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-data/runs}
mkdir -p "$LOG"
run() { uv run python -m oligopoly.run --markets-per-cell 6 "$@"; }

(
  run --run-id sweep-gptoss --models gpt-oss-20b --n 2 4 6 8 10 --orderings NT-T T-NT NT-NT \
      --budget-usd 18 --max-concurrent-markets 20 --max-concurrency 48
) > "$LOG/sweep-gptoss.log" 2>&1 &

(
  Q="--run-id sweep-qwen14b --models qwen3-14b --budget-usd 39 --max-concurrent-markets 16 --max-concurrency 40"
  run $Q --n 2 4 6 8 --orderings NT-T;
  run $Q --n 4 --orderings T-NT NT-NT;
  run $Q --n 2 4 6 8 10 --orderings NT-T T-NT NT-NT
) > "$LOG/sweep-qwen14b.log" 2>&1 &

wait
echo "sweep finished"
