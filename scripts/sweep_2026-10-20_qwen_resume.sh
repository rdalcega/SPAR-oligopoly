#!/usr/bin/env bash
# Resume of the 2026-10-20 Qwen3-14B sweep with more markets in parallel (32 instead of 16), after a network
# outage. Same run id, cap and priority order as scripts/sweep_2026-10-20.sh; finished markets are skipped and
# failed ones resume from their checkpoints.
set -u
cd "$(dirname "$0")/.."
Q="--markets-per-cell 6 --run-id sweep-qwen14b --models qwen3-14b --budget-usd 39 --max-concurrent-markets 32 --max-concurrency 80"
{
  uv run python -m oligopoly.run $Q --n 2 4 6 8 --orderings NT-T
  uv run python -m oligopoly.run $Q --n 4 --orderings T-NT NT-NT
  uv run python -m oligopoly.run $Q --n 2 4 6 8 10 --orderings NT-T T-NT NT-NT
} >> data/runs/sweep-qwen14b.log 2>&1
echo "qwen resume finished"
