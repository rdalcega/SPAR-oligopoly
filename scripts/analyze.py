"""Analyse one or more runs and write tidy CSV/JSON tables plus summary.json.

    uv run python scripts/analyze.py --run-id mockdev --out data/analysis/mockdev/
"""

from __future__ import annotations

import argparse
import sys

from oligopoly.analysis.report import build, write
from oligopoly.config import ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", action="append", required=True, help="repeatable")
    ap.add_argument("--out", required=True, help="output directory, e.g. data/analysis/<name>/")
    ap.add_argument("--runs-root", default=str(ROOT / "data" / "runs"))
    a = ap.parse_args(argv)
    tables, summary = build(a.run_id, a.runs_root)
    files = write(a.out, tables, summary)
    print(f"wrote {len(files)} files to {a.out}: {', '.join(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
