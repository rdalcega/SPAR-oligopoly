"""Analyse one or more runs and write tidy CSV/JSON tables plus summary.json.

    uv run python scripts/analyze.py --run-id mockdev --out data/analysis/mockdev/
"""

from __future__ import annotations

import argparse
import sys

from oligopoly.analysis.report import build, excluded_cells_from_config, write
from oligopoly.config import ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", action="append", required=True, help="repeatable")
    ap.add_argument("--out", required=True, help="output directory, e.g. data/analysis/<name>/")
    ap.add_argument("--runs-root", default=str(ROOT / "data" / "runs"))
    ap.add_argument("--models-file", default=str(ROOT / "config" / "models.toml"),
                    help="its exclude_cells are left out of the analysis and reported as excluded")
    a = ap.parse_args(argv)
    tables, summary = build(a.run_id, a.runs_root, excluded_cells_from_config(a.models_file))
    for r in summary["meta"]["excluded_cells"]:
        print(f"excluded cell {r['model']} n={r['n']} {r['ordering']}: {r['markets_dropped']} completed markets left out")
    files = write(a.out, tables, summary)
    print(f"wrote {len(files)} files to {a.out}: {', '.join(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
