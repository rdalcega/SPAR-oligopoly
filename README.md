# SPAR-oligopoly

LLMs play the firms in repeated Bertrand markets modelled on Fonseca & Normann (2012), "Explicit vs. tacit
collusion — The impact of communication in oligopoly experiments". Firms post integer prices from 0 to 100, and
the lowest price takes the market. A NoTalk part is followed by a surprise Talk part with free-form chat, so
collusion can be measured both tacitly and through talk, and compared with the paper's human markets. The repo
is part of the SPAR project *Learning dynamics in competitive multiagent environments*.

**Status (2026-10-22): paused.** The latest results and a two-page summary are in
[`write-up/mini-paper.pdf`](write-up/mini-paper.pdf).

## Where things stand

- **Latest setting:** the "summaries" prompt layout. Each request shows the last 3 period summaries, the firm's
  own private notes and the current chat. A notes-only transition call runs between the parts.
- **Small-model survey (2026-10-21):** one NoTalk → Talk market at n = 4 for each open reasoning model under 30B
  total parameters. Outcomes ranged widely:
  - Qwen3.5/3.6/3.8-27B, Gemma-4-26B-A4B and Bonsai-2-27B reached 100 with chat.
  - gpt-oss-20b, Qwen3-14B, Granite-4.2-8B and LFM-2.5-2.6B stayed far below humans even with chat.
- **Qwen3.5-9B replication (2026-10-22):**
  - At n = 2 it is at human level (Talk 94.8 vs 93.5 for humans).
  - At n = 4 it falls short (68.3 vs 81.3 over 7 markets). Chat mostly ratifies the price the firms reached tacitly.
- **Earlier work:** pilots, provider and contamination probes, a 152-market sweep of gpt-oss-20b and Qwen3-14B
  (2026-10-20) and Qwen3-14B prompt variants. Except for the variants, these used earlier prompt layouts. Each
  `results/` README says which.

## Layout

| Path | Contents |
|---|---|
| `fonseca-normann-summary.md` | The spec: the paper's design (§1–5), open decisions for an LLM run (§6) and the human benchmarks (§7) |
| `DECISIONS.md` | Dated log of the spec sign-off and every design choice, held fixed across models |
| `config/` | `experiment.toml` (design settings) and `models.toml` (model IDs, pinned providers, sampling) |
| `src/oligopoly/` | Engine (`engine.py`, `firm.py`, `market.py`), prompts (`prompts.py`), OpenRouter client with logging and cost caps (`llm.py`), parsing, runner (`run.py`), human benchmarks (`benchmarks.py`) |
| `src/oligopoly/analysis/` | Market tables, chat agreement coding, mistake flags, statistics and human comparison (driven by `scripts/analyze.py`) |
| `scripts/` | `analyze.py`, `audit_pilot.py` (run auditor), `show_market.py` (transcript viewer), `project_context.py` / `project_cost.py` (pre-run projections). `probes/` holds the 2026-10-20 one-off probes; `archive/` holds the frozen dashboard builder |
| `results/<date>-<name>/` | Tracked outputs per experiment: `README.md`, `analysis/`, `audits/`, `pages/` (sources of result pages) |
| `docs/` | `prompts.md` (every text a firm sees) and `related-work.md` (LLM collusion literature) |
| `write-up/` | The mini-paper (`mini-paper.tex`, `.pdf`) and `make_figures.py` |
| `tests/` | Offline tests: a mock client and mocked HTTP, so no paid calls |
| `data/` | Untracked scratch: raw run records and call logs (`data/runs/`), regenerated analyses and audits |

Raw records of every run (~2 GB) are in the main checkout's `data/runs/`. From a worktree, pass
`--runs-root ~/Documents/code/SPAR-oligopoly/data/runs` to the analysis scripts.

## Usage

```sh
uv sync
cp .env.example .env            # add OPENROUTER_API_KEY
uv run pytest -q                # offline test suite

# Dry run: list the cells and estimate cost, with no calls
uv run python -m oligopoly.run --run-id test --models qwen3.5-9b --n 4 --orderings NT-T --dry-run

# Live run: needs a cost cap (and, per CLAUDE.md, explicit approval)
uv run python -m oligopoly.run --run-id my-run --models qwen3.5-9b --n 4 --orderings NT-T \
    --markets-per-cell 1 --budget-usd 1

uv run python scripts/analyze.py --run-id my-run --out data/analysis/my-run
uv run python scripts/audit_pilot.py my-run

# Rebuild the mini-paper
uv run --group writeup python write-up/make_figures.py && (cd write-up && tectonic mini-paper.tex)
```

The runner defaults to the paper's cells (n ∈ {2, 4, 6, 8} × NoTalk → Talk, Talk → NoTalk, plus NoTalk → NoTalk
at n = 4). `--grid` plays every `--n` × `--orderings` pair as an extension. Every call is logged with the model
ID, the serving provider, the sampling settings and the seed.
