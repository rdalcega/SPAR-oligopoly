# Pre-run probes, 2026-10-19/20

Three separate probes run before the sweep; none is part of a game. The scripts are in `scripts/probes/` and now
write to the folders below (they were run when outputs went to `data/probes/`). Experiments are paused, and the
provider probe's requests use the old append-only layout, so a rerun would not reflect current requests.

## `provider/`: providers and sampling (`DECISIONS.md` "Providers and sampling", §6 item 9)

Characterised each lineup model on its OpenRouter endpoints: reasoning tokens at each effort, whether reasoning
can be disabled, JSON validity, latency, cost per price call, same-seed reproducibility, caching, rate limits and
a ~60k-token long-context call. Outcome: all 7 lineup models pinned to DeepInfra (seed supported) with model-card
sampling (`config/models.toml`); LFM-2.5-24B-A2B dropped (its only provider caps output at 8,192 tokens and its
reasoning cannot be disabled or shortened); Qwen3.5-9B on DeepInfra already showed heavy HTTP 429s and ~220 s calls.

- `scripts/probes/probe_models.py`: `probe_calls.jsonl` (every attempt), `probe_results_{primary,alt,long}.json`,
  `run_{primary,alt,long}.log`, `budget.json` ($0.25 of a shared $3 cap).
- `scripts/probes/probe_endpoints.py`: `endpoints.json`, a snapshot of the probed models' endpoints (pricing,
  quantisation, context length, supported parameters; no cost).
- `scripts/probes/probe_summarise.py`: `summary_table.md`, the per-model, per-provider table with sample chat
  messages.
- `qwen35_*`: an ad-hoc recheck of Qwen3.5-9B on DeepInfra (no script kept): 7 attempts, all HTTP 429, $0.

## `contamination/`: does each model know the paper? (`DECISIONS.md` §6 item 7)

`scripts/probes/contamination_probe.py`: three fresh questions per model, two samples each (what the paper found;
what humans do in this design, unnamed; which experiment the Appendix A instructions come from). Recall scored
0–3 by hand after reading the answers (in `contamination.json` → `scores`, scale in `scoring`). Result: no model
recognises the instructions or recalls the design or results; only Qwen3.5-27B and Qwen3.6-27B know bibliographic
details (recall 1), the other five score 0. Contamination risk from memorised results is low for all 7.
$0.22 of a $0.30 cap.

## `structured-outputs/`: structured outputs on the pinned endpoints (`DECISIONS.md` §6 item 8)

`scripts/probes/probe_structured.py`: 2 calls each for gpt-oss-20b and Qwen3-14B on a realistic Part 2 chat-turn
request with a strict json_schema and `require_parameters`. DeepInfra accepted the schema for both and all 4
answers were valid JSON matching it ($0.003). Structured outputs nevertheless stay off: the chat-turn wording and
re-ask fixed the forced passes without them. The schema used here is the earlier `chat_turn` with `notes` and
`message`; the script now sends the current message-only schema.
