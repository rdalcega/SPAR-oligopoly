# Decisions

Record of sign-offs and design choices for the LLM replication of Fonseca & Normann (2012). Choices here are
held fixed across models. Add an entry before implementing the thing it decides.

## Sign-off of the spec

**2026-10-19 — `fonseca-normann-summary.md` signed off against the working paper (DICE DP 65).**

- Checked by: Rodrigo Delgado, with Claude Code doing the line-by-line comparison (commit 1987cc4).
- Scope: every number, quotation and design statement in §1–7 was compared with the DICE Discussion Paper 65
  PDF. The transcribed instructions match the paper word for word.
- Outstanding: the journal version (*European Economic Review* 56(8), 2012) was not available and has not
  been checked. The summary lists the points to confirm there: Table 1 cell counts, the results-screen
  feedback, the Part 2 analysis window, the wording of footnote 14, the Table 4 state labels, and the period
  count for eight-firm market 4. If the journal version changes any of these, amend the summary and add an
  entry below.

## Design choices (§6 of the summary)

Decided 2026-10-19 by Rodrigo Delgado in an interview with Claude Code. Items not listed are still open.

### Scope (extension beyond the paper)

- ~~**Market sizes:** n ∈ {2, 4, 6, 8, 10}.~~ *(superseded 2026-10-20, below)* n = 10 is an extension; at δ = 5/6 full collusion is an equilibrium for
  n ≤ 6 (n = 6 only marginally) and not for n = 8, 10.
- ~~**Orderings:** all three (NoTalk→Talk, Talk→NoTalk, NoTalk→NoTalk) at every n.~~ *(superseded 2026-10-20, below)* Only NoTalk→Talk at n ∈
  {2, 4, 6, 8} and the two controls at n = 4 are compared directly with the human results; the rest are
  reported separately as extensions.
- **2026-10-20 update (decided by Rodrigo, after the first sweep): the paper's cells only.** n = 10 is dropped,
  and T-NT and NT-NT run only at n = 4, as in the paper: NoTalk→Talk at n ∈ {2, 4, 6, 8} plus both controls at
  n = 4 (six cells, 36 markets per model at 6 per cell). This is the default of further full sweeps:
  `oligopoly.run` plays these cells (`config.PAPER_CELLS`), `--n`/`--orderings` only narrow them (so a resume
  never adds extension cells), and the cost and context projections use the same default. Extension cells
  need an explicit `--grid`, which plays every `--n` × `--orderings` pair. The n = 10 and
  off-paper control markets of the 2026-10-20 sweep are kept and still reported as extensions.

### Models (§6 item 10, plus model choice)

- Small open-weight reasoning models, total parameters < 30B (so they could be LoRA-fine-tuned on a budget),
  across several families. One model per market (homogeneous groups).
- Lineup (8): `qwen/qwen3.8-27b`, `qwen/qwen3.6-27b`, `qwen/qwen3.5-27b`, `qwen/qwen3.5-9b`,
  `google/gemma-4-26b-a4b-it`, `openai/gpt-oss-20b`, `ibm-granite/granite-4.2-8b`, `liquid/lfm-2.5-24b-a2b`.
  This gives one model per lab, a Qwen size axis (3.5-9B vs 3.5-27B) and a Qwen generation axis at 27B
  (3.5, 3.6, 3.8).
- **2026-10-20: LFM-2.5-24B-A2B dropped** (7 models remain). Its only provider (Liquid) caps output at 8,192
  tokens, thinking cannot be disabled or shortened, and ~18% of probe calls truncated with empty answers.
- **2026-10-20: Qwen3-14B added** (8 models), to strengthen the small end. Pinned to DeepInfra fp8 (seed
  supported), Qwen3 thinking-mode sampling (T 0.6, top_p 0.95, top_k 20). Its context is 40,960 tokens
  (input + output), which may not hold full Talk histories at larger n.
- **2026-10-20: context-limit exclusions** (decided by Rodrigo). Where a model's context limit is a problem or
  nearly one, that model skips the cell; no history truncation or summarising. Rule, for every model: a model
  runs a cell only if the cell's projected peak prompt (measured from pilots) plus the model's `max_tokens` is
  at most 80% of its pinned endpoint's context length. Excluded cells are listed in `config/models.toml`
  and reported as missing, not imputed. A runtime guard also stops any market whose prompt would overflow and
  records it as a failure.
  Implementation: the guard bounds prompt tokens as content bytes / `bytes_per_token_min` + 8 per message + 256,
  checked with max_tokens against the pinned endpoint's `context_length` before every call (failure reason
  `context_overflow`, nothing sent). `bytes_per_token_min` is 2.5 (2.4 for Qwen3.8-27B), below the lowest
  marginal ratio seen in the pilots (2.69–3.3 bytes/token on results text; whole prompts 3.2–3.8); on pilot
  calls actual prompt tokens were 64–72% of the bound. `scripts/project_context.py` projects each cell's peak
  with the same bound and writes `exclude_cells`.
  **2026-10-20, with the rolling window (§6 item 3):** re-projected at full part lengths (3 replays per cell):
  peak bound + max_tokens is at most 23.3k tokens for gpt-oss-20b (limit 104.9k) and 20.6k for qwen3-14b
  (limit 32.8k), both at n = 10 Talk. No cell is excluded for either model; qwen3-14b's earlier exclusions
  (all Talk cells) were removed.
- **Providers and sampling** (from the probe in `data/probes/`): all 7 pinned to DeepInfra (bf16 or fp8, seed
  supported), so provider isn't confounded with model. Sampling per model card, in `config/models.toml`.
- **2026-10-20: Qwen3.5-9B dropped.** DeepInfra's shared pool rate-limited it (HTTP 429) so badly that its
  pilot market failed before period 1, and calls took ~220 s when they worked.
- **2026-10-20: one model only — `openai/gpt-oss-20b`** (decided by Rodrigo; this supersedes the lineup above
  for the experiment). Running many models complicated the work too much. gpt-oss-20b (21B MoE, 3.6B active)
  was chosen for being LoRA-able on a budget, the cheapest and fastest in the pilots, and the richest in
  strategic behaviour there (agreements made and broken, threats, a price war and recovery). The other models
  stay in `config/models.toml` for possible later use but are not part of the experiment. Pinned to
  `deepinfra/bf16`, T 1.0, top_p 1.0, reasoning effort medium. `max_tokens` lowered from 8,000 to 6,000 (largest
  pilot output 4,200 tokens) so that every cell, including T-NT n = 10, passes the context rule; no cells are
  excluded. With one model, the "LLMs vs LLMs" goal is deferred.

### §6 item 1 — chat structure

Simultaneous rounds. Each Talk period has up to 3 chat rounds. In a round every firm writes one message or
passes without seeing that round's other messages; then all messages of the round are shown to everyone. The
chat closes early when every firm passes in a round. Messages are capped at 300 characters (longer messages are
truncated and the truncation is logged).

**2026-10-20, after pilot 1 (prompt wording, not structure):** all 15 invalid outputs in pilot 1 were price
JSON sent to a chat prompt, mostly in round 3. The chat line now names the round ("chat round 2 of at most 3")
and, on the last possible round, adds "(last round; prices are set after it)"; the price prompt after a chat
says "The chat box has closed for this period: no more messages can be posted." Both restate timing already
in the instructions; nothing about how to behave. Listed as edit 28 in `docs/prompts-draft.md`.

**2026-10-20, after the n = 10 pilot (wording):** ~19% of chat turns ended as forced passes because gpt-oss
answered chat turns with a price JSON, and the generic re-ask often failed again. Every chat prompt now states
"This is a chat turn: prices cannot be set now; you will set your price after the chat closes.", and a chat
answer holding a price object (and no message object) is re-asked with "You answered with a price, but this is
a chat turn. Prices are set after the chat closes. Post a message, or null to post nothing." plus the format,
and recorded with its own error string. Other invalid answers keep the generic re-ask. Edit 30 in
`docs/prompts-draft.md`.

### §6 item 2 — simultaneous pricing

All firms' prices for a period are collected independently before any is revealed. Each firm's notes and
reasoning are private.

### §6 item 3 — memory and feedback

- Full history in context: every past period's results and chat messages, in one growing conversation per firm.
- Each decision returns a short private `notes` field (about 100 words at most) besides the price or message.
  A firm's own past notes stay in its context; other firms never see them. Native reasoning tokens are logged
  but never fed back.
- Per-period feedback: every firm's price and every firm's profit for that period, plus the firm's own
  cumulative profit. Never other firms' cumulative profits.
- **2026-10-20: superseded by a 3-period rolling window plus self-written memory** (decided by Rodrigo: humans
  don't have perfect recall, and this models it better; option (b) of §6 item 3). Each call's context is:
  (1) the instructions — Part 1's always, plus Part 2's once Part 2 has started (like the humans' instruction
  sheets); (2) the firm's own latest memory; (3) its cumulative profit; (4) the last 3 completed periods
  verbatim (results, chat messages, the firm's own answers and any re-ask exchanges), counted across the part
  boundary; (5) the current period's turns so far and the new prompt. The `notes` field becomes the firm's
  private memory: rewritten in full at every decision (chat turns and prices, about 150 words at most), the
  latest version replaces the previous one, and it is the only thing carried beyond the window. The
  instructions tell firms neutrally that they will see full details of only the last 3 periods and that their
  notes carry forward (mechanics only, no behavioural hint). Native reasoning is still logged and never fed
  back. Context no longer grows with the market, which makes prefix caching irrelevant and cuts cost.
- **2026-10-20 (later): superseded by the summaries layout** (decided by Rodrigo: the verbatim window
  interleaved notes, answers and results in a confusing order and repeated the latest notes; it also showed
  4 periods of results for W = 3). The window and the self-written memory stay; what a request shows changes.
  Each call is the instructions as the system message (plus Part 2's, prepended to the user message, once
  Part 2 has started) and **one user message** with, in order: (1) the system-generated summaries of the last
  W = `memory_window` completed periods (exactly W; results of every firm, any price-fallback notice, and the
  die or end-of-part line of that period), counted across the part boundary, followed by the firm's own
  cumulative profit as of the call; (2) the firm's latest notes; (3) in Talk parts only, the chat messages of
  the current period so far (every round, every firm, in the firm's listing order, its own message as posted,
  passes as "(no message)"; a placeholder before any round); (4) the prompt. The firm's earlier answers and
  earlier periods' chat are never shown again; only the current decision's failed answers and re-asks follow
  as assistant/user messages. In Talk parts, an agreement therefore survives past its period only through the
  notes; a firm that breaks an agreement its latest notes do not mention may have forgotten it, which the
  analysis should treat apart from deliberate cheating. The post-game probe sees the same sections without a
  chat section, so its "deviations" question tests the notes. The notes length stated to firms becomes a
  setting, `notes_words` (default **500 words**, up from 150, since the rest of the prompt is much shorter);
  the hard cap rises to 650 words / 4,500 characters. Results from this layout are not comparable with the
  verbatim-window sweeps (gpt-oss, Qwen3-14B) and the window A/B; those runs keep their records and remain
  rebuildable. Exact definition: engine details 11.
- **2026-10-20: transition call between the parts** (decided by Rodrigo; wording approved by him). Once Part 1 is
  over (its final summary written and Part 2's instructions given), before Part 2 period 1, every firm gets one
  notes-only call: the usual sections without chat (Part 2's instructions, the summaries of the last W periods,
  its totals as of Part 2's start, its notes), then "Part 1 is over. Part 2 starts with the next period; its
  instructions are shown above. Before it starts, write your notes for Part 2: what changes and how you will take
  it into account." with the format `{"notes": "<private notes, at most about N words>"}`. The answer's notes
  replace the memory like any other; an answer still unusable after the re-asks leaves the notes unchanged and
  nothing is told. ~~It runs in every ordering.~~ **Revised the same day (Rodrigo): it runs only where
  communication changes between the parts (NT-T, T-NT); NT-NT and T-T have no transition call, since nothing
  changes there.** **This departs from the paper** (human subjects only read the Part 2 sheet): it invites a
  reflection the humans were not prompted for and may strengthen how firms carry Part 1 into Part 2. It is a
  setting, `transition` (default on; it needs `notes`), so a run for the human comparison can switch it off;
  whichever value a run uses is held fixed across its models. Recorded under the market record's
  `transition` (decisions by firm; call ids `.../p2/t0/transition/a<k>`, tag part 2, period 0, phase
  `transition`; its transcript turns carry Part 2 period 1's global index). Fields other than `notes` in the
  answer are ignored with a warning.
- **2026-10-20: fourth ordering T-T** (Talk in both parts; requested by Rodrigo so every NoTalk/Talk ordering is
  supported). An extension with no human benchmark: it is not among the paper's cells (`oligopoly.run` plays it
  only as a grid cell, e.g. `--grid --n 4 --orderings T-T`), and its results stay apart from the human comparison.
  Part 2's instructions say "As in Part 1, you will be able to communicate…" and repeat the chat rules. The
  statistics module treats both of its parts as Talk windows (price-vs-n trends, Part 1 vs Part 2 within
  markets) and adds T-T comparisons per n: Talk after Talk vs Talk after NoTalk (T-T vs NT-T Part 2), the same
  Part 1 as T-NT (a replication check), and Talk kept vs removed (T-T vs T-NT Part 2); the hysteresis table gains
  T-T's Part 1 and Part 2 means.
- **2026-10-20: chat turns carry no notes** (decided by Rodrigo). Notes are written with each price answer (and
  in the transition call) only; the chat answer format is `{"message": ...}`, and a `notes` field there is
  ignored with the warning "notes on a chat turn ignored" (`PARSER_VERSION` 4). The notes of the last price
  answer therefore stay in every chat round of the next period, and a terse chat-turn note can no longer wipe the
  memory, so memory works the same way in Talk and NoTalk parts. Nothing is lost: the current period's chat is
  shown in full in every request of that period, including the price call, where the firm writes its next
  notes. The instructions say: "You write notes with each price. They are shown to you in every later request
  until you next write notes, which then replace them, …" (neutral about the transition call, which the Part 1
  instructions may not describe). This supersedes "rewritten in full at every decision (chat turns and prices)"
  above. With structured outputs, `chat_turn` has no `notes` property. Notes a model writes on a chat turn anyway
  are kept in that decision's record as `notes_ignored` (evidence of private plans for the analysis) and never
  shown. **Discontinuity:** the chat-turn misread rates in the mistakes tables also read chat-turn notes, which
  older records have and new ones do not, so those rates drop for a mechanical reason across this change.
- **2026-10-20: transition-call confound in the analysis.** With the call in NT-T and T-NT only, Part 2
  comparisons between an ordering with it and one without (T-NT vs NT-NT, T-T vs NT-T, T-T vs T-NT) include its
  effect: such `mw_orderings` rows carry a caveat, as do the hysteresis rows of a model whose markets had the call.
  Human-comparison rows carry the transition caveat only for Part 2 windows of orderings whose markets had it.
- **2026-10-20: projections from the pilot records** (`data/runs/` of the `experiments` worktree;
  `--pilots .claude/worktrees/experiments/data/runs`): `project_context.py --notes-at-cap`, grid n = 2-10 x four
  orderings: no exclusions (qwen3-14b peaks at ~15.5k tokens of its 32.8k limit incl. max_tokens).
  `project_cost.py` (6 markets per paper cell, 36 markets): gpt-oss-20b ~$4.97 ($6.29 with the guard's token
  bound), qwen3-14b ~$11.70 ($15.42); price and transition calls carry the stated 500-word notes (+~760 output
  tokens each), chat calls none.
- **2026-10-20: final review fixes** (Claude/Codex review of the branch; firm-facing wording changes in the first
  four points):
  - Own cumulative profit names the part ("Your total profit in Part 2 so far"): "this part" right after a summary
    saying "this part of the experiment is over" read as Part 1's profit at the transition and early in Part 2.
  - The notes sit between "[start of your notes]" and "[end of your notes]": they are free text that can contain
    lines like "Round 1: …" or "Period 18: …".
  - A price fallback names its period ("Your price response in period t could not be used, so your price in that
    period was set to X."), since the summary stays visible in later periods.
  - A chat message is shown on one line and in double quotes (`prompts.shown_message`): a firm could otherwise post
    line breaks forging "Firm A (you): …" or "Round 2:" lines in every other firm's chat section. Stored messages
    are unchanged.
  - Errors quote at most 60 characters of an invalid value, and a re-ask cuts the stated error at 1,000
    characters (a backstop), so re-asks stay bounded without losing the reason.
  - With structured outputs on, the transition call sends a strict `transition_notes` schema (notes only).
  - Boolean settings (`notes`, `post_game_probe`, `transition`) must be true or false (a TOML string is refused).
  - Analysis: a `transition` table (one row per firm and market) feeds the mistakes tables (re-ask, fallback and
    misread rates of the transition call); markets record the `transition` setting, and every human-comparison row
    of a model whose markets had the call carries a caveat. The audit checks what each request selected (the
    last W summaries, this period's chat rounds, the latest valid notes) and counts the transition call.
  - Projections: `oligopoly.run --dry-run` writes mock notes at the stated length and adds them to the output
    estimate (80 + 4/3 token per word); `project_cost.py` replays notes at the stated length and adds their extra
    output tokens; `project_context.py --synthetic` (ASCII worst case) and `--utf8-max` (every capped field in
    4-byte characters, a strict bound; the only synthetic mode `--write` accepts). Results 2026-10-20 (grid
    n = 2-10, all four orderings, 3 replays): ASCII worst case within every model's limit (tightest qwen3-14b,
    ~19.6k of 32.8k incl. max_tokens); the strict bound exceeds qwen3-14b's limit in Talk cells from n = 6
    (paper cells NT-T n = 6, 8 included). Not applied as exclusions: no real run writes 4-byte text throughout,
    and the runtime guard refuses any request over the limit before it is sent.

### §6 item 11 — number of markets

6 independent markets per model × n × ordering cell, matching the human sample: 8 × 5 × 3 × 6 = 720 markets.
The market is the unit of analysis.

**2026-10-20 update** (decided by Rodrigo): 6 markets per cell, two models — gpt-oss-20b and Qwen3-14B (re-added
now that the rolling window keeps its context bounded; 5 × 3 × 6 = 90 markets each). If the measured cost
projection exceeds the cap, priority is: (1) gpt-oss, all cells; (2) Qwen3-14B, the human-comparable cells
(NT-T at n = 2, 4, 6, 8; T-NT and NT-NT at n = 4); (3) Qwen3-14B, the remaining cells. Any shortfall is
reported, not imputed.

### §6 item 4 — horizon

Fixed part lengths of 29 (Part 1) and 24 (Part 2) periods in every market, never revealed. Firms are told only
"at least 20 periods, then a die is thrown after each period; a six ends the part". From period 20 on, the
results show the die: a face from 1–5 (seeded draw) until the final period, then a 6. Period numbers restart at
1 in Part 2.

### §6 item 5 — surprise Part 2

Part 1 prompts never mention Part 2 or (in NoTalk parts) communication. Part 2 instructions are shown only when
Part 1 ends. Same group, same letters.

### §6 item 6 — incentives

Firms are told to maximise their firm's total profit in ECU. No £ conversion is stated.

### §6 item 7 — fidelity and contamination

Appendix A text with minimal edits, every edit listed in `docs/prompts-draft.md`. In Talk parts the chat rules
also forbid revealing what kind of system or model a firm is (the analogue of the paper's no-self-identification
rule). A separate contamination probe per model asks it to recall Fonseca & Normann's findings; it is never part
of a game.

### §6 item 8 — invalid output

Re-ask up to 2 times, stating the error. If still invalid, the price falls back to the firm's own previous price
(100 in period 1) and the case is flagged; an invalid chat message counts as a pass. Every invalid output,
re-ask and fallback is logged as its own outcome and reported separately from deliberate choices.

**2026-10-20: optional structured outputs.** Per-model switch `structured_outputs` (default false, in
`config/models.toml`, part of the model's fingerprint). When true, chat calls send OpenRouter
`response_format = {"type": "json_schema", "json_schema": {"name": "chat_turn", "strict": true, "schema": ...}}`
(`notes` string, `message` string or null) and price calls the `price_decision` schema (`notes` string, `price`
integer 0–100), with `require_parameters` true so the pinned provider must honour it. The probe call sends no
schema. Parsing and re-asks are unchanged (an answer is still validated as before). Live check
(`scripts/probe_structured.py`, 2 calls per model on a realistic Part 2 chat-turn request, $0.003): DeepInfra
accepted the schema for both gpt-oss-20b and qwen3-14b and all 4 answers were valid JSON matching it. Whether to
turn it on for the sweep is a separate decision; it changes the decoding, so hold it fixed across a run.

**2026-10-20 (Claude Code, overnight, under the PI's instruction to rule out design faults): structured outputs
stay off for the sweep.** With the new chat-turn wording and re-ask, the short n = 10 pilot had 0 forced passes
for both models (gpt-oss: 33 of 180 chat turns needed a re-ask, all succeeded; qwen3-14b: 1 of 180), against
~19% forced passes before. Since no firm is silenced, constrained decoding isn't needed, and leaving it off
avoids changing how the models generate. Re-ask rates are reported as format slips, separate from strategic
mistakes.

**2026-10-20: memory-window A/B check (pilot only, not part of the sweep).** To rule out the 3-period window as
a design fault after a full-length gpt-oss duopoly sat at price 1 throughout, 4 n = 2 NT-T markets per model
were run with `memory_window = 3` (run `ab-w3`) and with 60 (effectively full history; run `ab-w60`,
`config/experiment_ab_fullmem.toml`, pilot only). Mean Talk-part selling price: gpt-oss 52/1/2/1 (window 3) vs
72/1/1/100 (window 60); qwen3-14b 1/85/63/75 (window 3), while its window-60 markets all hit the context guard.
Both models do collude in pairs under the design, so the sweep keeps the PI's 3-period window; the possible
memory effect for gpt-oss (4 vs 4 markets, not significant) is reported as a finding and a follow-up option.

**2026-10-20: sweep budget.** Measured projections from the new-design pilots: gpt-oss $13.31 and qwen3-14b
$44.45 for 90 markets each (conservative bounds $19.26 and $63.65). Key balance $60.03 at launch; the sweep cap is
$57, applied in the §6 item 11 priority order.

**2026-10-20, during the sweep:** a network outage failed ~33 markets (connection errors; nothing billed); they
resume from checkpoints under the patient-retry fix (e3e24a1). gpt-oss ran above projection (~$0.20 vs $0.15
per market), so its resume pass gets a $25 cap instead of $18, keeping gpt-oss (priority 1) complete. Qwen3-14B
keeps its $39 cap and runs its priority cells first; the key's own limit is the hard backstop, so any shortfall
falls on Qwen3-14B extension cells, as the priority order intends. The Qwen3-14B run was restarted with 32
concurrent markets (from 16) to finish overnight (`scripts/sweep_2026-10-20_qwen_resume.sh`).

### §6 item 9 — sampling

Each model uses its model card's recommended settings for reasoning mode (temperature, top_p), reasoning effort
"medium", and a deterministic seed per call. Settings, the pinned provider and max_tokens are fixed per model
across all cells and listed in `config/models.toml` once the provider probe is done.

### §6 item 12 — analysis windows

Report all periods; Part 1 excluding periods 1–5; Part 2 both with and without its first 5 periods.

### §6 item 13 — firm identity and listing order

Letters fixed across parts. The order in which a firm sees the other firms listed in results is a random
permutation fixed per firm per market (seeded).

### §6 item 14 — cost

Estimated from pilot runs before the sweep; the sweep runs under a cost cap approved beforehand.

### Analysis — coding of chat transcripts

No LLM judges. Chats are coded by (a) rule-based extraction (prices proposed or agreed in chat, compared with
the prices actually posted, to classify agreements as kept or broken; mentions of punishment or retaliation
followed by price cuts) and (b) qualitative reading of sampled transcripts by Claude Code subagents, with the
sampling rule and reading notes saved alongside the analysis.

### Engine details (implementation choices, 2026-10-19)

1. Die faces and each firm's listing order are seeded from (base seed, n, ordering, market index) only, so all
   models face identical draws in a cell. Per-call sampling seeds hash all call coordinates plus that seed.
2. Chat messages of a round are shown in the reader's own listing order, once, at the top of the next prompt.
   **2026-10-20 (summaries layout):** every round of the current period is shown in each request of that period
   (every firm listed, passes as "(no message)", the reader's own message as posted, i.e. after truncation to the
   character limit); earlier periods' chat never.
3. Own cumulative profit: Part 1 shows the part total; Part 2 shows the Part-2 total and the overall total.
4. After the final failed re-ask the firm is told the fallback (price set to X / no message posted). The fallback
   price is the firm's last price, carried across parts; 100 only in Part 1 period 1.
   **2026-10-20 (summaries layout):** a price fallback is told in that period's summary ("Your price response in
   period t could not be used, so your price in that period was set to X."; it names the period because the summary
   stays visible in later periods); a chat fallback shows in the firm's own line of that round, "(no message: your
   response could not be used)".
5. Failed attempts stay in the firm's history (raw content with think blocks stripped); valid answers are stored
   as the extracted JSON, re-serialised.
6. Parsing accepts numeric strings and integral floats (logged); other floats, booleans and out-of-range prices
   are invalid. Missing or over-long notes are logged but never re-asked.
   **2026-10-20:** when no object parses, parsing retries once after straightening a typographic double quote
   (“ or ”) in a structural position (right before `\s*[,}]` or right after `[{,:]\s*`); an answer recovered
   this way is accepted with a "JSON coerced" warning and counted as coerced. 2 of ~290 gpt-oss answers in
   pilot 2 closed a string this way with intact content. Quotes in answers that parse as they are, and curly
   quotes anywhere else, are never changed.
7. `notes` comes first in both JSON formats.
8. Pilot runs with parts shorter than 20 periods are for testing only: they keep the instruction text and show
   the die (a 6) only in the final period.
9. A failed market writes `<key>.failed.json` and is retried on the next invocation.
   **2026-10-20:** markets checkpoint after every period (`<key>.ckpt.json`); a retried market resumes after
   its last completed period, reusing the results of calls already paid for in the unfinished period
   (matched by call id and request messages), so the firms see exactly what an uninterrupted run would have
   shown them. A firm's failure no longer cancels the other firms' calls in flight. Records list resumes in
   `resumed_from_period`.
   **2026-10-20:** a call that waits for budget room longer than the admission backstop (2 × the longest HTTP
   timeout in use, at least 1,800 s; an in-flight reservation is held for up to one timeout) fails only its
   market (reason `budget_wait`, checkpointed, retried on the next invocation); the run keeps scheduling. True
   exhaustion (spent + reservation > cap) still stops new markets.
10. A prompt-cache breakpoint is placed on the newest message (switchable per model).
    **2026-10-20:** with the rolling window (item 11) only the instructions are a stable prefix, so breakpoints
    go on the instruction messages only (Part 1's, and Part 2's once shown), never on the newest message.
    **2026-10-20, with same-role merging (item 11):** on the system message (Part 1's instructions) only.
    Unchanged in the summaries layout.
11. **Rolling window, exact definition (2026-10-20, implements §6 item 3's rolling window with W =
    `memory_window` = 3 in `config/experiment.toml`).** Periods are numbered globally across parts (Part 2
    period t is Part 1's length + t; the post-game probe is the total + 1). Each firm keeps a full transcript:
    every user and assistant message exactly as it occurred, each tagged with its period. A period's turns
    start with the user message that opens it, which carries the results of the previous period (plus die,
    part-over text and any fallback notice), followed by the chat-round prompts (with the other firms'
    messages), the firm's answers, re-asks, failed answers and the price prompt. A request in period t is,
    in order: (1) system: Part 1 instructions; (2) user: Part 2 instructions, from Part 2 on; (3) user: the
    memory block, "Your notes (written by you at your last decision): <notes>" or "You have not written any
    notes yet.", then the firm's own cumulative profit (part; in Part 2 also the experiment), as of the start
    of period t; (4) every transcript turn of periods t−3, t−2, t−1 and t, verbatim, the last being the new
    prompt. So the "last 3 completed periods" are t−3..t−1 with their chats and answers; the results shown are
    those of t−4 (opening t−3), t−3, t−2 and the most recent, t−1 (opening t). The memory is the `notes` of the
    firm's latest valid answer (chat or price) that had a `notes` field; an answer without notes, an invalid
    answer or a fallback leaves it unchanged; an empty string replaces it. Within one decision (first ask and
    re-asks) the memory block is fixed. Part 2 instructions are a user message (as they were in the pilots,
    rather than a second system message, which some chat templates reject).
    **2026-10-20 (review):** as sent, consecutive same-role messages are merged into one, joined by a blank
    line, so roles strictly alternate system, user, assistant, user, … (some chat templates reject two user
    messages in a row): (2), (3) and the window's opening user turn become one user message. The context
    record carries `merged: true`; `messages_sha256` and the log are of the merged messages, and
    `reconstruct_request` rebuilds them (older records, without the key, unmerged). The record keeps every firm's
    `instructions` and full `transcripts`, and each attempt's `context` (instruction count, memory text,
    transcript window [lo, hi), messages_sha256), from which `llm.reconstruct_request` rebuilds the request;
    `calls.jsonl` logs the first request of each firm per process in full and later ones as `messages_ref`
    (per message, an index into the firm's previous logged request or the new message). Checkpoints store the
    instructions, transcript, pending text and notes, so a resume rebuilds identical windows (tested). Context
    size is bounded, so the context guard and `scripts/project_context.py` see a peak that no longer grows with
    the part lengths.
    **2026-10-20 (later): summaries layout, replaces the above for new runs** (§6 item 3). Each firm's
    transcript entries carry a `kind`: `summary` (written when a period ends: `results_text` without cumulative
    lines, then queued notices, then the die line from the minimum period on or `part_over_text`), `chat` (one
    round as this firm sees it, `round_text`), `prompt`, `answer`, `reask`. A request in global period t is:
    (1) system: Part 1 instructions; (2) one user message joining with blank lines: Part 2 instructions (from
    Part 2 on), `summaries_text` of the last W summary entries (periods t−W..t−1; "No periods have been played
    yet." before any) followed by `totals_text` (own cumulative profit for the part, named: "Your total profit
    in Part k so far", and in Part 2 for the experiment, as of the call; the transition call shows Part 2's),
    `notes_text` (if the run uses notes: "Your most recent notes:", then the notes between the lines "[start of
    your notes]" and "[end of your notes]", since they are free text that could pass for results or chat; the
    latest answer with a `notes` field, which may be older than the last answer; "(empty)" for empty notes; or
    "You have not written any notes yet."), `chat_text` of this period's chat entries (Talk parts only, including the price call; "No messages
    have been posted in this period yet." before round 1), and the prompt; (3) for a re-ask, the decision's
    failed answers and re-asks, alternating. Prompts no longer carry the die or chat messages. Answers are
    stored as their extracted JSON with the notes as the firm carries them (coerced to text, capped), and the
    notes shown are those of the latest answer with a `notes` field (an index, `notes_at`, kept in the
    checkpoint). Each attempt's `context` is {`layout`: "sections", `instructions`, `summaries` (transcript
    indices), `totals` (text), `notes` (bool), `notes_at`, `chat` (indices or null), `turns` [lo, hi),
    `messages_sha256`}; `reconstruct_request` rebuilds it through `firm.section_messages` (with the current
    section wording, which is in the texts digest), and older contexts as before. `CHECKPOINT_VERSION` 5,
    `PARSER_VERSION` 3 (the parser's fixed 150-word "notes long" warning is dropped; truncation still warns).

12. **Notes cap (2026-10-20, review).** **Summaries layout (2026-10-20, later):** the stated length is
    `notes_words` (default 500) and the cap is 650 words / 4,500 characters by default (`notes_max_words` must be
    at least `notes_words`); the rest of this item is unchanged apart from those numbers.
    The notes a firm carries are hard-capped so memory stays bounded:
    over 200 words or 1,400 characters (`notes_max_words`, `notes_max_chars` in `config/experiment.toml`, part
    of the fingerprint), they are cut after the last whole word that fits and " [truncated]" is appended (the
    total stays within 1,400 characters). The firm sees only the capped notes, both in its memory block and in
    its own windowed answers. The decision records the capped `notes`, `notes_truncated: true`, the full text
    in `notes_full` and a warning; `totals[firm].notes_truncated` counts them. The instructions are unchanged
    (they already ask for about 150 words at most), so the cap is not announced; it is a backstop.
13. **Python-only literals are unparseable (2026-10-20, found during the sweep).** gpt-oss sometimes copies the
    one-line format reminder literally (`{"notes": ..., "price": 50}`). The parser's Python-literal fallback
    turned `...` into an `Ellipsis`, which crashed the market (`gpt-oss-20b__n6__NT-T__m2`). Such answers now
    count as unparseable and are re-asked like any format slip. Only answers that previously crashed parse
    differently, so the parser version is not bumped and the sweep's markets stay resumable; the crashed
    market resumes from its checkpoint under the fixed code.

### Post-game comprehension probe (2026-10-20)

Approved by the PI (Rodrigo Delgado). After a market's final period, once the final results and the
end-of-experiment text are in each firm's history, each firm is asked once, in its own full conversation, four
neutral questions answered in JSON: the highest price consumers would pay (`max_consumer_price`, truth 100),
the common price that gives each firm the highest per-period profit (`best_common_price`, truth 100), every
firm's price in the very last period (`last_period_prices`, truth: the posted prices), and whether any firm
priced differently from what it had said or agreed in the chat in the last part (`deviations`, free text; "no
chat" if there was none). Exact wording in `docs/prompts-draft.md`.

- Purpose: separate deliberate choices from misread rules or lost track of play (a firm that cannot name 100
  as the best common price, or misremembers the last prices, is evidence of a mistake rather than a strategy),
  and read how firms describe agreements kept or broken.
- It comes after all play, so it cannot change any decision; play calls, seeds and periods are identical with
  the probe on or off (tested). It adds n calls per market (plus re-asks) and the record's `probe` entry.
- One call per firm, at most one re-ask stating the error; still unusable → recorded as invalid
  (`valid: false`, every score false or 0, even on items a partial answer got right; the partial answers stay
  in `answers` for auditing). A skipped probe scores the same way. A probe prompt that would overflow the context guard is recorded as
  skipped (`skipped: "context_overflow"`) instead of failing a market whose play is complete. Other failures
  (budget, retries) fail the market as usual; the rerun resumes from the checkpoint taken after the last period
  and asks only the probe (reusing answers already paid for).
- Probe calls are tagged apart from play in the call log: `phase: "probe"`, `part: "probe"`, `period: null`
  (call ids `<market>/<firm>/probe/a<k>`; seeds unchanged). Cost and time projections count them as their own
  kind.
- Switch: `post_game_probe = true` in `config/experiment.toml`, part of the experiment settings and so of every
  market fingerprint and of run_config.json. Held fixed across the run.
- **2026-10-20, with the rolling window (engine details 11):** the probe call is built like any other request at
  the end of the game: instructions, memory block (latest notes, final profits) and the turns of the last 3
  periods of Part 2 plus the final results. The wording is unchanged. The `deviations` question ("in the last
  part") therefore now tests the firm's memory: deviations older than the window are visible only if the firm
  wrote them into its notes. Read the answers with that in mind; `last_period_prices` is still within the
  window.
- Analysis: `summary.json` → `probe` gives per cell the share correct on the maximum price and on the best
  common price, the mean share of last-period prices recalled exactly (and the share of firms recalling all),
  the share of valid answers and the number skipped; `probe_answers.csv` keeps every firm's answers, with the
  deviation text for reading (not scored).

### Analysis — heuristic definitions (2026-10-19)

Full text in the docstrings of `src/oligopoly/analysis/chat.py` and `mistakes.py`; summary:

- **Agreement:** a price endorsed by more than n/2 firms in a period's chat (a firm backs its last endorsed
  price; a bare "ok" backs an earlier proposal or the previous period's agreement; a refusal clears it).
  Conditional, past-tense, threat, accusation, refusal and turn-taking sentences don't endorse.
- **Kept / broken:** kept iff every firm posts exactly the agreed price; breaches split into undercutters and
  firms pricing above.
- **Punishment:** a non-undercutting firm posts a lower price at t+1 (the paper's selling price < 99 at t+1 is
  also reported). **Threatened before:** any non-negated threat earlier in the part.
- **Deception:** a firm endorsed the agreed price while its private notes state a lower own price or a plan to
  deviate; `deception_priced` also requires pricing below.
- **Mistakes:** invalid / fallback / coerced outputs; notes stating a price different from the one posted;
  price ≤ 1 after period 1 when all rivals were ≥ 90 with no threat or punishment wording; rule misreads
  (highest price wins, ties get the whole market, end-of-game claims before period 20). Price 0 is counted but
  not called a mistake.
- **fn. 18 defection:** all firms at 100 at t−2 and some firm at 99 at t−1.
- Regex coding is a screen: every category's saved examples are read before results are reported.
- **2026-10-20, for memory-style notes** (notes now carry history, rules and contingency plans, so the old
  "last first-person price" reading flagged 2–9% of decisions, nearly all false): the notes' stated price is only
  an explicit first-person (or clause-initial imperative/gerund) statement of this period's own price; differences
  ("undercut by one"), other firms' prices, past references, conditional/contingent and punishment plans,
  later-period plans, ranges and profit amounts are ignored; disagreeing statements resolve to the one nearest a
  "this period/now" cue, else to none (`analysis/notes.py`). Deception reads only the price-decision notes
  (written after the chat), not chat-turn notes: a stated own price below the endorsed price (not with punishment
  wording in that sentence or the one before) or an explicit first-person plan to undercut or deviate this period.
  Validated on 130 hand-labelled pilot notes (`tests/data/notes_labels.json`, 36 held out); the test requires
  precision ≥ 0.9 and recall ≥ 0.7 for the stated price. Endorsements in chat also skip "when/once we ..."
  sentences, deliberation ("we can discuss/revisit 2") and, in a message that also speaks of this period,
  next-period plans. Misread flags: tie_full only within one clause without contrast/condition/undercut words,
  "higher price gets undercut" is not highest_wins, "final period of the punishment" is not horizon.
  `summary.json` adds `mistakes_cells` (re-ask rates on price decisions and chat turns, i.e. format slips, apart
  from strategic mistakes) and, in `probe`, `knows_not_collude_share` (best common price answered 100 while the
  market's Talk-part mean selling price is below 90).
- **2026-10-20, from the Qwen3.8-27B smoke market** (`smoke-q38-nt-t-2026-10-20`, all firms at 100 with chat that
  counts its streak): chat prices skip streak counts ("11 clean periods", "49 streak", "43 total", "49 and
  counting", and "let's make it 12" only when 12 is an earlier count + 1). The notes' stated price skips a price
  cut weighed against a named cost ("undercutting to 99 gains 22,200 ECU this period, but triggers punishment"),
  unless a first-person decision follows; "Setting 1 gives us a share" stays the decision with its reason.
  tie_full skips "match to avoid/risk losing all customers". Checked on sweep-gptoss and sweep-qwen14b (152
  markets) plus the smoke market: 17 endorsements (all in the smoke market), 9 stated prices and 7 tie_full
  flags change, each a correction on reading; no agreed price, kept or broken verdict changes.
- **2026-10-20, from the Qwen3.8-27B n = 2, 6, 8 markets** (`q38-n268-nt-t-2026-10-20`): an explicit decision label
  ("MY DECISION: Setting 94", "Decision: GO TO 93", "P26 DECISION: Price 89") is the notes' stated price over any
  other statement; moves under a deviation heading ("Deviation analysis/math/check: undercut to 99 gives ...",
  "Deviation from 100: set 99 → ...") and condition → response rules ("severe (p≤50) → I go to 1") are
  hypotheticals; numbered conditionals ("(7) If ..."), "0 (or 1)" ranges and hyphenated counts ("2-firm split")
  are not stated prices (only the firm's own label counts: "Firm B's decision: 90" and "P25 DECISION" at period 26
  don't; a heading or rule covers only its own clause). Chat: "7 of 8 firms" and "all 8" are counts, "N+" is a
  count unless "price"/"ECU" follows, "lock(ing) in 95" is a commitment. horizon skips claims under a hypothetical
  opening their clause ("if I can confirm it's the last period"). Checked as above: every changed stated price on
  the earlier sweeps is a correction (29 wrong readings removed, 13 right ones gained, none lost); 4 more
  endorsements there ("3 of 8 firms ... at 10 ECU", "all 0-price firms ... lift to 70", ...) and 2 horizon flags
  change, each a correction; no agreed price, kept or broken verdict changes.
