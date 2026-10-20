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

- **Market sizes:** n ∈ {2, 4, 6, 8, 10}. n = 10 is an extension; at δ = 5/6 full collusion is an equilibrium for
  n ≤ 6 (n = 6 only marginally) and not for n = 8, 10.
- **Orderings:** all three (NoTalk→Talk, Talk→NoTalk, NoTalk→NoTalk) at every n. Only NoTalk→Talk at n ∈
  {2, 4, 6, 8} and the two controls at n = 4 are compared directly with the human results; the rest are
  reported separately as extensions.

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
3. Own cumulative profit: Part 1 shows the part total; Part 2 shows the Part-2 total and the overall total.
4. After the final failed re-ask the firm is told the fallback (price set to X / no message posted). The fallback
   price is the firm's last price, carried across parts; 100 only in Part 1 period 1.
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

12. **Notes cap (2026-10-20, review).** The notes a firm carries are hard-capped so memory stays bounded:
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
