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
10. A prompt-cache breakpoint is placed on the newest message (switchable per model).

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
