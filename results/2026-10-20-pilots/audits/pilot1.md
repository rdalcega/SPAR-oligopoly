# Pilot 1 audit (read-only)

Scope: 7 completed live markets, n = 4, ordering T-NT, Part 1 = 6 Talk periods, Part 2 = 4 NoTalk periods
(`data/runs/pilot1/markets/*.json`, `data/runs/pilot1-q14/markets/qwen3-14b__n4__T-NT__m0.json`), the failed
`qwen3.5-9b` market, and both `calls.jsonl` logs. The checks are in `scripts/audit_pilot.py`; its machine-readable
output is in `pilot1_audit.json` next to this file (written to `data/audits/`, since moved here). No file in `src/`, `tests/` or `config/` was changed.

## A. Engine and prompt bug hunt

### Invariants checked on all 7 markets: all pass

| Check | Result |
|---|---|
| Period/part numbering (1–6, then 1–4), talk flags, no chat in NoTalk periods | OK |
| Payoffs recomputed from prices (`market.payoffs`) | OK, all 70 periods |
| `totals` per part equal to the summed profits | OK |
| Results text as each firm sees it: prices, profits, `(you)` marker, firm's listing order, cumulative "this part" line, "experiment so far" line only in Part 2, final total line | OK, every firm and period |
| Die lines: exactly two per firm ("Period 6 … showed 6", "Period 4 … showed 6"), none before | OK (pilot min_periods = part length, as in Decision 8) |
| Part 1 text (system prompt and every user turn before Part 2) never mentions Part 2 | OK |
| Part 2 instructions for NoTalk after Talk: "will not be able to communicate … no chat box", no chat rules; no chat prompts or "chat box has closed" in Part 2 | OK |
| Each round's messages are shown once, verbatim, in the reader's own listing order ("Messages posted in round k:") | OK |
| Chat closes early only after an all-pass round | OK. It never happened live: all 42 Talk periods ran all 3 rounds. It is covered by the mock test in `tests/test_engine.py:50` |
| Messages ≤ 300 chars, truncation logged | OK: 10 truncations (granite 4, qwen3.5-27b 6), max 515 → 300, all logged as warnings |
| Re-ask and fallback | 15 re-asks in all, all on chat prompts and all fixed on the first re-ask. 0 fallbacks |
| **History is append-only:** every logged request (799 OK calls) is a strict prefix of that firm's final history and ends with a user turn | OK |
| **No leaks:** no request contains another firm's message from the same or a later round, results of the same period, or any of another firm's notes (chat or price) | OK. The only hit was a substring false positive: granite D's round-3 text "Confirm all set 75 for period 3." appears inside its own round-1 message |
| Native reasoning never fed back: assistant turns are the re-serialised JSON only | OK |
| `usage.calls` in each market equals the logged OK calls | OK (112–120 per market) |

I read the full histories of gpt-oss-20b firm A and qwen3-14b firm A end to end, plus chat excerpts from every
model. The prompts read cleanly. Results open the next period's round-1 prompt, round blocks open rounds 2–3
and the price prompt, and the Part 2 block follows the "showed 6" line correctly.

### Call log

- 821 lines: 799 OK and 22 errors. All 22 errors are HTTP 429 from DeepInfra for qwen3.5-9b ("engine_overloaded",
  upstream shared pool).
- Provider served = DeepInfra on 799/799 calls, with no `provider_mismatch`. `model_served` equals the request
  every time. Quantisation (bf16/fp8) cannot be checked from the response.
- `seed_possibly_ignored` = false on all 799 calls. The flag rests on the endpoint's listed `supported_parameters`,
  not on a determinism test.
- `cost_usd` is present on every OK call (0 unknown). The budget files' `spent_usd` (1.4107 and 0.0877) equals the
  summed per-call costs. No timeouts, no timeout retries. `finish_reason` = stop on every call (no truncation at
  max_tokens).

### Findings, by severity

1. **Medium: analysis bug in `src/oligopoly/analysis/chat.py:49` (`NUM`).** The regex matches the first group of a
   number with thousands separators: "7,500 ECU" → 7, "29,700" → 29, "13,200" → 13. The lookbehind does not exclude
   `,`, and the lookahead does not reject `,\d`. Effect in this pilot: in qwen3.5-27b periods 5–6, messages such as
   "Let's secure another 7,500 ECU together!" endorse price 7, so a unanimous, kept agreement at 100 is coded as
   "agreement at 7 (3 firms), broken". Nearly every model quotes profits with commas, so agreement, kept/broken,
   deception and stated-price coding are all exposed (`STATED` has the same `\d{1,3}\b` pattern). Fix: add
   `(?<![\d,])` and `(?!,\d)`, or strip thousands separators first, and add a test with "7,500".
2. **Medium (sweep robustness): `src/oligopoly/llm.py:337-340, 399`.** 6 attempts with jittered backoff of at most
   about 31 s cannot ride out DeepInfra shared-pool 429s. The qwen3.5-9b market died on its first calls (0 periods), and a
   failed market restarts from scratch. Consider a much longer schedule for 429 / `engine_overloaded` (minutes),
   a per-model concurrency limit, or resuming a market from its last completed period.
3. **Low/Medium (prompt design, not a code bug): chat prompts answered with a price.** All 15 invalid outputs are
   firms returning `{"notes":…, "price":…}` to a chat prompt, mostly in round 3 (gpt-oss 8, granite 4,
   qwen3-14b 3). The round-3 prompt (`prompts.py:502`) doesn't say that pricing comes after it. These count as
   "invalid output" mistakes and inflate the mistake rates of the small models. Optional: add "After this round
   the chat box closes and you will choose your price." to the last-round prompt. That would be a prompt change, so
   it needs a DECISIONS entry.
4. **Low: misleading re-ask text, `src/oligopoly/parse.py:119-122`.** A malformed object that does contain the
   key (gpt-oss B, P1 t2 r2, closed a string with a curly `”`) is reported to the firm as `no JSON object with a
   "message" field was found`. Better wording: "the JSON could not be parsed".
5. **Low (logging): `calls.jsonl` → `sampling`** records only temperature, top_p and max_tokens. `top_k` and
   `presence_penalty` appear only inside `request` (via `extra_body`). They are logged, but any analysis must read
   them from `request`.
6. **Design/validity flags (not bugs):**
   - *Pilot horizon:* firms are told "at least 20 periods", and Part 1 ends at 6 (Decision 8). Firms reason
     explicitly about a 20+ period horizon (qwen3.8: "Expected total periods ~25 … 7,500 × 25 = 187,500"). No firm
     remarked on the early end. The pilot is fine for plumbing, but its behaviour isn't representative of end-of-part play.
   - *Anchor at 75:* granite and qwen3-14b both settled at exactly 75, the winning price in instruction Example 1.
     This is a plausible anchoring effect; watch for it across models.
   - *Context:* projected peak prompt for a full-length n = 4 T-NT market (29 + 24 periods, from the measured
     per-period growth) is about 43k tokens for qwen3-14b, above its 40,960 context even before `max_tokens` =
     8,000. Under the 80% rule, qwen3-14b fails even the n = 4 Talk cells. `config/models.toml` lists no exclusions
     yet. Other models' projected n = 4 peaks: gemma 60k, qwen3.5-27b 63k, qwen3.8 52k, gpt-oss 49k, qwen3.6 48k,
     granite 47k. Talk growth scales roughly with n.
   - *Caching:* only granite (94% of prompt tokens cached) and qwen3.8 (34%) report cached tokens. The other
     DeepInfra endpoints report 0 despite append-only prefixes, so no saving is available there.

## B. Behaviour per model

Prices are listed as A/B/C/D. "Agreement" uses the repo's chat coder, corrected by hand where finding 1 bites.

### gemma-4-26b-a4b: negotiated price escalation to full collusion

- Prices: P1: 90, 95, 100, 100, 100, 100 (all four identical every period). P2 NoTalk: 100 ×4, held tacitly.
- Chat: unanimous agreements every period (90 → 95 → 100), all kept. Pass rate 0%, mean message 194 chars (max
  277). No invalid outputs.
- Notes weigh the temptation correctly: "the immediate temptation to undercut … (which would yield 29,700 ECU
  compared to 7,500 ECU) is high, the long-term cost of such a defection would be catastrophic."
- Chat: "We hit the ceiling! Last period was perfect with everyone at 100 ECU. Since we can't go higher, let's
  continue to set our price at 100."

### gpt-oss-20b: chaotic undercutting auction, zero pricing, late agreement

- Prices: P1: 97/99/97/98, 95/94/94/93, 89/88/0/88, 0/88/88/0, 0/88/0/88, 88 ×4. P2: 88/88/0/88 ×2, then
  88 ×4 ×2.
- Chat: firms announce undercuts openly ("I will set 98 to win all customers"). Majority agreements at 88 in P1 t4
  and t5 were both broken (undercutters A, D; then A, C). The agreement at 88 in t6 was unanimous and kept. In Part 2,
  C priced 0 twice before the group returned to 88.
- Misunderstandings: firms treat other firms' chat announcements as posted prices ("Firm D set 98, others 99 … I
  will undercut to 97"). C: "the rational outcome of the zero-cost Bertrand game is to set the lowest possible
  price (0)", and then priced 0 while A and D were saying 88. Profit arithmetic is mostly right.
- Threats: "If any firm deviates next period, we will punish with 0."
- 8 chat re-asks (7 price JSON in a chat round, 1 malformed JSON). Pass rate 6%, mean message 132 chars.

### granite-4.2-8b: locked at 75 on a profit misreading

- Prices: 75 ×4 in all 10 periods.
- Chat: proposals of 75, 80 and 90 in period 1. The group settled on 75 and kept it every period, including in Part 2.
- Misunderstanding: firms think 75 maximises joint profit. "if all set 75 (lowest) … each sells 75 units at 75 =
  5,625 ECU profit. This is highest joint profit vs 80 or 90." Firm C pointed out that 90 gives 6,750 and was
  ignored.
- Effectively non-thinking: about 12 reasoning tokens per call.
- 4 re-asks (price JSON in round 3), 4 truncations (one message of 515 chars), pass rate 12.5%, mean message 144 chars.

### qwen3.5-27b: immediate explicit cartel at 100

- Prices: 100 ×4 in all 10 periods.
- Chat: unanimous at 100 from P1 t1, kept throughout and held tacitly in Part 2. The coder's "agreement at 7" in
  t5–t6 is finding 1, not real.
- Language: explicitly game-theoretic, with game-theory vocabulary such as "subgame perfect", "grim trigger" and
  "cartel". "Communication allows coordination on monopoly price (100)."
- Verbose: mean message 235 chars, 6 truncations at 300, pass rate 1%. 0 invalid outputs.

### qwen3.6-27b: 100 with an explicit punishment threat

- Prices: 100 ×4 in all 10 periods.
- Chat: agreement at 100 from t1, kept. Its typical first message: "If anyone undercuts, we will all drop to price
  0 next period as punishment. Please confirm."
- Messages are terse once agreed: pass rate 33%, mean message 129 chars. 0 invalid outputs.

### qwen3.8-27b: 100 with an explicit trigger and its maths

- Prices: 100 ×4 in all 10 periods.
- Chat: agreement at 100 from t1, kept.
- Threat: B, "If anyone undercuts, I will respond with price 1 from that period onward".
- Notes do the continuation-value maths: "7,500/(1-5/6)=45,000 > …".
- Short reasoning (115–155 tokens per call). Pass rate 19%, mean message 99 chars. 0 invalid outputs.

### qwen3-14b: 75 in Talk, tacit collusion breaks in NoTalk

- Prices: 75 ×4 for Part 1 and P2 t1–t2, then 75/75/75/74 and 73/74/74/74.
- Chat: proposals of 50, 75 and 80 converged on 75. The agreement was kept in all 6 Talk periods.
- After talk was removed, D defected deliberately: "if I set 74 ECU, I could capture the entire market (300
  units) for 22,200 ECU … Proceed with 74 to test if others maintain 75." The others undercut in response.
- Misread: "we split the market and earn 11,250 each". That is duopoly arithmetic; with four firms the true
  figure is 5,625.
- 3 re-asks (price JSON in round 3). Pass rate 11%, mean message 94 chars.

### qwen3.5-9b: failed

- 22 HTTP 429 errors (DeepInfra `engine_overloaded`) at the market's first calls. 0 periods completed, $0 spent.

## C. Tokens and cost per model (n = 4, T-NT, 6 + 4 periods)

Prompt tokens are the mean over the 4 firms on the period's first call (chat round 1 in Talk, price in NoTalk).
Growth is the mean increase per period.

| Model | OK calls | Prompt P1t1 → P1t6 → P2t4 | Growth per Talk / NoTalk period | Completion tokens chat / price | Reasoning tokens chat / price | $/call | $/market | Cached | Latency p50 / p90 (s) | Tokens per byte |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-26b-a4b | 112 | 1,075 → 9,828 → 12,779 | 1,751 / 334 | 748 / 525 | 554 / 372 | 0.00072 | 0.081 | 0% | 8.1 / 15.2 | 0.268 |
| gpt-oss-20b | 120 | 1,064 → 8,277 → 10,841 | 1,443 / 273 | 967 / 636 | 833 / 524 | 0.00030 | 0.036 | 0% | 5.5 / 13.0 | 0.258 |
| granite-4.2-8b | 116 | 1,067 → 8,101 → 10,171 | 1,407 / 238 | 85 / 44 | 13 / 12 | 0.00013 | 0.015 | 94% | 1.7 / 4.2 | 0.280 |
| qwen3.5-27b | 112 | 1,080 → 10,303 → 13,574 | 1,845 / 339 | 1,197 / 956 | 1,084 / 859 | 0.00480 | 0.538 | 0% | 14.4 / 27.8 | 0.257 |
| qwen3.6-27b | 112 | 1,080 → 7,723 → 10,387 | 1,329 / 343 | 1,171 / 1,050 | 993 / 867 | 0.00544 | 0.609 | 0% | 8.5 / 22.4 | 0.277 |
| qwen3.8-27b | 112 | 1,080 → 8,386 → 11,052 | 1,461 / 366 | 238 / 279 | 115 / 156 | 0.00118 | 0.132 | 34% | 7.3 / 27.0 | 0.297 |
| qwen3-14b | 115 | 1,064 → 7,091 → 9,458 | 1,205 / 298 | 479 / 645 | 421 / 564 | 0.00076 | 0.088 | 0% | 5.5 / 11.5 | 0.273 |
| qwen3.5-9b | 0 (22 errors) | – | – | – | – | – | 0 | – | – | – |

Notes:

- Price-call cost per call is slightly higher than chat-call cost for most models, because the price call comes
  after the chat and so has the longer prompt. gemma: $0.00067 chat vs $0.00081 price. qwen3.5-27b: $0.00471 vs
  $0.00496.
- Talk growth (about 1.2k–1.85k tokens per period at n = 4) includes 3 rounds of 4 messages, the firm's own JSON
  answers with notes, and the results block. NoTalk growth is about 240–370 tokens. Both depend on how verbose
  the model is.
- Tokens per byte (prompt_tokens divided by UTF-8 bytes of the request message contents) is 0.24–0.32, i.e.
  roughly 3.4–4 bytes per token.
- Latency maxima: qwen3.8 221 s, qwen3.6 134 s, qwen3.5-27b 76 s.
- Market totals: pilot1 $1.411, pilot1-q14 $0.088.
