# Fonseca & Normann (2012): what to know to replicate it with LLMs

**Paper:** Miguel A. Fonseca and Hans-Theo Normann, "Explicit vs. tacit collusion — The impact of communication in
oligopoly experiments", *European Economic Review* 56(8), 1759–1772, 2012.
DOI [10.1016/j.euroecorev.2012.09.002](https://doi.org/10.1016/j.euroecorev.2012.09.002) ·
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0014292112001237)

**Source used for this summary:** the open working-paper version, DICE Discussion Paper No. 65 (August 2012),
<https://d-nb.info/1026225752/34> (also at [EconStor](https://www.econstor.eu/handle/10419/62592) and the
[Exeter repository](https://ore.exeter.ac.uk/repository/handle/10871/14991); direct PDF:
<https://www.econstor.eu/bitstream/10419/62592/1/724283404.pdf>). The journal version is paywalled and was not
read. Page numbers below refer to the working paper. Treat any small numeric differences from the journal version
as possible.

**Verification (2026-10-19):** the numbers, quotations and design statements below were re-checked against
the working-paper PDF. The instructions in §4 match the paper word for word. The journal version could not be
obtained (ScienceDirect blocks automated access and the Exeter repository was unreachable), so the check against
it is still outstanding. Points to confirm in the journal version, all flagged where they occur: Table 1 cell
counts (§2), the results-screen feedback (§3), the analysis window for Part 2 (§5), the wording of fn. 14 (§7),
the "p < 100" labels in Table 4 (§7) and the "remaining seven periods" count for eight-firm market 4 (§7).

Sections 1–5 report what the paper says. Section 6 lists what the paper leaves unspecified and what an LLM run
must decide. Section 7 gives the human results to compare against.

---

## 1. The market (stage game)

| Item | Value |
|---|---|
| Model | Bertrand price competition, homogeneous good, inelastic demand, constant marginal cost |
| Consumers | m = 300, each buys one unit, reservation price p_max = 100 |
| Firms per market | n ∈ {2, 4, 6, 8} |
| Action | Each firm picks an **integer price in {0, 1, …, 100}**, simultaneously and independently |
| Cost | Zero; each firm can supply all 300 consumers |
| Payoff | Lowest-price firm earns 300 × p. Ties at the lowest price split 300 × p equally. Higher-priced firms earn 0 |
| Currency | ECU (Experimental Currency Unit) |

Worked examples from the instructions (duopoly):
- A = 85, B = 75 → B earns 300 × 75 = 22,500 ECU; A earns 0.
- A = B = 70 → each sells 150 units, earning 10,500 ECU.

**Theory (pp. 7–8, fn. 2, fn. 16):**
- Static Nash: all firms price at 0, and with integer prices also all at 1, so profits are about zero.
- Repeated game with grim-trigger (Nash-reversion) strategies: everyone pricing at 100 is subgame perfect iff
  δ ≥ (n−1)/n. That is 0.5, 0.75, 0.833 and 0.875 for n = 2, 4, 6, 8.
- With the continuation probability of 5/6 ≈ 0.833 used as δ, under (1) full collusion is an equilibrium for
  n = 2 and 4, exactly at the boundary for n = 6, and **not an equilibrium for n = 8**. That is the "conventional wisdom"
  the design is built to test. Footnote 2 adds that below 0.5 none of the four oligopolies has a collusive
  equilibrium and at 0.875 or above all of them do.
- The paper derives (1) by bounding the deviation profit by p·m (undercutting by a vanishing amount). With
  integer prices the best deviation from all-100 is 99, so the exact condition is δ ≥ 1 − 100/(99n): about 0.495,
  0.747, 0.832 and 0.874 for n = 2, 4, 6, 8. At δ = 5/6 the n = 6 condition holds, but only by a margin of
  0.0017 (0.8333 vs. 0.8316), and by 0.0003 if punishment reverts to the all-at-1 equilibrium of fn. 3 instead of
  all-at-0 (threshold (99n − 100)/(99n − 1) ≈ 0.8331). For design purposes n = 6 at δ = 5/6 is at the threshold,
  not safely inside it. The qualitative split (n ≤ 6 yes, n = 8 no) is unchanged.
- Taking turns as the 99-pricer (all others charge 100) needs δ solving 1 − δⁿ − δⁿ⁻¹ = 0, which ranges from about
  0.62 (n = 2) to 0.91 (n = 8), slightly above (n−1)/n in each case (fn. 16).

## 2. Treatments

Each session has **two parts**, and the same group stays together through both. Communication is varied
**within subjects**.

| Part 1 → Part 2 | n=2 | n=4 | n=6 | n=8 | Purpose |
|---|---|---|---|---|---|
| NoTalk → Talk | 6 | 6 | 6 | 6 | Main design |
| Talk → NoTalk | | 6 | | | Order effects and hysteresis (collusion lasting after talk is removed) |
| NoTalk → NoTalk | | 6 | | | Control for repetition and learning |

Cells give the number of independent markets (groups). The text says "six markets (or groups) for each
treatment" (p. 6), and 168 subjects = 6 × (2 + 4 + 6 + 8) + 2 × 24 only works with six. **Table 1 of the working
paper nevertheless prints "4" in every cell**; that is almost certainly a typo, but check it against the journal
version. The paper uses "part" and "phase" interchangeably; the instructions say "part".

- **NoTalk:** firms post prices with no communication.
- **Talk:** before pricing in **every** period, firms get a **1-minute free-form typed chat** in an instant-messenger
  tool, visible to everyone in their market and no one else.
  - Firms may send any number of messages, but may not identify themselves (age, gender, location, etc.) or use
    offensive language. Nothing else is restricted.
  - Most chats ended before the minute was up, even in 8-firm markets.
  - The chat box then closes and firms enter prices.
- **Surprise restart:** subjects read Part 2's instructions only after Part 1 ended. Communication was never
  mentioned before Part 2.
- **Firm labels:** firms are anonymous letters (A, B, C, …). Each subject's letter was printed on their
  instruction sheet, and chat transcripts show "Firm A:", "Firm B:", and so on.

## 3. Repetition, matching and feedback

- **Matching:** fixed groups (a partner design) for the whole session.
- **Horizon:** at least 20 periods per part. From period 20 on, a virtual die decides whether play continues: a
  six ends the part, so the continuation probability is 5/6. The die is described to subjects (instructions,
  p. 31).
- **Fixed lengths:** the actual lengths were drawn in advance and **used in every session**: **Part 1 = 29
  periods, Part 2 = 24 periods**, or 53 in total. Periods 1–29 are Part 1 and 30–53 are Part 2.
  - The instructions say the die is thrown "once the 20th period is over". The design text says "from the 21st
    period on, a random stopping rule determined whether the experiment would go on". In both readings, play
    continues after period 20 with probability 5/6.
- **Order within a period:** (chat, in Talk) → every firm enters a price → results screen.
- **Results screen (design text, p. 6):**
  - **every firm's price** in the market and **each firm's profit** for that period;
  - the subject's **own** cumulative profit, **not** other firms' cumulative profits.
  - **Inconsistency:** the instructions (Appendix A) say only that firms are "informed of the chosen prices by
    all firms and their own profits". The design text is more specific and describes the software, so the
    summary follows it, but the two readings differ in whether a firm sees rivals' period profits. In this game
    rivals' period profits are computable from the price vector anyway (300 × p for the lowest-price firms,
    split on ties, 0 otherwise), so the choice matters for prompt wording more than for information. Decide and
    record it (§6, item 3).

## 4. Subjects, incentives and procedures

- **Subjects:** 168 students at the FEELE lab, University of Exeter, programmed in z-Tree.
  - Each person took part only once and had no prior oligopoly experiments.
  - Group counts: 6 × (2 + 4 + 6 + 8) = 120 in the main design, plus 24 + 24 in the two n = 4 controls, giving 168.
- **Sessions:** 1 for all duopolies, 2 each for n = 4 and n = 6, and 3 for n = 8. The paper doesn't say how the
  control treatments were split across sessions.
- **Instructions:** written and read aloud. Subjects were asked to stay silent. The text frames the subject as
  "in the role of a firm".
- **Pay:**
  - £5 show-up fee plus total profits over both parts, at **£1 = 50,000 ECU in duopolies**.
  - The rate was adjusted for larger n "such that maximum possible earnings were equalized across treatments".
    The exact rates are not given.
- **Duration and earnings:** about 60 minutes; average payment £13.71 (about $22).

### Subject instructions (Appendix A, duopoly version, verbatim apart from layout)

> **Instructions – Part 1**
> Hello and welcome to our experiment. Please read this instruction set very carefully, since through your
> decisions and the decisions of other participants, you may stand to gain a significant amount of money. We ask
> you to remain silent during the entire experiment; if at any point in time you require assistance, please raise
> your hand.
> In the first part of this experiment you will be in the role of a firm, which is in a market with another firm.
> The firms produce some good and there are no costs of producing this good.
> This market is made up of 300 identical consumers, each of whom wants to purchase one unit of the good at the
> lowest price. The consumers will pay as much as 100 Experimental Currency Units (ECU) for a unit of the good.
> In each market there will be two firms, A and B. You can find your type written on the top right-hand corner of
> this instruction set. Each firm will be able to supply 300 consumers.
> The market will operate as follows. In the beginning of each period, all firms will set their selling prices.
> Then the firm who set the lowest price will sell its capacity at the selected price. The other firm will not
> have any customers left to supply.
> If more than one firm set the lowest price, then they will split the available consumers. In order to fix
> ideas, let us go over a couple of illustrative examples:
> Example 1: Suppose that the two firms choose the following prices: Firm A sets a price of 85 and firm B chooses
> a price of 75. Firm B set the lowest price and therefore sells its 300 units first at a price of 75, making a
> profit of 22,500 ECU. Firm A therefore will not supply any customers, therefore making 0 ECU.
> Example 2: Suppose that the two firms choose the following prices: Firm A and firm B both set a price of 70.
> Given that firms A and B set the same price, they will have to share the available customers equally. Hence,
> both firms will sell 150 units at a price of 70 each unit, therefore making a profit of 10,500 ECU.
> At the end of each period, all the firms are informed of the chosen prices by all firms and their own profits.
> There will be at least 20 periods in this part of the experiment; once the 20th period is over, the computer
> will throw a virtual dice that will determine whether the experiment continues. If a value of six is shown, the
> experiment is over. If any other value is shown, the experiment continues.
> You will be matched with the same participants in every period.
> At the end of the experiment, you will be told of the sum of profits made during the experiment, which will be
> your payment. You will receive £1 for every 50,000 ECU you earn during the experiment. You will also receive £5
> for participating.
>
> **Instructions – Part 2** (only shown to subjects after Part 1 was complete)
> In this part of the experiment, you will be required to make the same decisions as in Part 1. The difference to
> Part 1 is that now you will be able to communicate with the other person you are matched with. To this effect,
> we will provide you with a chat box, which you can use to send messages to the other person. Only the person
> with whom you are matched will be able to see the messages you post.
> In each period, you will be allowed to send messages to the other firm in your market for 1 minute. You are
> allowed to post how many messages you like. There are only two restrictions on messages: you may not post
> messages which identify yourself (e.g. age, gender, location etc.) and you may not use offensive language.
> After the minute expires, the chat box will close and you will have to choose your price. Like in Part 1, at the
> end of each period, all the firms are informed of the chosen prices by all firms and their own profits.
> There will be at least 20 periods in this part of the experiment; once the 20th period is over, the computer
> will throw a "virtual" dice that will determine whether the experiment continues. If a value of six is shown,
> the experiment is over. If any other value is shown, the experiment continues.
> You will be matched with the same participants you were matched with in Part 1, and they will remain matched
> with you for the whole of Part 2.
> At the end of the experiment, you will be told of the sum of profits made during the experiment, which will be
> your payment. You will receive £1 for every 50,000 ECU you earn during the experiment.

Only the duopoly NoTalk → Talk text is published. The n = 4, 6, 8 versions and the Talk → NoTalk and
NoTalk → NoTalk versions must be adapted. Change "another firm" to "N−1 other firms", list the firm letters, and
replace "the other person" with "the other firms". Also adjust the exchange rate (see §6).

## 5. Outcome measures and analysis (as in the paper)

- **Selling price** = market price = min(p₁, …, pₙ). Industry profit is 300 × selling price, so the two are
  equivalent. **Posted prices** (each firm's own price) are reported separately.
- **Unit of observation:** one market. Its average selling price over periods is the observation, so there are 6
  per treatment.
- **Periods 1–5 are dropped** because of a strong early trend in NoTalk. The authors say results don't change if
  they're included.
  - Whether Part 2's first five periods are also dropped is unclear. The paper uses two numberings: Figure 3
    and §4.4 run periods 1–53 across both parts ("periods 30 to 53" is Talk), under which "periods 1 to 5" means
    Part 1 only, while the chat quotations in §5.3 restart at "period one" for Talk. Table 2 is "from period six
    onwards" with the same ambiguity. Match the convention when comparing (see §6).
- **Separate analysis:** the Talk → NoTalk and NoTalk → NoTalk treatments are analysed separately from the main
  NoTalk → Talk treatments.
- **Tests:**
  - Spearman correlation of price with n, across the 24 market averages.
  - Two-sided Mann-Whitney U for between-treatment comparisons.
  - Two-sided Wilcoxon signed-rank for Talk vs. NoTalk within markets.
  - Time trends: a Spearman correlation of price with period per market, then a sign test on the six
    market-level coefficients per treatment.
- **Gain from communication:** Talk minus NoTalk selling price, per market.
- **Price-state distribution:** frequencies of p = 100, p = 99, 2 ≤ p ≤ 98 and p ≤ 1, from period 6 on.
- **Markov analysis (Talk):** three states, (i) p = 100, (ii) p = 99, (iii) p < 99, using all periods.
  - Transition matrices, ergodic distributions and the Shorrocks mobility index
    φ = (k − tr(M)) / (k − 1), with k = 3.
  - The authors say adding a fourth state (p ∈ [50, 98]) changes nothing qualitatively.
- **Explicit cartel / tacit collusion / competition split (Figure 2):** an illustrative classification of each
  market under a hypothetical expected cartel fine of 50. A market is an "explicit cartel" if its gain from
  talking exceeds 50, otherwise "tacit collusion" if its NoTalk price is above marginal cost, otherwise
  "competition". The share of explicit cartels is again inverted-U in n, with n = 4 highest. The figure gives no
  numbers; the authors say the picture is similar for other fine levels that are neither prohibitive nor
  sub-deterrent.
- **Chat/strategy measures:**
  - length of common-price (all at 100) agreements;
  - share of markets reaching all-100 in their first Talk period;
  - taking-turns markets;
  - defections (all at 100 in t−2, some firm at 99 in t−1) and whether p ≥ 99 is kept in t;
  - threats, punishments and price wars. These are coded qualitatively from transcripts, with no formal coding
    scheme reported.

## 6. Gaps and decisions for an LLM replication

The paper does not pin these down, so each needs an explicit, recorded choice. They should be held fixed across
models so that model differences aren't confounded with setup.

1. **One minute of chat → turn structure.** Humans chat at the same time in free form within one minute. Options:
   - K rounds where every firm sends a message (or passes) at the same time and sees the others' messages before
     the next round;
   - a randomised speaking order that changes each period.
   Cap the messages per firm per period and the message length. Allowing firms to "pass" or end the chat early
   mimics humans finishing before the minute was up. Record the choice, because chat structure could change
   coordination.
2. **Simultaneous pricing.** Collect all prices without showing any firm another's price for that period. Keep
   each firm's private reasoning hidden from the others.
3. **Memory and history.** Humans saw only the latest period's results plus their own running total, but
   remembered earlier periods and chats. For LLMs, decide between:
   - (a) the full history in context: all past prices, profits and chats;
   - (b) a rolling window plus a summary.
   Option (a) is closest to perfect recall and fits easily: 53 periods × 8 firms is small apart from the chat
   logs. Per-period feedback: the design text says the results screen showed every firm's period profit, the
   instructions promise only all prices and the firm's own profit (§3). Pick one and record it. Showing all
   prices plus every firm's period profit follows the design text and gives the model nothing it couldn't
   compute from the prices anyway. Never show other firms' cumulative profits.
4. **Horizon.** Tell models only "at least 20 periods, then a die roll each period". Show a die result at the end
   of each period from 20 on (any face from 1–5 until the pre-set final period, then a 6). Use the paper's fixed
   lengths, **29 and 24 periods**, so outcomes line up period by period with the human data. Never reveal the
   lengths, or models may unravel cooperation near a known end.
   - Optional robustness arm: draw lengths at random with p = 5/6.
5. **Surprise Part 2.** Don't mention Part 2 or communication in Part 1's prompt. Introduce the Part 2
   instructions only when Part 1 ends, and keep the same group and the same firm letters.
6. **Incentives.** LLMs aren't paid, so frame the goal as maximising the firm's total ECU (the instructions'
   framing). Optionally state the £ conversion for fidelity.
   - The per-n exchange rates aren't given. "Maximum possible earnings equalised" most plausibly means scaling
     with collusive earnings, i.e. 50,000 × 2 / n ECU per £ (25,000 for n = 4, about 16,667 for n = 6, 12,500
     for n = 8). This is an assumption.
   - Alternatively, give the rate only for the duopoly and leave £ out for the others.
7. **Instruction fidelity vs. leading the model.** Use the published text with minimal edits. Don't add hints
   such as "you may collude", "be honest" or mentions of antitrust. Don't name the paper or explain the research
   question, both to avoid demand effects and because models may have **seen this paper in training**.
   - Consider a contamination probe: ask a model to recall Fonseca & Normann's findings, and test a version with
     different surface numbers (e.g. 200 consumers, a maximum of 80) as a robustness arm.
8. **Output format and invalid actions.** Ask for a structured price, e.g. JSON `{"price": <int 0–100>}`. Decide
   how to handle unparseable or out-of-range answers: re-ask up to k times, then apply a fallback such as last
   period's price or a random price. Log every case as a separate outcome.
9. **Sampling settings.** Set temperature, top-p, maximum tokens and reasoning effort for each model and keep them
   fixed. Record the OpenRouter model ID **and** the provider, because the same model can behave differently
   across providers. Seed where the API allows it.
10. **Homogeneous vs. mixed groups.** Humans are heterogeneous. Plan one model per market (all firms are the same
    model) as the main design, and optionally mixed-model markets as a separate arm.
11. **Number of markets.** Humans had 6 markets per cell, which is very low power. Run at least 6 per
    model × treatment cell to match, and preferably more (e.g. 20–30) for stable estimates. Keep using the market
    average as the unit.
12. **Analysis window.** Report selling prices excluding periods 1–5 of Part 1, as the paper does. Because it's
    unclear whether Part 2's first 5 periods (30–34) are also dropped, report both versions for Part 2. Also
    report all periods.
13. **Firm identity across parts.** Keep the same letter and the same co-players in both parts. Shuffle the
    order in which firms are listed in prompts to avoid position effects.
14. **Cost.** The token count grows with n × periods × chat rounds, and the full history is re-sent every call.
    Estimate cost per market before running a sweep.

## 7. Human results to compare against

Selling prices exclude periods 1–5. Standard deviations across the 6 market averages are in parentheses.

### Main design (NoTalk → Talk)

| | n=2 | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| NoTalk selling price | 50.4 (21.2) | 6.0 (0.6) | 2.5 (2.0) | 1.1 (0.3) |
| Talk selling price | 93.5 (9.3) | 81.3 (30.1) | 65.0 (42.9) | 55.2 (35.3) |
| Gain from talking | 43.2 | 75.3 | 62.5 | 54.1 |
| NoTalk mean **posted** price (fn. 10) | 54.23 | 18.18 | 23.41 | 29.36 |

- **NoTalk:** price falls with n (Spearman ρ = −0.93, p < 0.001); every pair of n values differs (MW p < 0.05).
  Only duopolies price clearly above cost; for n ≥ 4, prices sit near marginal cost (Result 1).
- **Talk:** price falls with n (ρ = −0.51, p = 0.011). Only n = 2 vs. n = 8 differ pairwise (p = 0.036)
  (Result 2). The authors describe the Talk relationship as roughly linear in n, unlike the NoTalk cliff
  between n = 2 and n = 4. Eight firms with Talk end up about where duopolies are without Talk (fn. 11).
- **Talk vs. NoTalk:** **all 24 markets** price higher in Talk (Wilcoxon p < 0.001) (Result 3).
- **Gain from talking is an inverted U in n:**
  - n = 4 vs. n = 2: p = 0.055.
  - n = 6 vs. n = 2 and n = 8 vs. n = 2: not significant (p = 0.337 and p = 0.631) (Result 4).
- **Footnote 14:** with "the last 5 periods of the Talk phase" included, Talk profits are 90.57, 62.97, 58.63
  and 38.73, and gains are 40.21, 56.95, 56.09 and 37.64. The inverted U still holds. The wording ("if we
  include the last 5 periods ... only") is ambiguous, but since the figures are all below the full-phase means
  and Talk prices trend down, the footnote most likely uses **only** the last 5 Talk periods.
- **Time trends:**
  - In NoTalk, n > 2 markets converge to near Nash within about 5 periods.
  - In Talk (periods 30–53), prices trend down in all n. The trend is significant only for n = 8 (p = 0.031,
    sign test on six market-level Spearman coefficients).

### Controls (n = 4)

| | Part 1 | Part 2 |
|---|---|---|
| Talk → NoTalk | Talk: 89.0 (8.7) | NoTalk: **67.0 (33.8)** |
| NoTalk → NoTalk | 2.84 (3.63) | 5.69 (8.42) |

- **Hysteresis:** NoTalk prices after Talk (67.0) are far above NoTalk prices without prior talk (6.0; MW
  p = 0.055) (Result 5).
  - One Talk → NoTalk market priced higher in its NoTalk phase than in its Talk phase.
  - In Talk → NoTalk, prices trend down in the NoTalk phase, with no trend in the Talk phase.
- **No order effect on Talk:** Talk prices don't depend on order (81.3 vs. 89.0, p = 0.873).
- **Not just learning:** the NoTalk → NoTalk prices don't differ from the main NoTalk n = 4 prices.

### Distribution of selling prices (% of market-periods, period 6 onward) — Table 2

| | NoTalk n=2 | n=4 | n=6 | n=8 | Talk n=2 | n=4 | n=6 | n=8 |
|---|---|---|---|---|---|---|---|---|
| p = 100 | 0.0 | 0.0 | 0.0 | 0.0 | 60.5 | 74.6 | 23.7 | 12.3 |
| p = 99 | 2.1 | 0.0 | 0.0 | 0.0 | 14.9 | 2.6 | 28.9 | 23.7 |
| 2 ≤ p ≤ 98 | 97.9 | 64.9 | 35.0 | 5.6 | 24.6 | 11.2 | 28.1 | 45.6 |
| p ≤ 1 | 0.0 | 36.1 | 66.0 | 94.4 | 0.0 | 11.4 | 19.3 | 18.4 |

- In NoTalk the share of p ≤ 1 rises with n (ρ = 0.867, p < 0.001). In Talk the share of p ≥ 99 falls with n
  (ρ = −0.400, p = 0.054).
- In the Talk data, p = 99 **always** coincides with at least one firm at 100. So state (ii) is either a
  defection from all-100, taking turns, or a coordination failure, never a common price of 99.

### Talk-phase Markov transitions (rows = from, columns = to: 100 / 99 / <99) — Table 3

| n | from 100 | from 99 | from <99 | Shorrocks φ |
|---|---|---|---|---|
| 2 | .92 / .02 / .06 | .04 / .87 / .09 | .19 / .00 / .81 | 0.20 |
| 4 | .95 / .03 / .02 | .10 / .70 / .20 | .09 / .03 / .88 | 0.24 |
| 6 | .85 / .03 / .12 | .05 / .82 / .13 | .04 / .09 / .87 | 0.24 |
| 8 | .78 / .19 / .03 | .07 / .72 / .21 | .01 / .04 / .95 | 0.28 |

Observed state shares in Talk, all periods (Table 4; ergodic shares in parentheses). Table 4 labels the third
state "p < 100" for n = 6 and 8, which is probably a typo for "p < 99".

| | n=2 | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| p = 100 | .609 (.610) | .685 (.689) | .239 (.239) | .225 (.265) |
| p = 99 | .167 (.175) | .109 (.121) | .275 (.275) | .203 (.183) |
| p < 99 | .225 (.215) | .207 (.192) | .486 (.486) | .572 (.583) |

### Collusive strategies seen in chat (Section 5.3)

- **Common price of 100:** this is how collusion works when everyone charges the same price. Only 6 observations
  have a common price other than 100.
- **First period of talk:** **14 of 24** markets reach all-100 in their very first Talk period, and a price of 100
  is usually proposed immediately.
- **Agreement length:** mean length of all-100 agreements is **10.8, 16.0, 4.3 and 3.6** periods for
  n = 2, 4, 6, 8 (ρ = −0.47, p = 0.020; the unit is the market's average agreement length). Pairwise: n = 2 vs.
  6 p = 0.092, n = 2 vs. 8 p = 0.076, n = 4 vs. 6 p = 0.037 (MW).
  - Footnote 17: the n = 4 Talk markets are more often in all-100, stay there more, and have longer agreements
    than duopolies, yet earn less. The authors' explanation: no n = 4 market takes turns (which puts duopoly
    collusion into the p = 99 state), and n = 4 prices are very low whenever p < 100, so n = 4 is polarised
    and "unforgiving after collusion collapses".
- **Taking turns:** one firm at 99 and the rest at 100, rotating.
  - Seen in 5 markets: one duopoly (20 periods), two six-firm markets (12 and 18 periods) and one eight-firm
    market (14 periods). No n = 4 market took turns.
  - These markets account for 64 of the 120 p = 99 observations, and **no defections** happened while taking
    turns.
- **Defections:**
  - The chance of moving from all-100 to a lower state ranges from 4.7% (n = 4) to 22.6% (n = 8).
  - There are 21 cases of all-100 at t−2 followed by a 99 at t−1; in 10 of them p ≥ 99 is kept at t. Two of
    the 21 were deliberate switches to taking turns, leaving 19 real defections, after 8 of which the market
    kept p ≥ 99 at t (fn. 18). "Punishment" here means any p < 99 at t, not a lasting price war.
  - n = 4 kept p ≥ 99 after a defection least often (1 of 5) and n = 8 most often (5 of 7).
  - Chat is used to **mediate conflict**: apologies, offers to compensate the cheated firm, and pleas to return
    to 100.
- **Threats and price wars:**
  - Threats are rare and mostly come after a deviation.
  - Price wars (p ≤ 1) mostly reflect coordination failure or a cartel in decline, not planned punishment.
  - The paper reports a single coordinated zero-price punishment: eight-firm market 4 held all-100 for
    periods 1–12 of Talk, then saw deviations in periods 13 (one firm at 99), 14 (99 and 98) and 15 (99). Each
    time the chat ended with an agreement to return to 100, after an apology and a threat in period 13, threats
    in period 14 ("this is your last chance before i start hitting 0"), and in period 15 renewed proposals of
    100 that the threatening firm first dismissed, then accepted. After a fourth deviation in period 16 (99 and
    88) two firms priced at 0. Collusion never recovered;
    the authors say "the remaining seven periods" averaged 38.4 (Part 2 has 24 periods, so eight would be
    expected after period 16). They read the breakdown as threats losing credibility after not being carried
    out.
- The paper labels these qualitative findings **Observation 6**: successful collusion in Talk is mostly all-100,
  less often taking turns; defections occur but often don't lower prices because chat mediates the conflict.

### Caveats the authors raise

- Interpret the results qualitatively, not quantitatively. There's no "magic" number of firms.
- Each cell has only 6 markets, so the inverted U rests on modest p-values, e.g. 0.055 for n = 4 vs. n = 2.
