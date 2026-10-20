# SPAR-oligopoly

This repo runs experiments closely inspired by Fonseca & Normann (2012), "Explicit vs. tacit collusion — The
impact of communication in oligopoly experiments", with LLMs playing the firms. The paper uses repeated Bertrand
markets with n ∈ {2, 4, 6, 8} firms, NoTalk and Talk (chat) phases, and two n = 4 control orderings. We aren't
limited to that design: runs can use more than 8 firms, and the control orderings can be run for any n. Keep
such extensions separate from the comparison with the human results.

It is part of the SPAR project *Learning dynamics in competitive multiagent environments* (pilot in
`~/Documents/code/SPAR-pilot-run`). That project asks how traits such as honesty, cooperation and collusion
change when competing actors train their own models on economically useful work. These markets add a simple
competitive setting with a human benchmark, where collusion can be measured both tacitly and through chat.

## Goals (equal weight)

- **LLMs vs. humans:** run the design and compare outcomes with the human results.
- **LLMs vs. LLMs:** compare how different models price, collude and communicate under an identical setup.

In both, measure the traits the SPAR project cares about: collusion with and without talk, whether firms keep or
break agreements they made in chat, and threats and punishment. Tell deliberate choices apart from mistakes
(e.g. an invalid price or a misread rule).

## Spec

- `fonseca-normann-summary.md` is the spec. §1–5 cover the design, §6 the open decisions for an LLM run, and
  §7 the human benchmarks.
- **Before writing any code, the whole summary must be checked against the paper and signed off.** Use the
  working paper (DICE DP 65) and the journal version where it's available.
- Record the sign-off and every choice from §6 (and any others) in `DECISIONS.md` before implementing it.
  Hold those choices fixed across models.
- Under grim trigger, full collusion (every firm at 100) is an equilibrium roughly when δ ≥ (n − 1)/n. With
  integer prices the best deviation is to 99, so the exact condition is δ ≥ 1 − 100/(99n). At the paper's
  δ = 5/6 it holds for n ≤ 6 and fails for n = 8. After choosing δ, run several n on each side of the
  threshold, avoiding n right at it.
- In prompts to firms, never reveal the pre-drawn part lengths (do state the minimum of 20 periods and the die
  rule), never describe Part 2 or communication during Part 1, never name the paper, and never tell firms how
  to behave on a trait we measure (e.g. to collude, to be honest, or that antitrust applies). Separate probes,
  such as the §6 contamination check, may name the paper.

## Stack

- Python, managed with `uv`.
- All models are called through OpenRouter. `OPENROUTER_API_KEY` is in `.env`, which is gitignored and must
  never be committed.
- Pin the OpenRouter provider with fallbacks off. For every call, log the model ID, the provider that served
  it, the sampling settings and the seed, and flag calls where the seed may not have been honoured.
- Ask before any run that calls a paid model, and set a cost cap for it.
