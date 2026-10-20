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

None recorded yet. Each §6 item (1–14) needs an entry here before the code that depends on it is written.
