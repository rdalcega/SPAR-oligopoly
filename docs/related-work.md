# LLM oligopoly and collusion research: what exists before replicating Fonseca & Normann

**Question:** has anyone recreated Fonseca & Normann (2012) with LLMs? (See `fonseca-normann-summary.md` for the
original design and human results.)

**Short answer:** no direct replication was found as of October 2026. No study we found combines F&N's four
features:
- a homogeneous good where the lowest price wins;
- n = 2, 4, 6, 8;
- a surprise NoTalk → Talk switch with free-form chat;
- the Talk → NoTalk and NoTalk → NoTalk controls.

None of the LLM papers below cite F&N. Related work covers parts of the design: communication vs. none (mostly
duopolies) and number of firms (tacit collusion only).

**Method and limits:** web searches (arXiv, SSRN, RePEc, Google-style queries) on 2026-10-19, plus reading the
abstract or relevant sections of each paper.
- Guinea Pig Trials, Lee & Park and J.Y. Lee were read in their relevant sections.
- Agrawal et al. and Keppo et al. were read only at abstract or summary level, and Keppo et al.'s per-n numbers
  come from one results table.
- A search can miss work, so "no replication found" is not proof that none exists.

---

## 1. Communication vs. no communication

| Paper | Market | Firms | Models | Chat design | Result |
|---|---|---|---|---|---|
| Han, Wu & Xiao, "Guinea Pig Trials" (arXiv 2308.10974, 2023/24) | Bertrand, differentiated goods | 2 | GPT-4 | Up to 3 open-ended messages per firm per round, then price | No chat: price ≈ 7 (Nash 6, cartel 8). Chat: explicit price agreements within the first 20 rounds; price climbs to almost 8 by about round 1,000 |
| Wu et al., "Shall We Team Up" (EMNLP Findings 2024, arXiv 2402.12327) | Bertrand, differentiated goods (same group, similar setup) | 2 | GPT-4 family | Free-form chat between rounds | No chat: prices start rising only after about 200 rounds. Chat: explicit price agreements in the first 30 rounds. Authors say the pattern "aligns with human behavioral data" but don't compare with F&N |
| Agrawal et al., "Evaluating LLM Agent Collusion in Double Auctions" (ICML 2025, arXiv 2507.01413) | Continuous double auction (sellers value $80, buyers $100) | 5 sellers, 5 buyers | GPT-4.1, Claude 3.7 Sonnet, others | Sellers can message each other | Seller chat markedly increases collusion. Oversight reduces it; an authority figure pushing for profit increases it. Mixed-model groups are not always less collusive. Code: github.com/andyjliu/llm-agent-collusion |
| Neves & Bussmann, "Smart Agent-Based Modelling with LLMs" (Stanford Computational Antitrust, 2026) | Bertrand duopoly | 2 | LLM agents | Inter-agent communication | Communication amplifies emergent behaviours, including agents imitating concerns about collusion |

**Hysteresis analogue.** Guinea Pig Trials ran a version where chat is allowed for 400 rounds, then banned. After the
ban, prices stay at the level they had reached and stop fluctuating; they do not fall back to the no-chat
level. This is qualitatively like F&N's Talk → NoTalk result (n = 4: NoTalk price 67.0 after Talk vs. 6.0 without
prior Talk). It is a single duopoly run with differentiated goods, though.

## 2. Number of firms (tacit, no chat)

| Paper | Market | Firms | Models | Result |
|---|---|---|---|---|
| Fish, Gonczarowski & Shorrer, "Algorithmic Collusion by LLMs" (arXiv 2404.00806, revised 2026) | Bertrand, logit demand, 300 periods | 2 (mainly) | GPT-4 | LLM pricing agents reach supracompetitive prices without being told to collude. Small changes in prompt wording change how much they collude. Off-path analysis points to fear of price wars |
| Keppo, Li, Tsoukalas & Yuan, "On the Fragility of AI Agent Collusion" (arXiv 2603.20281, Mar 2026) | Stylised repeated pricing | 2–5 | Open-source (32B, 14B) plus Q-learning | More firms delay and weaken collusion; with 5 sellers, none emerged within 1,000 rounds. Price lift over competitive, from one table: about +22% (2 firms), +36% (3), +11% (4), +3% (5), so not monotonic. Differences in patience or data access reduce collusion; differences in model size do not |
| Lee & Park, "Faithful yet Collusive" (arXiv 2609.18346, Sept 2026) | Bertrand, logit demand, 300 rounds | 2, 3 | 9 models | Collusion score Δ (0 = Nash, 1 = full collusion), 2 → 3 firms: Sonnet 4.5 0.98 → 0.36, GPT-5 0.56 → 0.27, Haiku 4.5 0.37 → 0.19, Mistral 7B 0.18 → 0.10. Qwen, Llama and Gemma price below Nash. Chain-of-thought monitoring does not detect collusion |
| J.Y. Lee, "Who Leads and Who Collects" (arXiv 2610.11256, Oct 2026) | Bertrand, logit demand (Calvano et al.), 200 periods | 4 | Gemini 3.1 Flash-Lite, Claude Haiku 4.5, GPT-5-nano, DeepSeek | Homogeneous markets: Claude Δ = 0.79, Gemini 0.72, DeepSeek 0.24, GPT −0.15 (GPT prices above the monopoly level). A market with all four models is less collusive than single-model markets (−0.38, p = 0.02). Within mixed markets, DeepSeek earns the most and GPT the least |
| Lin et al., "Strategic Collusion of LLM Agents" (arXiv 2410.00031) | Multi-commodity Cournot | 2 | Several | LLM firms divide markets, each effectively taking over specific commodities |

## 3. Takeaways for an LLM replication of F&N

- **Direction matches humans:** chat raises LLM collusion, and more firms weaken tacit collusion.
- **Untested:** none of the following has been tested with LLMs:
  - F&N's near-competitive NoTalk prices for n ≥ 4 in a homogeneous-goods market;
  - the inverted U in the gain from talking (largest at n = 4);
  - the n = 4 controls, both Talk → NoTalk hysteresis and NoTalk → NoTalk repetition.
  
  A replication would add something new.
- **Design gap:** every LLM Bertrand study above uses differentiated goods (mostly logit demand) and runs for
  hundreds of rounds. F&N uses a homogeneous good where the lowest price wins, with a reservation price of 100,
  over 53 periods. Results from the logit-demand papers may not carry over, because in F&N undercutting by 1
  captures the whole market.
- **Model choice matters a lot:** Claude models are among the most collusive in two independent papers, and GPT-5
  variants are mixed. Report results separately by model and keep each market to one model in the main design, as
  `fonseca-normann-summary.md` §6.10 plans.
- **Prompt sensitivity:** Fish et al. show that small wording changes shift collusion. This supports using F&N's
  verbatim instructions and recording every prompt.
- **Contamination:** no paper probes whether models know F&N's results. Our contamination probe (§6.7 of the design
  summary; `DECISIONS.md` §6 item 7) is, as far as we found, new. It ran on 2026-10-20 for the 7 lineup models
  (`results/2026-10-20-probes/contamination/`): none recognised the instructions or recalled the design or
  results.

## Sources

- Han, Wu & Xiao, "Guinea Pig Trials": <https://arxiv.org/abs/2308.10974>
- Wu et al., "Shall We Team Up": <https://arxiv.org/abs/2402.12327>
- Agrawal et al., double auctions: <https://arxiv.org/abs/2507.01413>
- Fish, Gonczarowski & Shorrer: <https://arxiv.org/abs/2404.00806>
- Keppo et al., fragility: <https://arxiv.org/abs/2603.20281>
- Lee & Park, "Faithful yet Collusive": <https://arxiv.org/abs/2609.18346>
- J.Y. Lee, "Who Leads and Who Collects": <https://arxiv.org/abs/2610.11256>
- Lin et al., market division: <https://arxiv.org/abs/2410.00031>
- Neves & Bussmann (Stanford): <https://law.stanford.edu/wp-content/uploads/2026/04/Neves-Bussmann.pdf>
- Fonseca & Normann (2012), DICE DP 65: <https://d-nb.info/1026225752/34>
