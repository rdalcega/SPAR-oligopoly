"""Analysis of LLM oligopoly runs against Fonseca & Normann (2012).

Modules:
    load      market JSON records -> tidy tables (firm-period, market-period, chat messages, markets)
    measures  paper measures: windows, selling-price means, Table 2 bins, Markov (Tables 3-4), §5.3 strategy
              measures, time series, grim-trigger threshold
    stats     nonparametric tests with markets as units, hysteresis comparison
    human     side-by-side comparison with the human benchmarks (`oligopoly.benchmarks`)
    chat      rule-based coding of chat messages and Talk periods (agreements kept/broken, threats, punishment)
    notes     reading a firm's private notes for this period's stated own price and explicit deviation plans
    mistakes  invalid output rates and heuristic flags for mistakes and misread rules
    probe     post-game comprehension probe: per-firm answers and per-cell accuracy
    report    builds everything and writes CSV/JSON tables plus summary.json
"""
