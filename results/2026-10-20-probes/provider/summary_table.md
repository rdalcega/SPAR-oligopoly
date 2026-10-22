| model | provider | med rtok (mean, range) | low / high / default rtok | disabled | max ctok seen | JSON fail (stop calls) | modes | med latency s | $/price call (med) | same-seed repro (content / reasoning) | cached on repeat | 429 attempts | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| google/gemma-4-26b-a4b-it | deepinfra/fp8 | 646 (415–934) | 641 / 711 / 0 | rtok=0 | 1208 | 0/13 | empty,fenced,strict | 7.4 | 0.00037 | True / False | 0 | 0 | - |
| google/gemma-4-26b-a4b-it | google-vertex/global | 812 (783–841) | - / - / - | rtok=0 | 1117 | 0/3 | strict | 7.5 | 0.00083 | True / False | 1067 | 0 | - |
| ibm-granite/granite-4.2-8b | coreweave/bf16 | 76 (69–83) | - / - / - | rtok=0 | 91 | 0/3 | strict | 1.4 | 0.00009 | True / False | 1056 | 0 | - |
| ibm-granite/granite-4.2-8b | deepinfra/bf16 | 25 (8–43) | 18 / 1178 / 1652 | rtok=0 | 2188 | 0/14 | strict | 5.4 | 0.00003 | True / False | 1056 | 0 | - |
| liquid/lfm-2.5-24b-a2b | liquid/fp8 | 4927 (2467–7366) | 5916 / 7478 / 4049 | OpenRouter error 400: {'message': 'Reaso | 7562 | 0/9 | empty,strict | 24.2 | 0.00392 | False / False | 0 | 0 | price_disabled_0: OpenRouter error 400: {'message': 'Reasoning is mandatory fo; price_disabled_1: OpenRouter error 400: {'message': 'Reasoning is mandatory fo |
| openai/gpt-oss-20b | deepinfra/bf16 | 807 (383–1110) | 8 / 4230 / 1488 | OpenRouter error 400: {'message': 'Reaso | 4317 | 0/11 | empty,strict | 8.1 | 0.00015 | True / False | 0 | 0 | price_disabled_0: OpenRouter error 400: {'message': 'Reasoning is mandatory fo; price_disabled_1: OpenRouter error 400: {'message': 'Reasoning is mandatory fo |
| openai/gpt-oss-20b | groq | 1088 (812–1364) | - / - / - | OpenRouter error 400: {'message': 'Reaso | 1379 | 0/2 | strict | 1.9 | 0.00041 | True / False | 0 | 0 | alt_price_disabled: OpenRouter error 400: {'message': 'Reasoning is mandatory fo |
| qwen/qwen3.5-27b | alibaba | 4492 (3104–5880) | - / - / - | rtok=0 | 5892 | 0/3 | strict | 58.8 | 0.00724 | True / False | 0 | 0 | - |
| qwen/qwen3.5-27b | deepinfra/fp8 | 2608 (500–3897) | 4198 / 2987 / 3720 | rtok=0 | 5173 | 0/13 | empty,strict | 46.8 | 0.00863 | True / False | 0 | 0 | - |
| qwen/qwen3.5-9b | deepinfra/bf16 | 3766 (483–6149) | 4877 / 3516 / 3682 | rtok=0 | 7335 | 0/12 | empty,strict | 220.0 | 0.00080 | True / False | 0 | 20 | price_disabled_1: 6 attempts failed; last error: HTTP 429: {"message": "Provid |
| qwen/qwen3.5-9b | parasail/bf16 | 6330 (6330–6330) | - / - / - | 6 attempts failed; last error: HTTP 429: | 6340 | 0/1 | strict | 74.3 | 0.00169 | - | - | 14 | alt_price_disabled: 6 attempts failed; last error: HTTP 429: {"message": "Provid; alt_price_medium_0: 6 attempts failed; last error: HTTP 429: {"message": "Provid |
| qwen/qwen3.6-27b | alibaba | 1860 (1598–2123) | - / - / - | rtok=0 | 2135 | 0/3 | strict | 33.9 | 0.00555 | True / False | 0 | 0 | - |
| qwen/qwen3.6-27b | deepinfra/fp8 | 1752 (1267–2638) | 1991 / 1235 / 2036 | rtok=0 | 2990 | 0/13 | empty,strict | 25.8 | 0.00689 | True / False | 0 | 0 | - |
| qwen/qwen3.8-27b | alibaba | 590 (415–764) | - / - / - | rtok=0 | 775 | 0/3 | strict | 21.2 | 0.00200 | False / False | 0 | 0 | - |
| qwen/qwen3.8-27b | deepinfra/bf16 | 746 (296–1257) | 510 / 382 / 4743 | rtok=0 | 6671 | 1/13 | empty,strict | 52.4 | 0.00206 | False / False | 0 | 0 | - |

Total charged across all logged attempts: $0.3704

Chat messages (chat_medium*):
- google/gemma-4-26b-a4b-it [deepinfra/fp8] chat_medium (strict, 218 chars, rtok=1038): 'Hello everyone. In Part 1, we saw a race to the bottom that minimized profits for everyone. I suggest we try to coordinate on a higher price (e.g., 85) to maximize our individual profits instead. What do you all think?'
- google/gemma-4-26b-a4b-it [deepinfra/fp8] chat_medium_b (strict, 253 chars, rtok=1034): 'Hi everyone. In Part 1, we were driving prices down, which minimized our profits. If we coordinate and set higher prices, we can all make much more. Would you be open to discussing a higher price level for this period to increase our collective profits?'
- ibm-granite/granite-4.2-8b [deepinfra/bf16] chat_medium (strict, 160 chars, rtok=11): "Hello firms A, C, D. I am firm B. Let's coordinate to avoid undercutting. Suggest we set prices around 70 for fairness. Please share your preferred price range."
- ibm-granite/granite-4.2-8b [deepinfra/bf16] chat_medium_b (strict, 101 chars, rtok=29): "Let's coordinate to set price 70 this period to avoid undercutting and maximize our shares when tied."
- liquid/lfm-2.5-24b-a2b [liquid/fp8] chat_medium (strict, 308 chars, rtok=3250): "Hello, let's try to cooperate. I suggest we all set price 70 in every period. If anyone sets a lower price, they will capture all demand. To ensure fairness, if someone undercuts, we will all set a price just below theirs in the next period, so they get nothing. This way we all earn 7000 per period. Agreed?"
- liquid/lfm-2.5-24b-a2b [liquid/fp8] chat_medium_b (strict, 205 chars, rtok=4344): "Let's cooperate to maximize profits. Agree to set a price of 80 in every period. If anyone sets below 80, we will set 79 next period to undercut. This way, all of us can earn 75*80=6000 per period. Agreed?"
- openai/gpt-oss-20b [deepinfra/bf16] chat_medium (strict, 227 chars, rtok=511): "Hi all, I'm Firm B. I'd like to propose a uniform price of 70 for this period. This gives each of us 21,000 ECU. If anyone prices lower, we get the market but later periods could be shorter. Let's coordinate for mutual benefit."
- openai/gpt-oss-20b [deepinfra/bf16] chat_medium_b (strict, 217 chars, rtok=1342): 'Let’s try a simple coop: agree on a common price (e.g., 70) and split customers equally. If someone deviates, we’ll re‑start an agreed‑price round. That keeps the price stable and gives everyone a share of the profit.'
- qwen/qwen3.5-27b [deepinfra/fp8] chat_medium (strict, 125 chars, rtok=1315): "Let's coordinate to avoid Part 1 losses. I propose all firms set price to 100. We split demand equally for max profit. Agree?"
- qwen/qwen3.5-27b [deepinfra/fp8] chat_medium_b (strict, 215 chars, rtok=2959): "Hi A, C, D. In Part 1, low prices meant low profits. Let's collaborate by choosing price 100. When prices tie, we split sales equally. This maximizes joint profit. I commit to 100 if you do. Let's stop undercutting."
- qwen/qwen3.5-9b [deepinfra/bf16] chat_medium (strict, 209 chars, rtok=2721): "Let's try to improve our situation after the losses in Part 1. I propose we all set our price to 100 ECU this period. This ensures high profits for everyone. I commit to 100 if you follow suit. Please respond."
- qwen/qwen3.5-9b [deepinfra/bf16] chat_medium_b (strict, 235 chars, rtok=2091): "Part 1 was costly for everyone due to price undercutting. Let's coordinate to set a mutually profitable price, like 85 ECU or higher. By avoiding undercutting, we all secure positive profits. I propose we aim for stability this period."
- qwen/qwen3.6-27b [deepinfra/fp8] chat_medium (strict, 258 chars, rtok=1430): "Let's all set price 100. We'll split the 300 consumers equally (75 each), earning 7,500 ECU/period. This maximizes joint profits. Undercutting triggers a price war, dropping earnings to near zero. Cooperation benefits everyone. Please confirm by setting 100."
- qwen/qwen3.6-27b [deepinfra/fp8] chat_medium_b (strict, 258 chars, rtok=1943): 'Hi everyone. I propose we all set price=100 each period. With equal capacity (300), tying at 100 splits the market 4 ways, yielding 7,500 ECU each. This avoids destructive price wars and maximizes our joint profits. I commit to 100 if you do. Please confirm.'
- qwen/qwen3.8-27b [deepinfra/bf16] chat_medium (strict, 244 chars, rtok=391): 'I propose we all set our price to 100 every period. If we all match at 100, each of us sells 75 units and earns 7,500 ECU per period. This is far better than undercutting each other. I will set 100 this period. Can we all agree to match at 100?'
- qwen/qwen3.8-27b [deepinfra/bf16] chat_medium_b (strict, 173 chars, rtok=468): 'I suggest we all set our price at 100 this period. If all four of us match at 100, we each sell 75 units and earn 7,500 ECU. High prices benefit everyone. What do you think?'

Long context (long_60k):
- google/gemma-4-26b-a4b-it [deepinfra/fp8]: prompt_tokens=65199 finish=stop json=strict value=89 latency=23.39s $0.0049
- ibm-granite/granite-4.2-8b [deepinfra/bf16]: prompt_tokens=66619 finish=stop json=strict value=95 latency=7.03s $0.0040
- liquid/lfm-2.5-24b-a2b [liquid/fp8]: prompt_tokens=67355 finish=stop json=strict value=95 latency=5.39s $0.0039
- openai/gpt-oss-20b [deepinfra/bf16]: prompt_tokens=64951 finish=stop json=strict value=95 latency=2.31s $0.0020
- qwen/qwen3.5-27b [deepinfra/fp8]: prompt_tokens=68585 finish=stop json=strict value=90 latency=41.56s $0.0241
- qwen/qwen3.5-9b [deepinfra/bf16]: prompt_tokens=68585 finish=stop json=strict value=97 latency=32.02s $0.0070
- qwen/qwen3.6-27b [deepinfra/fp8]: prompt_tokens=68585 finish=stop json=strict value=90 latency=39.61s $0.0297
- qwen/qwen3.8-27b [deepinfra/bf16]: prompt_tokens=69287 finish=stop json=strict value=97 latency=7.89s $0.0106
