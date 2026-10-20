import asyncio
import dataclasses
import json
import os
import re

import pytest

from oligopoly import engine, market
from oligopoly.config import ConfigError, ExperimentConfig, ModelConfig, load_models
from oligopoly.engine import MarketSpec, environment_seed, market_path, run_market
from oligopoly.llm import BudgetExceeded, RetriesExhausted
from oligopoly.mock import MockClient
from oligopoly.run import build_specs, main, plan, run_all

MOCK = ModelConfig("mock", "mock", "mock", temperature=1.0, max_tokens=100, reasoning={"effort": "medium"})
SHORT = ExperimentConfig(part_lengths=(6, 4))
COMM = ["chat", "communicat", "message", "talk"]


def spec(n=2, ordering="NT-T", idx=0, run_id="test"):
    return MarketSpec(run_id, MOCK, n, ordering, idx, environment_seed(0, n, ordering, idx))


async def play(s, tmp_path, exp=SHORT, client=None):
    client = client or MockClient(record=True)
    path = await run_market(s, client, tmp_path, exp)
    return json.loads(path.read_text()), client, path


def calls_by_firm(client):
    out = {}
    for c in client.calls:
        out.setdefault(c["tag"]["firm"], []).append(c)
    return out


def firm_decisions(rec, L):
    """Firm L's decisions in the order they were made: per period its chat rounds then its price; the probe last."""
    out = []
    for p in rec["periods"]:
        out += [r["decisions"][L] for r in p["chat"]] + [p["decisions"][L]]
    if rec.get("probe"):
        f = rec["probe"]["firms"][L]
        out.append({**f, "notes": None, "fallback": not f["valid"]})
    return out


def results_label(q, exp):
    """The results heading of global period q."""
    L1 = exp.part_lengths[0]
    return f"Results of period {q} (Part 1)" if q <= L1 else f"Results of period {q - L1} (Part 2)"


def check_contexts(rec, client, exp):
    """Every recorded request is the windowed context of DECISIONS.md §6 item 3 (2026-10-20): the instructions
    (Part 2's from Part 2 on), the memory block with the firm's latest notes and own cumulative profit, then the
    verbatim turns of the last `memory_window` completed periods and of the current period. Each request is
    rebuilt exactly from the record."""
    from oligopoly.llm import messages_sha256, reconstruct_request
    from oligopoly.prompts import memory_text

    W, L1 = exp.memory_window, exp.part_lengths[0]
    letters = sorted(rec["transcripts"])
    sent = {c["tag"]["call_id"]: c for c in client.calls}
    all_notes = {F: {d["notes"] for d in firm_decisions(rec, F) if d["notes"]} for F in letters}
    checked = 0
    for L in letters:
        tr, instr = rec["transcripts"][L], rec["instructions"][L]
        assert len(instr) == 2 and instr[1].startswith("Part 1 is over.")
        notes, profit = None, [0.0, 0.0]
        done = 0  # periods completed before the decision
        for d in firm_decisions(rec, L):
            for a in d["attempts"]:
                c, ctx = sent[a["call_id"]], a["context"]
                msgs, tag = c["messages"], c["tag"]
                if tag["phase"] == "probe":
                    assert (tag["part"], tag["period"]) == ("probe", None)
                    g = sum(exp.part_lengths) + 1
                else:
                    g = tag["period"] if tag["part"] == 1 else L1 + tag["period"]
                assert messages_sha256(msgs) == ctx["messages_sha256"] and ctx["merged"] is True
                assert reconstruct_request(rec, L, ctx) == msgs
                # roles alternate strictly: system, user, assistant, user, ..., user
                assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * ((len(msgs) - 2) // 2) + [
                    "user"]
                # instructions always present; Part 2's only from Part 2 on
                k = 2 if g > L1 else 1
                assert ctx["instructions"] == k and msgs[0] == {"role": "system", "content": instr[0]}
                # memory = the latest notes written before this decision, and the firm's own profit so far
                part = 1 if g <= L1 else 2
                mem = memory_text(exp.notes, notes, part, profit[part - 1], sum(profit) if part == 2 else None)
                assert ctx["memory"] == mem
                # the window: every turn of periods g - W .. g, verbatim, ending with this request's prompt
                lo, hi = ctx["window"]
                assert [m["period"] for m in tr[lo:hi]] == sorted(m["period"] for m in tr[lo:hi])
                assert all(g - W <= m["period"] <= g for m in tr[lo:hi])
                assert lo == 0 or tr[lo - 1]["period"] < g - W
                assert hi == len(tr) or tr[hi]["role"] == "assistant"
                assert tr[lo]["role"] == "user"
                # Part 2's instructions, the memory block and the window's opening turn form one user message
                opening = [*instr[1:k], mem, tr[lo]["content"]]
                assert msgs[1] == {"role": "user", "content": "\n\n".join(opening)}
                assert msgs[2:] == [{"role": m["role"], "content": m["content"]} for m in tr[lo + 1:hi]]
                assert msgs[-1]["role"] == "user" and tr[hi - 1]["period"] == g
                # results visible: those of the W windowed periods and the most recent; none older than g - W - 1
                text = "\n".join(m["content"] for m in tr[lo:hi])
                for q in range(max(1, g - W - 1), g):
                    assert results_label(q, exp) in text, (L, g, q)
                for q in range(1, g - W - 1):
                    assert results_label(q, exp) not in text, (L, g, q)
                # no other firm's notes
                blob = json.dumps(msgs, ensure_ascii=False)
                assert not any(x in blob for F, xs in all_notes.items() if F != L for x in xs)
                checked += 1
            if not d["fallback"] and d["notes"] is not None:
                notes = d["notes"]
            if d["attempts"] and d["attempts"][0]["call_id"].split("/")[-2] == "price":
                p = rec["periods"][done]
                profit[p["part"] - 1] += p["profits"][L]
                done += 1
    assert checked == len(client.calls)


@pytest.mark.parametrize("n", [2, 10])
@pytest.mark.parametrize("ordering", ["NT-T", "T-NT", "NT-NT"])
async def test_mock_market(tmp_path, n, ordering):
    rec, client, _ = await play(spec(n, ordering), tmp_path)
    talk = engine.ORDERINGS[ordering]
    letters = [chr(65 + i) for i in range(n)]

    # period counts and Talk only in the right parts
    assert [(p["part"], p["period"]) for p in rec["periods"]] == \
        [(1, t) for t in range(1, 7)] + [(2, t) for t in range(1, 5)]
    for p in rec["periods"]:
        assert bool(p["chat"]) == talk[p["part"] - 1] == p["talk"]
        assert len(p["chat"]) <= 3
        if len(p["chat"]) < 3 and p["chat"]:
            assert not p["chat"][-1]["messages"]  # closed early only when everyone passed
    for c in client.calls:
        assert c["tag"]["phase"] in ("price", "probe") or \
            c["tag"]["phase"].startswith("chat") == talk[c["tag"]["part"] - 1]

    # payoffs consistent with market.payoffs, totals consistent with periods
    for p in rec["periods"]:
        assert p["profits"] == market.payoffs(p["prices"])
    for L in letters:
        assert rec["totals"][L]["profit_total"] == pytest.approx(sum(p["profits"][L] for p in rec["periods"]))

    by_firm = calls_by_firm(client)
    for L, calls in by_firm.items():
        tr = rec["transcripts"][L]
        # Part 1 texts (every request of Part 1) never mention Part 2, nor communication in a NoTalk Part 1
        for c in calls:
            if c["tag"]["part"] == 1:
                for m in c["messages"]:
                    if m["role"] != "assistant":
                        low = m["content"].lower()
                        assert "part 2" not in low and "second part" not in low
                        if not talk[0]:
                            assert not any(w in low for w in COMM), m["content"]
        # the transcript keeps every turn, each request sent once
        assert sum(m["role"] == "user" for m in tr) == len(calls) + (rec["probe"] is None)
        assert all(m["role"] != "assistant" or "mock reasoning" not in m["content"] for m in tr)
    check_contexts(rec, client, SHORT)

    # simultaneity: no same-round message / same-period price is visible
    posted = {(p["part"], p["period"], r["round"]): r["messages"] for p in rec["periods"] for r in p["chat"]}
    for c in client.calls:
        t = c["tag"]
        text = "\n".join(m["content"] for m in c["messages"])
        if t["phase"] == "probe":  # after all play
            assert (t["part"], t["period"]) == ("probe", None) and "Results of period 4 (Part 2)" in text
        elif t["phase"] == "price":
            assert f"Results of period {t['period']} (Part {t['part']})" not in text
        else:
            for F, msg in posted[(t["part"], t["period"], int(t["phase"][4:]))].items():
                if F != t["firm"]:
                    assert msg not in text

    # notes never reach another firm
    for p in rec["periods"]:
        notes = [(L, x) for L, x in p["notes"].items()] + [
            (L, d["notes"]) for r in p["chat"] for L, d in r["decisions"].items()]
        for L, note in notes:
            if note:
                for F in letters:
                    if F != L:
                        assert note not in json.dumps(rec["transcripts"][F], ensure_ascii=False)

    # seeds: unique within the market and recorded
    seeds = [c["seed"] for c in client.calls]
    assert len(set(seeds)) == len(seeds) and all(0 <= s < 2**31 for s in seeds)
    assert rec["usage"]["calls"] == len(client.calls)


async def test_rerun_is_identical(tmp_path):
    _, c1, p1 = await play(spec(4, "NT-T"), tmp_path / "a")
    _, c2, p2 = await play(spec(4, "NT-T"), tmp_path / "b")
    assert p1.read_bytes() == p2.read_bytes()
    assert [c["seed"] for c in c1.calls if c["tag"]["firm"] == "A"] == \
        [c["seed"] for c in c2.calls if c["tag"]["firm"] == "A"]
    _, c3, _ = await play(spec(4, "NT-T", run_id="other"), tmp_path / "c")
    assert not {c["seed"] for c in c1.calls} & {c["seed"] for c in c3.calls}


class BadA(MockClient):
    """Firm A answers garbage in Part 1 period 3 (price) and Part 2 period 2 (chat); everyone else is valid."""

    def __init__(self):
        super().__init__(record=True, malformed_rate=0.0)

    async def chat(self, messages, *, tag=None, **kw):
        r = await super().chat(messages, tag=tag, **kw)
        if tag["firm"] == "A" and ((tag["part"], tag["period"], tag["phase"]) in {(1, 3, "price"), (2, 2, "chat1")}):
            r.content = "I refuse to answer in JSON."
            self.calls[-1]["content"] = r.content
        return r


async def test_fallbacks(tmp_path):
    rec, client, _ = await play(spec(2, "NT-T"), tmp_path, client=BadA())
    p13 = rec["periods"][2]
    d = p13["decisions"]["A"]
    assert d["fallback"] and not d["valid"] and d["reasks"] == 2 and len(d["errors"]) == 3
    assert p13["prices"]["A"] == rec["periods"][1]["prices"]["A"]
    assert rec["totals"]["A"]["price_fallbacks"] == 1 and rec["totals"]["A"]["chat_fallbacks"] == 1
    c = rec["periods"][7]["chat"][0]["decisions"]["A"]
    assert c["fallback"] and c["value"] is None and "A" not in rec["periods"][7]["chat"][0]["messages"]
    hist = rec["transcripts"]["A"]
    assert sum("could not be used: no JSON object" in m["content"] for m in hist) == 4  # 2 re-asks x 2
    assert sum(m["content"] == "I refuse to answer in JSON." for m in hist) == 6  # failed attempts kept
    assert any(f"your price for this period was set to {p13['prices']['A']}" in m["content"] for m in hist)
    attempts = [c for c in client.calls if c["tag"]["firm"] == "A" and c["tag"]["period"] == 3
                and c["tag"]["part"] == 1]
    assert [c["tag"]["attempt"] for c in attempts] == [0, 1, 2] and len({c["seed"] for c in attempts}) == 3


async def test_first_period_fallback_is_100(tmp_path):
    class Bad(MockClient):
        async def chat(self, messages, *, tag=None, **kw):
            r = await super().chat(messages, tag=tag, **kw)
            if tag["firm"] == "B" and tag["period"] == 1 and tag["part"] == 1:
                r.content = '{"price": 101}'
            return r

    rec, _, _ = await play(spec(2, "NT-NT"), tmp_path, client=Bad(malformed_rate=0.0))
    assert rec["periods"][0]["prices"]["B"] == 100 and rec["periods"][0]["decisions"]["B"]["fallback"]


async def test_full_horizon_die(tmp_path):
    rec, _, _ = await play(spec(2, "NT-NT"), tmp_path, exp=ExperimentConfig(part_lengths=(22, 21)))
    dies = [p["die"] for p in rec["periods"] if p["part"] == 1]
    assert dies[:19] == [None] * 19 and all(1 <= x <= 5 for x in dies[19:21]) and dies[21] == 6
    text = "\n".join(m["content"] for m in rec["transcripts"]["A"])
    assert "Period 20 is over. The computer threw the virtual dice" in text
    assert "Period 22 is over. The computer threw the virtual dice and it showed 6" in text
    assert "Period 21 is over. The computer threw the virtual dice and it showed 6" in text
    probe = next(m for m in rec["transcripts"]["A"] if "Please answer these questions" in m["content"])
    assert "The experiment is over. The sum of your firm's profits" in probe["content"]  # before the probe
    assert not {"21", "22"} & set(re.findall(r"(?<![\d,])\d+(?![\d,])", rec["instructions"]["A"][0]))


async def test_resume_and_failures(tmp_path):
    specs = build_specs("r", [MOCK], [2], ["NT-NT", "NT-T"], 2)
    client = MockClient()
    s1 = await run_all(specs, {"mock": client}, tmp_path, SHORT)
    assert len(s1["done"]) == 4 and all(market_path(tmp_path, s).exists() for s in specs)
    n_calls = len(client.calls)
    s2 = await run_all(specs, {"mock": client}, tmp_path, SHORT)
    assert len(s2["skipped"]) == 4 and len(client.calls) == n_calls

    class Failing(MockClient):
        def __init__(self, exc, key):
            super().__init__()
            self.exc, self.key = exc, key

        async def chat(self, messages, *, tag=None, **kw):
            if tag["market"] == self.key and tag["period"] == 2:
                raise self.exc("boom")
            return await super().chat(messages, tag=tag, **kw)

    specs = build_specs("f", [MOCK], [2], ["NT-NT"], 3)
    s3 = await run_all(specs, {"mock": Failing(RetriesExhausted, specs[1].key)}, tmp_path / "f", SHORT)
    assert len(s3["done"]) == 2 and s3["failed"] == [specs[1].key]
    fail = json.loads(engine.failure_path(tmp_path / "f", specs[1]).read_text())
    assert fail["error_type"] == "RetriesExhausted" and fail["periods_completed"] == 1

    s4 = await run_all(specs, {"mock": Failing(BudgetExceeded, specs[0].key)}, tmp_path / "b", SHORT,
                       max_concurrent_markets=1)
    assert s4["failed"] == [specs[0].key] and len(s4["not_started"]) == 2

    # waiting too long for budget room is not exhaustion: only that market fails (with a checkpoint to resume)
    from oligopoly.llm import BudgetWaitTimeout
    s5 = await run_all(specs, {"mock": Failing(BudgetWaitTimeout, specs[0].key)}, tmp_path / "w", SHORT,
                       max_concurrent_markets=1)
    assert s5["failed"] == [specs[0].key] and len(s5["done"]) == 2 and not s5["not_started"]
    fail = json.loads(engine.failure_path(tmp_path / "w", specs[0]).read_text())
    assert fail["reason"] == "budget_wait" and engine.checkpoint_path(tmp_path / "w", specs[0]).exists()


async def test_atomic_write(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    engine.write_json_atomic(path, {"a": 1})

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(engine.os, "fsync", boom)  # fails after the temporary file was written
    with pytest.raises(OSError):
        engine.write_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


PENDING_TOML = """
["pending"]
model = "x/pending"
provider = "TODO"
max_tokens = 100
"""


def test_models_config(tmp_path):
    models = load_models()
    assert "mock" in models and len(models) == 8
    live = [m for m in models.values() if not m.is_mock]
    assert all(not m.provider_pending and m.max_tokens for m in live)
    assert models["qwen3.5-27b"].chat_kwargs()["extra_body"] == {"top_k": 20, "presence_penalty": 1.5}
    (tmp_path / "m.toml").write_text(PENDING_TOML)
    with pytest.raises(ConfigError):
        load_models(tmp_path / "m.toml")["pending"].check_live()


async def test_cli(tmp_path, capsys):
    with pytest.raises(ConfigError, match="provider"):
        (tmp_path / "m.toml").write_text(PENDING_TOML)
        await main(["--run-id", "x", "--models", "pending", "--n", "2", "--budget-usd", "1",
                    "--models-file", str(tmp_path / "m.toml"), "--out-root", str(tmp_path)])
    args = ["--run-id", "cli", "--models", "mock", "--n", "2", "3", "--orderings", "T-NT",
            "--markets-per-cell", "1", "--part-lengths", "3,2", "--out-root", str(tmp_path)]
    assert await main(args) == 0
    run = tmp_path / "cli"
    assert len(list((run / "markets").glob("*.json"))) == 2 and (run / "calls.jsonl").exists()
    assert len(json.loads((run / "run.json").read_text())) == 1
    assert await main(args + ["--dry-run"]) == 0
    assert "Estimated cost" in capsys.readouterr().out


async def test_market_with_openrouter_client(tmp_path):
    """The engine's call kwargs fit the real client (HTTP mocked)."""
    import httpx
    import respx

    from oligopoly.llm import BASE_URL, Budget, OpenRouterClient

    cfg = ModelConfig("tiny", "acme/tiny", "deepinfra/fp8", temperature=0.6, top_p=0.95, top_k=20, max_tokens=64,
                      reasoning={"effort": "medium"})
    ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "pricing": {"prompt": "0.000001", "completion": "0"},
          "supported_parameters": ["seed", "top_k", "temperature"]}
    bodies = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        last = body["messages"][-1]["content"]
        content = '{"notes": "n", "price": 99}' if '"price": <integer' in last.splitlines()[-1] else '{"notes": "n", "message": null}'
        return httpx.Response(200, json={"id": "g", "model": "acme/tiny", "provider": "DeepInfra",
                                         "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0}})

    with respx.mock() as r:
        r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
            return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
        r.post(f"{BASE_URL}/chat/completions").mock(side_effect=reply)
        async with OpenRouterClient(tmp_path / "calls.jsonl", Budget(1.0), api_key="k") as client:
            s = MarketSpec("live", cfg, 2, "NT-T", 0, 1)
            path = await run_market(s, client, tmp_path, ExperimentConfig(part_lengths=(2, 1), post_game_probe=False))
    rec = json.loads(path.read_text())
    assert [p["prices"] for p in rec["periods"]] == [{"A": 99, "B": 99}] * 3
    b = bodies[0]
    assert b["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert b["top_k"] == 20 and b["temperature"] == 0.6 and isinstance(b["seed"], int)
    # the cache breakpoint only on the system message (the stable prefix), not on the newest message
    for x in bodies:
        assert "cache_control" in x["messages"][0]["content"][-1]
        assert all(isinstance(m["content"], str) for m in x["messages"][1:])
        assert [m["role"] for m in x["messages"]][:2] == ["system", "user"]
        assert all(a["role"] != c["role"] for a, c in zip(x["messages"], x["messages"][1:]))  # roles alternate
    # Part 2: its instructions, the memory block and the opening turn are one user message
    p2 = next(x for x in bodies if x["messages"][1]["content"].startswith("Part 1 is over."))
    assert "\n\nYour notes" in p2["messages"][1]["content"] or "\n\nYou have not written" in p2["messages"][1][
        "content"]
    assert rec["usage"]["calls"] == len(bodies) == 2 * (2 + 1 + 1)  # 2 firms x (2 NoTalk + 1 chat + 1 price)
    assert rec["usage"]["seed_possibly_ignored_calls"] == 0


async def test_compact_call_log_reconstructs_requests(tmp_path):
    """calls.jsonl logs the full request only for each firm's first call and for failed (not rate-limited)
    attempts; every request is rebuilt from the market record and matches its logged sha256."""
    import httpx
    import respx

    from oligopoly.llm import BASE_URL, Budget, OpenRouterClient, messages_sha256, reconstruct_request

    cfg = ModelConfig("tiny", "acme/tiny", "deepinfra/fp8", max_tokens=64)
    ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "pricing": {"prompt": "0.000001", "completion": "0"},
          "supported_parameters": ["seed"]}
    n_posts = 0

    def reply(request):
        nonlocal n_posts
        n_posts += 1
        if n_posts in (5, 9):  # a server error and a rate limit, each retried
            return httpx.Response(500 if n_posts == 5 else 429, json={"error": {"message": "x"}})
        last = json.loads(request.content)["messages"][-1]["content"]
        content = '{"notes": "n", "price": 99}' if '"price": <integer' in last.splitlines()[-1] \
            else '{"notes": "n", "message": "hi"}'
        return httpx.Response(200, json={"id": "g", "model": "acme/tiny", "provider": "DeepInfra",
                                         "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0}})

    with respx.mock() as r:
        r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
            return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
        r.post(f"{BASE_URL}/chat/completions").mock(side_effect=reply)
        async with OpenRouterClient(tmp_path / "calls.jsonl", Budget(1.0), api_key="k",
                                    backoff_base_s=0, rate_limit_backoff_base_s=0) as client:
            s = MarketSpec("live", cfg, 2, "T-NT", 0, 1)
            rec = json.loads((await run_market(s, client, tmp_path, ExperimentConfig(part_lengths=(3, 2)))).read_text())
    log = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    log = [x for x in log if "event" not in x]
    assert len(log) == n_posts
    from oligopoly.llm import reconstruct_from_log

    contexts = {a["call_id"]: a["context"] for L in ("A", "B") for d in firm_decisions(rec, L) for a in d["attempts"]}
    seen: dict[str, list] = {}
    refs = 0
    for i, line in enumerate(log):
        L = line["tag"]["firm"]
        ctx = contexts[line["tag"]["call_id"]]
        plain = reconstruct_request(rec, L, ctx)
        msgs = reconstruct_request(rec, L, ctx, line["cache_upto"])
        assert messages_sha256(msgs) == line["messages_sha256"] and messages_sha256(plain) == ctx["messages_sha256"]
        assert reconstruct_from_log(log, i) == msgs
        assert line["prompt_bytes"] == sum(len(m["content"].encode()) for m in plain)
        assert line["cache_upto"] == [0]
        full = L not in seen or (line["error"] is not None and not line.get("rate_limited"))
        assert ("request" in line) == full, line["tag"]
        if full:
            assert line["request"]["messages"] == msgs
        else:  # each message is an index into the previous logged request of the firm, or new
            assert len(line["messages_ref"]) == len(plain)
            for ref, m in zip(line["messages_ref"], plain):
                assert (seen[L][ref] if isinstance(ref, int) else ref) == m
            refs += sum(isinstance(r, int) for r in line["messages_ref"])
            assert line["request_params"]["seed"] == line["seed"] and "messages" not in line["request_params"]
        seen[L] = plain
    assert sum("request" in x for x in log) == 2 + 1  # one per firm, plus the HTTP 500
    assert refs > sum(len(x.get("messages_ref", [])) for x in log) / 2  # most messages are references


async def test_context_overflow_fails_market(tmp_path):
    """A prompt that outgrows the pinned endpoint's context stops the market before the call is sent."""
    import httpx
    import respx

    from oligopoly.llm import BASE_URL, Budget, ContextOverflow, OpenRouterClient

    cfg = ModelConfig("tiny", "acme/tiny", "deepinfra/fp8", max_tokens=64)
    ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "context_length": 2600,
          "pricing": {"prompt": "0.000001", "completion": "0"}, "supported_parameters": ["seed"]}
    reply = httpx.Response(200, json={"id": "g", "model": "acme/tiny", "provider": "DeepInfra",
                                      "choices": [{"message": {"content": '{"notes": "n", "price": 99}'}}],
                                      "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0}})
    with respx.mock() as r:
        r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
            return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
        post = r.post(f"{BASE_URL}/chat/completions").mock(return_value=reply)
        async with OpenRouterClient(tmp_path / "calls.jsonl", Budget(1.0), api_key="k") as client:
            s = MarketSpec("live", cfg, 2, "NT-NT", 0, 1)
            with pytest.raises(ContextOverflow):
                await run_market(s, client, tmp_path, ExperimentConfig(part_lengths=(6, 4)))
    fail = json.loads(engine.failure_path(tmp_path, s).read_text())
    assert fail["reason"] == "context_overflow" and fail["periods_completed"] >= 1
    log = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    overflows = [x for x in log if (x.get("error") or "").startswith("ContextOverflow")]
    assert overflows and post.call_count == len(log) - len(overflows)  # overflowing calls were never sent
    # a rerun does not pay to overflow again
    summary = await run_all([s], {"tiny": MockClient()}, tmp_path, ExperimentConfig(part_lengths=(6, 4)))
    assert summary["failed"] == [s.key] and not summary["done"]
    # the failure stores the guard settings; a change in any of them makes the market worth retrying
    assert fail["guard"] == {"bytes_per_token_min": 2.5, "max_tokens": 64, "context_length": 2600}
    exp = ExperimentConfig(part_lengths=(6, 4))
    assert plan(tmp_path, s, exp) == plan(tmp_path, s, exp, 2600) == "overflow"
    assert plan(tmp_path, s, exp, 4000) == "run"  # the endpoint's context grew
    looser = MarketSpec("live", dataclasses.replace(cfg, bytes_per_token_min=3.0), 2, "NT-NT", 0, 1)
    assert plan(tmp_path, looser, exp, 2600) == "run"
    # a failure record from before the guard was stored is not retried
    engine.write_json_atomic(engine.failure_path(tmp_path, s), {k: v for k, v in fail.items() if k != "guard"})
    assert plan(tmp_path, looser, exp, 4000) == "overflow"


def test_pick_error_prefers_context_overflow():
    from oligopoly.llm import ContextOverflow

    eg = ExceptionGroup("x", [RetriesExhausted("r"), ContextOverflow("c")])
    assert isinstance(engine.pick_error(eg), ContextOverflow)
    assert isinstance(engine.pick_error(ExceptionGroup("x", [ContextOverflow("c"), BudgetExceeded("b")])),
                      BudgetExceeded)


async def test_excluded_cells_are_skipped(tmp_path):
    f = tmp_path / "models.toml"
    f.write_text('[mock]\nmodel = "mock"\nprovider = "mock"\nmax_tokens = 100\n'
                 'exclude_cells = [["NT-T", 4], ["NT-NT", 2]]\nbytes_per_token_min = 2.2\n')
    m = load_models(f)["mock"]
    assert m.exclude_cells == (("NT-T", 4), ("NT-NT", 2)) and m.bytes_per_token_min == 2.2
    assert m.excludes(4, "NT-T") and not m.excludes(4, "T-NT")
    assert m.identity() == MOCK.identity() | {"max_tokens": 100, "temperature": None, "reasoning": None}
    specs = build_specs("x", [m], [2], ["NT-T", "NT-NT"], 1)
    s = await run_all(specs, {"mock": MockClient()}, tmp_path, SHORT)
    assert s["excluded"] == [specs[1].key] and s["done"] == [specs[0].key] and not s["failed"]
    assert not market_path(tmp_path, specs[1]).exists() and not engine.failure_path(tmp_path, specs[1]).exists()
    for bad in ('exclude_cells = [["X", 4]]', 'exclude_cells = [["NT-T", "4"]]', 'exclude_cells = [["NT-T", 1]]',
                'exclude_cells = [["NT-T", 4, 6]]', 'exclude_cells = [["NT-T"]]', 'exclude_cells = [4]',
                'bytes_per_token_min = "2.4"', 'bytes_per_token_min = true', 'bytes_per_token_min = 0'):
        f.write_text(f'[mock]\nmodel = "mock"\nprovider = "mock"\n{bad}\n')
        with pytest.raises(ConfigError):
            load_models(f)


# ---------------------------------------------------------------- review regressions


class Scripted(MockClient):
    """Answers by phase with fixed content."""

    def __init__(self, by_phase):
        super().__init__(record=True)
        self.by_phase = by_phase

    async def chat(self, messages, *, tag=None, **kw):
        r = await super().chat(messages, tag=tag, **kw)
        r.content = self.by_phase(tag)
        return r


async def test_history_stores_value_used(tmp_path):
    def answer(tag):
        if tag["phase"] == "price":
            return '{"notes": {"plan": 1}, "price": "99", "extra": "x"}'
        return '{"notes": "hi", "message": "' + "x" * 400 + '"}' if tag["firm"] == "A" else '{"message": "pass"}'

    rec, _, _ = await play(spec(2, "T-NT"), tmp_path, exp=ExperimentConfig(part_lengths=(1, 1), chat_rounds=1),
                           client=Scripted(answer))
    a = [json.loads(m["content"]) for m in rec["transcripts"]["A"] if m["role"] == "assistant"]
    b = [json.loads(m["content"]) for m in rec["transcripts"]["B"] if m["role"] == "assistant"]
    assert a[0] == {"notes": "hi", "message": "x" * 300}  # truncated as posted
    assert b[0] == {"message": None}  # "pass" stored as a pass
    assert a[1] == {"notes": {"plan": 1}, "price": 99}  # coerced price; notes as returned


async def test_gather_prefers_budget_exceeded():
    async def retries():
        raise RetriesExhausted("x")

    async def budget_later():
        await asyncio.sleep(0.01)
        raise BudgetExceeded("cap")

    finished = []

    async def sibling():
        await asyncio.sleep(0.05)
        finished.append(True)
        return "ok"

    m = engine.Market(spec(), MockClient(), SHORT)
    with pytest.raises(BudgetExceeded):
        await m.gather({"A": budget_later(), "B": retries(), "C": sibling()})
    assert finished == [True]  # a failure does not cancel the other firms' calls; they run to the end

    # cancelling the gather itself still cancels the decisions
    t = asyncio.create_task(m.gather({"A": asyncio.sleep(10), "B": asyncio.sleep(10)}))
    await asyncio.sleep(0.01)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    nested = ExceptionGroup("g", [RetriesExhausted("a"), ExceptionGroup("h", [ValueError(), BudgetExceeded("b")])])
    assert isinstance(engine.pick_error(nested), BudgetExceeded)
    assert isinstance(engine.pick_error(ExceptionGroup("g", [ValueError(), KeyError()])), ValueError)


async def test_gather_cancels_siblings_on_context_overflow():
    """A ContextOverflow market is not retried under the same settings, so the other firms' calls in flight
    are cancelled instead of run to the end (other errors still let them finish)."""
    from oligopoly.llm import ContextOverflow

    async def overflow():
        await asyncio.sleep(0.01)
        raise ContextOverflow("too long")

    cancelled = []

    async def slow_sibling():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    m = engine.Market(spec(), MockClient(), SHORT)
    with pytest.raises(ContextOverflow):
        await asyncio.wait_for(m.gather({"A": overflow(), "B": slow_sibling(), "C": slow_sibling()}), 2)
    assert cancelled == [True, True]


async def test_checkpoint_serialised_once_and_written_in_thread(tmp_path, monkeypatch):
    threaded = []
    real = asyncio.to_thread

    async def spy(fn, *a, **k):
        threaded.append(fn.__name__)
        return await real(fn, *a, **k)

    monkeypatch.setattr(engine.asyncio, "to_thread", spy)

    def no_loads(*a, **k):
        raise AssertionError("checkpoint re-parsed")

    s = spec()
    m = engine.Market(s, MockClient(), SHORT, checkpoint=engine.checkpoint_path(tmp_path, s))
    monkeypatch.setattr(engine.json, "loads", no_loads)
    await m.save_checkpoint()
    monkeypatch.undo()
    assert threaded == ["dumps", "write_text_atomic"]  # serialised and written off the event loop
    assert engine.checkpoint_path(tmp_path, s).read_text() == m._boundary
    assert json.loads(m._boundary)["periods_completed"] == 0


async def test_fingerprint_and_digest_computed_once_per_market(tmp_path, monkeypatch):
    calls = []
    real_fp, real_td = engine.market_fingerprint, engine.texts_digest
    monkeypatch.setattr(engine, "market_fingerprint", lambda *a: calls.append("fp") or real_fp(*a))
    monkeypatch.setattr(engine, "texts_digest", lambda *a: calls.append("td") or real_td(*a))
    rec, _, _ = await play(spec(2, "NT-T"), tmp_path)  # 10 periods + probe, a checkpoint after each
    assert calls == ["td", "fp"]
    monkeypatch.undo()
    assert rec["fingerprint"] == real_fp(spec(2, "NT-T"), SHORT)
    assert rec["settings"]["texts_digest"] == real_td(SHORT)


async def test_setup_error_writes_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad die")

    monkeypatch.setattr(engine.market, "die_schedule", boom)
    s = spec()
    with pytest.raises(RuntimeError):
        await run_market(s, MockClient(), tmp_path, SHORT)
    fail = json.loads(engine.failure_path(tmp_path, s).read_text())
    assert fail["error_type"] == "RuntimeError" and fail["periods_completed"] == 0


async def test_mock_kind_from_phase():
    c = MockClient(malformed_rate=0.0)
    msgs = [{"role": "user", "content": "anything"}]
    for seed in range(20):
        price = await c.chat(msgs, model="m", provider="mock", seed=seed, tag={"phase": "price"})
        chat = await c.chat(msgs, model="m", provider="mock", seed=seed, tag={"phase": "chat1"})
        assert "price" in price.content and "message" in chat.content


async def test_market_fingerprint_mismatch_not_complete(tmp_path):
    specs = build_specs("r", [MOCK], [2], ["NT-NT"], 1)
    await run_all(specs, {"mock": MockClient()}, tmp_path, SHORT)
    path = market_path(tmp_path, specs[0])
    assert engine.is_complete(tmp_path, specs[0], SHORT)
    assert engine.is_complete(tmp_path, specs[0], ExperimentConfig(part_lengths=(6, 5))) is None
    before = path.read_bytes()
    s = await run_all(specs, {"mock": MockClient()}, tmp_path, ExperimentConfig(part_lengths=(6, 5)))
    assert s["failed"] == [specs[0].key] and path.read_bytes() == before  # never overwritten


def test_fingerprint_covers_firm_facing_text(monkeypatch):
    s = build_specs("r", [MOCK], [2], ["NT-NT"], 1)[0]
    before = engine.market_fingerprint(s, SHORT)
    monkeypatch.setattr(engine.prompts, "price_reminder", lambda notes: "Reply with JSON, please.")
    assert engine.market_fingerprint(s, SHORT) != before


def models_file(tmp_path, temperature=1.0):
    f = tmp_path / "models.toml"
    f.write_text(f'[mock]\nmodel = "mock"\nprovider = "mock"\ntemperature = {temperature}\nmax_tokens = 100\n'
                 '[mock2]\nmodel = "mock"\nprovider = "mock"\nmax_tokens = 100\ntimeout_s = 5\n'
                 '[live]\nmodel = "acme/tiny"\nprovider = "nope"\nmax_tokens = 100\n')
    return str(f)


async def test_run_config_resume(tmp_path):
    base = ["--run-id", "rc", "--n", "2", "--orderings", "NT-NT", "--markets-per-cell", "1", "--part-lengths",
            "2,1", "--out-root", str(tmp_path / "runs")]
    mf = models_file(tmp_path)
    assert await main(base + ["--models", "mock", "--models-file", mf]) == 0
    cfg_path = tmp_path / "runs" / "rc" / "run_config.json"
    assert list(json.loads(cfg_path.read_text())["models"]) == ["mock"]
    assert await main(base + ["--models", "mock", "mock2", "--models-file", mf]) == 0  # new model appended
    cfg = json.loads(cfg_path.read_text())
    assert list(cfg["models"]) == ["mock", "mock2"] and cfg["experiment"]["part_lengths"] == [2, 1]
    assert "timeout_s" not in cfg["models"]["mock2"]
    with pytest.raises(ConfigError, match="new --run-id"):
        await main(base + ["--models", "mock", "--models-file", models_file(tmp_path, temperature=0.5)])
    with pytest.raises(ConfigError, match="experiment"):
        await main(base[:-3] + ["3,1", "--out-root", str(tmp_path / "runs"), "--models", "mock2",
                                "--models-file", mf])
    with pytest.raises(ConfigError, match="base_seed"):
        await main(base + ["--models", "mock", "--models-file", mf, "--seed", "1"])


async def test_preflight_aborts_before_any_market(tmp_path, monkeypatch):
    import httpx
    import respx

    from oligopoly.llm import BASE_URL

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "pricing": {"prompt": "0", "completion": "0"}}
    with respx.mock() as r:
        r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
            return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
        with pytest.raises(ConfigError, match="(?s)preflight.*'nope'"):
            await main(["--run-id", "pf", "--models", "mock", "live", "--n", "2", "--budget-usd", "1",
                        "--models-file", models_file(tmp_path), "--out-root", str(tmp_path / "runs")])
    run = tmp_path / "runs" / "pf"
    assert not (run / "markets").exists() and not (run / "run_config.json").exists()


def rc_args(tmp_path, *extra):
    return ["--run-id", "rc", "--n", "2", "--orderings", "NT-NT", "--part-lengths", "2,1", "--models", "mock",
            "--models-file", models_file(tmp_path), "--out-root", str(tmp_path / "runs"), *extra]


async def test_second_invocation_on_a_run_fails_at_once(tmp_path):
    from oligopoly.run import lock_run_dir

    run = tmp_path / "runs" / "rc"
    held = lock_run_dir(run)
    try:
        with pytest.raises(ConfigError, match=f"in use by another invocation \\(pid {os.getpid()}"):
            await main(rc_args(tmp_path, "--markets-per-cell", "1"))
        assert not (run / "run_config.json").exists() and not (run / "markets").exists()
        assert await main(rc_args(tmp_path, "--markets-per-cell", "1", "--dry-run")) == 0  # read-only: no lock
    finally:
        held.close()
    assert await main(rc_args(tmp_path, "--markets-per-cell", "1")) == 0
    assert await main(rc_args(tmp_path, "--markets-per-cell", "1")) == 0  # the lock was released on exit


async def test_prompt_wording_change_refused_for_run(tmp_path, monkeypatch):
    assert await main(rc_args(tmp_path, "--markets-per-cell", "1")) == 0
    run = tmp_path / "runs" / "rc"
    cfg = json.loads((run / "run_config.json").read_text())
    [rec_path] = (run / "markets").glob("*.json")
    assert cfg["texts_digest"] == json.loads(rec_path.read_text())["settings"]["texts_digest"]
    monkeypatch.setattr(engine.prompts, "price_reminder", lambda notes: "Reply with JSON, please.")
    with pytest.raises(ConfigError, match="texts_digest.*new --run-id"):
        await main(rc_args(tmp_path, "--markets-per-cell", "2"))
    assert len(list((run / "markets").glob("*.json"))) == 1  # nothing scheduled


async def test_legacy_run_config_without_digest(tmp_path, monkeypatch):
    assert await main(rc_args(tmp_path, "--markets-per-cell", "1")) == 0
    run = tmp_path / "runs" / "rc"
    [rec_path] = (run / "markets").glob("*.json")

    def strip_digests():
        cfg = json.loads((run / "run_config.json").read_text())
        del cfg["texts_digest"]
        (run / "run_config.json").write_text(json.dumps(cfg))
        rec = json.loads(rec_path.read_text())
        rec["settings"].pop("texts_digest", None)
        rec_path.write_text(json.dumps(rec))

    strip_digests()  # records from before the digest was stored are checked by their fingerprint
    assert await main(rc_args(tmp_path, "--markets-per-cell", "2")) == 0
    assert "texts_digest" in json.loads((run / "run_config.json").read_text())
    strip_digests()
    monkeypatch.setattr(engine.prompts, "price_reminder", lambda notes: "Reply with JSON, please.")
    with pytest.raises(ConfigError, match="completed market"):
        await main(rc_args(tmp_path, "--markets-per-cell", "3"))


async def test_dry_run_counts_only_markets_that_would_run(tmp_path, capsys):
    assert await main(rc_args(tmp_path, "--markets-per-cell", "3")) == 0
    run = tmp_path / "runs" / "rc"
    exp = ExperimentConfig(part_lengths=(2, 1))
    m = load_models(models_file(tmp_path))["mock"]
    specs = build_specs("rc", [m], [2], ["NT-NT"], 4)
    rec = json.loads(market_path(run, specs[1]).read_text())
    rec["fingerprint"] = "other settings"  # mismatch: never replayed
    market_path(run, specs[1]).write_text(json.dumps(rec))
    market_path(run, specs[2]).unlink()  # overflowed under these settings: not retried
    engine.failure_path(run, specs[2]).write_text(json.dumps(
        {"reason": "context_overflow", "fingerprint": engine.market_fingerprint(specs[2], exp)}))
    capsys.readouterr()
    assert await main(rc_args(tmp_path, "--markets-per-cell", "4", "--dry-run")) == 0
    assert "To run: 1 markets" in capsys.readouterr().out  # only m3


async def test_dry_run_plans_with_endpoint_context_length(tmp_path, capsys):
    """The dry run judges a context_overflow failure with the pinned endpoint's current context length, as
    run_all does: a market that overflowed at 2600 tokens is counted again once the endpoint offers 4000."""
    import httpx
    import respx

    from oligopoly.llm import BASE_URL

    f = tmp_path / "models.toml"
    f.write_text('[live]\nmodel = "acme/tiny"\nprovider = "deepinfra/fp8"\nmax_tokens = 64\n')
    live = load_models(f)["live"]
    exp = ExperimentConfig(part_lengths=(2, 1))
    run = tmp_path / "runs" / "dr"
    [s] = build_specs("dr", [live], [2], ["NT-NT"], 1)
    engine.write_json_atomic(engine.failure_path(run, s), {
        "reason": "context_overflow", "fingerprint": engine.market_fingerprint(s, exp),
        "guard": {"bytes_per_token_min": 2.5, "max_tokens": 64, "context_length": 2600}})
    args = ["--run-id", "dr", "--models", "live", "--n", "2", "--orderings", "NT-NT", "--markets-per-cell", "1",
            "--part-lengths", "2,1", "--models-file", str(f), "--out-root", str(tmp_path / "runs"), "--dry-run"]
    for ctx, todo in ((2600, 0), (4000, 1)):
        ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "context_length": ctx,
              "pricing": {"prompt": "0.000001", "completion": "0.000001"}}
        with respx.mock() as r:
            r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
                return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
            capsys.readouterr()
            assert await main(args) == 0
        assert f"To run: {todo} markets" in capsys.readouterr().out


# ---------------------------------------------------------------- checkpoints


class FailOnce(MockClient):
    """Raises RetriesExhausted on the call at `at` = (firm, part, period, phase, attempt); otherwise the mock."""

    def __init__(self, at):
        super().__init__(record=True)
        self.at = at

    async def chat(self, messages, *, tag=None, **kw):
        if (tag["firm"], tag["part"], tag["period"], tag["phase"], tag["attempt"]) == self.at:
            raise RetriesExhausted("injected")
        return await super().chat(messages, tag=tag, **kw)


def without_resume(rec):
    return {k: v for k, v in rec.items() if k != "resumed_from_period"}


@pytest.mark.parametrize("at", [("B", 1, 3, "price", 0), ("A", 2, 1, "chat1", 0), ("C", 2, 4, "price", 0)])
async def test_checkpoint_resume_matches_uninterrupted(tmp_path, at):
    s = spec(3, "NT-T")
    ref, ref_client, _ = await play(s, tmp_path / "ref")
    assert ref["resumed_from_period"] == [] and not engine.checkpoint_path(tmp_path / "ref", s).exists()
    out = tmp_path / "run"
    c1 = FailOnce(at)
    with pytest.raises(RetriesExhausted):
        await run_market(s, c1, out, SHORT)
    ck = engine.checkpoint_path(out, s)
    k = at[2] if at[1] == 1 else SHORT.part_lengths[0] + at[2]  # 1-based over both parts
    saved = json.loads(ck.read_text())
    assert saved["periods_completed"] == k - 1 and saved["partial_calls"]  # siblings' results kept
    # the replay key is the request's context hash (one scheme: messages_sha256 of the messages sent)
    from oligopoly.llm import messages_sha256
    sent = {c["tag"]["call_id"]: c["messages"] for c in c1.calls}
    assert all(v["messages_sha256"] == messages_sha256(sent[cid]) for cid, v in saved["partial_calls"].items())
    c2 = MockClient(record=True)
    rec = json.loads((await run_market(s, c2, out, SHORT)).read_text())
    assert not ck.exists() and not engine.failure_path(out, s).exists()
    assert rec["resumed_from_period"] == [k]
    assert without_resume(rec) == without_resume(ref)  # same histories, periods, totals and usage
    first, second = [c["tag"]["call_id"] for c in c1.calls], [c["tag"]["call_id"] for c in c2.calls]
    assert not set(first) & set(second)  # no completed call is made again, not even within period k
    assert all((c["tag"]["part"], c["tag"]["period"]) >= at[1:3] for c in c2.calls if c["tag"]["phase"] != "probe")
    assert len(first) + len(second) == len(ref_client.calls)
    # the resumed run rebuilt exactly the windows (instructions, memory, turns) of the uninterrupted one
    ref_msgs = {c["tag"]["call_id"]: c["messages"] for c in ref_client.calls}
    assert all(c["messages"] == ref_msgs[c["tag"]["call_id"]] for c in c1.calls + c2.calls)
    check_contexts(rec, ref_client, SHORT)


async def test_boundary_only_checkpoint_resumes_at_period(tmp_path):
    """A process killed mid-period leaves only the last boundary checkpoint: the period is replayed, earlier
    ones are not."""
    s = spec(2, "T-NT")
    ref, _, _ = await play(s, tmp_path / "ref")
    out = tmp_path / "run"
    with pytest.raises(RetriesExhausted):
        await run_market(s, FailOnce(("A", 1, 4, "chat1", 0)), out, SHORT)
    ck = engine.checkpoint_path(out, s)
    engine.write_json_atomic(ck, {**json.loads(ck.read_text()), "partial_calls": {}})
    c2 = MockClient(record=True)
    rec = json.loads((await run_market(s, c2, out, SHORT)).read_text())
    assert without_resume(rec) == without_resume(ref) and rec["resumed_from_period"] == [4]
    assert min((c["tag"]["part"], c["tag"]["period"]) for c in c2.calls if c["tag"]["phase"] != "probe") == (1, 4)


@pytest.mark.parametrize("field", ["die_schedules", "listing_orders", "fingerprint", "texts_digest", "garbage"])
async def test_mismatched_checkpoint_is_ignored(tmp_path, capsys, field):
    s = spec(2, "NT-T")
    ref, _, _ = await play(s, tmp_path / "ref")
    out = tmp_path / "run"
    with pytest.raises(RetriesExhausted):
        await run_market(s, FailOnce(("B", 1, 3, "price", 0)), out, SHORT)
    ck = engine.checkpoint_path(out, s)
    saved = json.loads(ck.read_text())
    if field == "garbage":
        ck.write_text("{not json")
    else:
        saved[field] = {"die_schedules": [[1], [2]], "listing_orders": {"A": ["B", "A"], "B": ["A", "B"]}}.get(
            field, "0" * 64)
        engine.write_json_atomic(ck, saved)
    c2 = MockClient(record=True)
    rec = json.loads((await run_market(s, c2, out, SHORT)).read_text())
    assert "CHECKPOINT ignored" in capsys.readouterr().out
    assert rec == ref and rec["resumed_from_period"] == []  # restarted from scratch
    assert min((c["tag"]["part"], c["tag"]["period"]) for c in c2.calls if c["tag"]["phase"] != "probe") == (1, 1)


async def test_checkpoint_files_are_not_market_records(tmp_path):
    assert engine.is_market_record_file(tmp_path / "x__n2__NT-T__m0.json")
    assert not engine.is_market_record_file(tmp_path / "x__n2__NT-T__m0.ckpt.json")
    assert not engine.is_market_record_file(tmp_path / "x__n2__NT-T__m0.failed.json")


async def test_budget_exhaustion_checkpoints_and_rerun_resumes(tmp_path):
    class Exhausted(FailOnce):
        async def chat(self, messages, *, tag=None, **kw):
            if (tag["firm"], tag["part"], tag["period"], tag["phase"], tag["attempt"]) == self.at:
                raise BudgetExceeded("cap exhausted")
            return await MockClient.chat(self, messages, tag=tag, **kw)

    [s] = build_specs("test", [MOCK], [3], ["NT-T"], 1)
    ref, _, _ = await play(s, tmp_path / "ref")
    c1 = Exhausted(("B", 2, 2, "price", 0))
    s1 = await run_all([s], {"mock": c1}, tmp_path / "run", SHORT)
    assert s1["failed"] == [s.key]
    ck = json.loads(engine.checkpoint_path(tmp_path / "run", s).read_text())
    assert ck["periods_completed"] == 7 and {"A", "C"} <= {k.split("/")[1] for k in ck["partial_calls"]}
    c2 = MockClient(record=True)
    s2 = await run_all([s], {"mock": c2}, tmp_path / "run", SHORT)
    assert s2["done"] == [s.key]
    rec = json.loads(market_path(tmp_path / "run", s).read_text())
    assert without_resume(rec) == without_resume(ref) and rec["resumed_from_period"] == [8]
    assert not {c["tag"]["call_id"] for c in c1.calls} & {c["tag"]["call_id"] for c in c2.calls}


async def test_parser_version_in_fingerprint_and_run_config(tmp_path, monkeypatch):
    import oligopoly.run as run_mod
    from oligopoly.parse import PARSER_VERSION

    s = build_specs("r", [MOCK], [2], ["NT-NT"], 1)[0]
    before = engine.market_fingerprint(s, SHORT)
    monkeypatch.setattr(engine, "PARSER_VERSION", PARSER_VERSION + 1)
    assert engine.market_fingerprint(s, SHORT) != before
    monkeypatch.undo()
    assert await main(rc_args(tmp_path, "--markets-per-cell", "1")) == 0
    run = tmp_path / "runs" / "rc"
    assert json.loads((run / "run_config.json").read_text())["parser_version"] == PARSER_VERSION
    [rec_path] = (run / "markets").glob("*.json")
    rec = json.loads(rec_path.read_text())
    assert rec["settings"]["parser_version"] == PARSER_VERSION
    # a legacy record (no parser_version, no texts_digest) is still matched by its fingerprint
    legacy = {**rec, "settings": {k: v for k, v in rec["settings"].items()
                                  if k not in ("parser_version", "texts_digest")}}
    st = legacy["settings"]
    identity = {k: v for k, v in st["model"].items() if k not in ModelConfig.OPERATIONAL}
    legacy["fingerprint"] = engine._fingerprint_from(legacy["spec"], identity, st["experiment"],
                                                     rec["settings"]["texts_digest"])
    assert engine.record_texts_digest_matches(legacy, rec["settings"]["texts_digest"])
    monkeypatch.setattr(run_mod, "PARSER_VERSION", PARSER_VERSION + 1)
    with pytest.raises(ConfigError, match="parser_version"):
        await main(rc_args(tmp_path, "--markets-per-cell", "2"))


# ---------------------------------------------------------------- post-game probe


NO_PROBE = dataclasses.replace(SHORT, post_game_probe=False)


@pytest.mark.parametrize("ordering", ["NT-T", "T-NT", "NT-NT"])
async def test_probe_does_not_change_play(tmp_path, ordering):
    s = spec(3, ordering)
    on, c_on, _ = await play(s, tmp_path / "on")
    off, c_off, _ = await play(s, tmp_path / "off", exp=NO_PROBE)
    play_calls = [c for c in c_on.calls if c["tag"]["phase"] != "probe"]
    assert [(c["tag"], c["seed"], c["messages"], c["content"]) for c in play_calls] == \
        [(c["tag"], c["seed"], c["messages"], c["content"]) for c in c_off.calls]
    assert on["periods"] == off["periods"] and on["totals"] == off["totals"] and off["probe"] is None
    probe_calls = [c for c in c_on.calls if c["tag"]["phase"] == "probe"]
    assert {c["tag"]["firm"] for c in probe_calls} == {"A", "B", "C"}
    assert all(c["tag"]["attempt"] <= engine.PROBE_MAX_REASKS for c in probe_calls)
    assert on["usage"]["calls"] == off["usage"]["calls"] + len(probe_calls)
    pr = on["probe"]
    assert pr["truth"] == {"max_consumer_price": 100, "best_common_price": 100,
                           "last_period_prices": on["periods"][-1]["prices"]}
    for L, f in pr["firms"].items():
        # the probe extends the history that play left: final results and part-over text, then the questions
        n_play = len(off["transcripts"][L]) - 1  # the probe-off history ends with the flushed final texts
        assert on["transcripts"][L][:n_play] == off["transcripts"][L][:n_play]
        assert on["transcripts"][L][n_play]["content"].startswith(off["transcripts"][L][n_play]["content"])
        assert "Please answer these questions" in on["transcripts"][L][n_play]["content"]
        assert set(f["score"]) >= {"max_consumer_price", "best_common_price", "last_period_share"}


async def test_probe_records_answers_scores_and_invalid(tmp_path):
    def answer(tag):
        if tag["phase"] == "probe":
            if tag["firm"] == "B":
                return "I would rather not say."
            return json.dumps({"max_consumer_price": 100, "best_common_price": 99,
                               "last_period_prices": {"A": 7, "B": 7}, "deviations": "B undercut in period 1."})
        return '{"notes": "n", "price": 7}' if tag["phase"] == "price" else '{"notes": "n", "message": "hi"}'

    rec, client, _ = await play(spec(2, "NT-T"), tmp_path, exp=ExperimentConfig(part_lengths=(2, 1)),
                                client=Scripted(answer))
    a, b = rec["probe"]["firms"]["A"], rec["probe"]["firms"]["B"]
    assert a["valid"] and a["answers"]["deviations"] == "B undercut in period 1." and len(a["attempts"]) == 1
    assert a["score"] == {"max_consumer_price": True, "best_common_price": False, "last_period_correct": 2,
                          "last_period_share": 1.0, "last_period_all": True}
    assert not b["valid"] and len(b["attempts"]) == 2 and len(b["errors"]) == 2  # one re-ask, then invalid
    assert b["score"]["max_consumer_price"] is False and b["score"]["last_period_share"] == 0
    assert "could not be used" in rec["transcripts"]["B"][-2]["content"]  # the re-ask is in B's history
    assert sum(c["tag"]["phase"] == "probe" for c in client.calls) == 3


async def test_probe_failure_resumes_only_the_probe(tmp_path):
    s = spec(3, "NT-T")
    ref, ref_client, _ = await play(s, tmp_path / "ref")
    out = tmp_path / "run"
    c1 = FailOnce(("B", "probe", None, "probe", 0))
    with pytest.raises(RetriesExhausted):
        await run_market(s, c1, out, SHORT)
    ck = json.loads(engine.checkpoint_path(out, s).read_text())
    assert ck["periods_completed"] == 10 and ck["probe"] is None
    assert {k.split("/")[1] for k in ck["partial_calls"]} >= {"A", "C"}  # siblings' probe answers kept
    fail = json.loads(engine.failure_path(out, s).read_text())
    assert fail["periods_completed"] == 10
    c2 = MockClient(record=True)
    rec = json.loads((await run_market(s, c2, out, SHORT)).read_text())
    assert {(c["tag"]["firm"], c["tag"]["phase"]) for c in c2.calls} <= {("B", "probe")} and c2.calls
    assert without_resume(rec) == without_resume(ref) and rec["resumed_from_period"] == [11]
    assert len(c1.calls) + len(c2.calls) == len(ref_client.calls)


async def test_probe_context_overflow_does_not_fail_market(tmp_path):
    from oligopoly.llm import ContextOverflow

    class OverflowProbe(MockClient):
        async def chat(self, messages, *, tag=None, **kw):
            if tag["phase"] == "probe" and tag["firm"] == "A":
                raise ContextOverflow("too long")
            return await super().chat(messages, tag=tag, **kw)

    s = spec(2, "NT-NT")
    rec, _, _ = await play(s, tmp_path, client=OverflowProbe())
    a = rec["probe"]["firms"]["A"]
    assert a["skipped"] == "context_overflow" and not a["valid"] and a["score"]["last_period_share"] == 0
    assert rec["probe"]["firms"]["B"]["skipped"] is None


WRONG = {"max_consumer_price": False, "best_common_price": False, "last_period_correct": 0,
         "last_period_share": 0.0, "last_period_all": False}
PARTIAL = json.dumps({"max_consumer_price": 100, "best_common_price": 100, "deviations": "none"})  # no last prices


def probe_or_play(probe):
    def answer(tag):
        if tag["phase"] == "probe":
            return probe(tag)
        return '{"notes": "n", "price": 100}' if tag["phase"] == "price" else '{"notes": "n", "message": null}'
    return answer


async def test_partially_valid_exhausted_probe_scores_wrong(tmp_path):
    """A probe still invalid after its re-ask scores wrong on every item, even on the parts it got right; the
    partial answers stay in the record."""
    rec, _, _ = await play(spec(2, "NT-NT"), tmp_path, exp=ExperimentConfig(part_lengths=(1, 1)),
                           client=Scripted(probe_or_play(lambda tag: PARTIAL)))
    for f in rec["probe"]["firms"].values():
        assert not f["valid"] and len(f["attempts"]) == 2 and f["skipped"] is None
        assert f["answers"]["max_consumer_price"] == 100 and f["answers"]["last_period_prices"] is None
        assert f["score"] == WRONG


async def test_probe_overflow_on_reask_scores_wrong(tmp_path):
    from oligopoly.llm import ContextOverflow

    class OverflowReask(Scripted):
        async def chat(self, messages, *, tag=None, **kw):
            if tag["phase"] == "probe" and tag["attempt"] == 1:
                raise ContextOverflow("too long")
            return await super().chat(messages, tag=tag, **kw)

    rec, _, _ = await play(spec(2, "NT-NT"), tmp_path, exp=ExperimentConfig(part_lengths=(1, 1)),
                           client=OverflowReask(probe_or_play(lambda tag: PARTIAL)))
    for f in rec["probe"]["firms"].values():
        assert f["skipped"] == "context_overflow" and not f["valid"] and len(f["attempts"]) == 1
        assert f["answers"]["best_common_price"] == 100  # kept for auditing, not scored
        assert f["score"] == WRONG


def test_probe_setting_and_wording_in_fingerprint(monkeypatch):
    s = build_specs("r", [MOCK], [2], ["NT-NT"], 1)[0]
    before = engine.market_fingerprint(s, SHORT)
    assert engine.market_fingerprint(s, NO_PROBE) != before
    monkeypatch.setitem(engine.prompts.PROBE_QUESTIONS, "deviations", "Did anyone cheat?")
    assert engine.market_fingerprint(s, SHORT) != before
    assert engine.best_common_price(["A", "B", "C"]) == 100


def test_memory_window_setting(tmp_path):
    from oligopoly.config import load_experiment

    s = build_specs("r", [MOCK], [2], ["NT-NT"], 1)[0]
    assert load_experiment().memory_window == 3 and ExperimentConfig().memory_window == 3
    assert engine.market_fingerprint(s, dataclasses.replace(SHORT, memory_window=2)) != \
        engine.market_fingerprint(s, SHORT)
    assert engine.texts_digest(dataclasses.replace(SHORT, memory_window=2)) != engine.texts_digest(SHORT)
    for bad in (0, -1, True, "3", 2.0):
        with pytest.raises(ConfigError):
            ExperimentConfig(memory_window=bad)


async def test_price_on_chat_turn_gets_its_own_reask(tmp_path):
    """A chat turn answered with a price object is re-asked with the chat-turn wording; other invalid answers
    keep the generic re-ask."""
    from oligopoly.prompts import CHAT_PRICE_ERROR

    def answer(tag):
        if tag["phase"] == "chat1" and tag["attempt"] == 0:
            return '{"notes": "n", "price": 100}' if tag["firm"] == "A" else "no json here"
        return '{"notes": "n", "price": 99}' if tag["phase"] == "price" else '{"notes": "n", "message": null}'

    rec, client, _ = await play(spec(2, "T-NT"), tmp_path, exp=ExperimentConfig(part_lengths=(1, 1)),
                                client=Scripted(answer))
    da = rec["periods"][0]["chat"][0]["decisions"]["A"]
    db = rec["periods"][0]["chat"][0]["decisions"]["B"]
    assert da["errors"] == [CHAT_PRICE_ERROR] and da["reasks"] == 1 and not da["fallback"]
    assert db["reasks"] == 1 and db["errors"] != [CHAT_PRICE_ERROR]
    reask = {c["tag"]["firm"]: c["messages"][-1]["content"] for c in client.calls
             if c["tag"]["phase"] == "chat1" and c["tag"]["attempt"] == 1}
    assert reask["A"].startswith("You answered with a price, but this is a chat turn. Prices are set after the "
                                 "chat closes. Post a message, or null to post nothing.")
    assert reask["B"].startswith("Your previous response could not be used:")


# ---------------------------------------------------------------- rolling window and memory


@pytest.mark.parametrize("window", [1, 2, 3, 12])
@pytest.mark.parametrize("ordering", ["NT-T", "T-NT"])
async def test_window_sizes(tmp_path, window, ordering):
    exp = dataclasses.replace(SHORT, memory_window=window)
    rec, client, _ = await play(spec(3, ordering), tmp_path, exp=exp)
    check_contexts(rec, client, exp)
    if window == 12:  # wider than the market: every request holds the whole transcript so far
        for c in client.calls:
            ctx = next(a["context"] for d in firm_decisions(rec, c["tag"]["firm"]) for a in d["attempts"]
                       if a["call_id"] == c["tag"]["call_id"])
            assert ctx["window"][0] == 0


def test_cap_notes():
    cap = engine.cap_notes
    assert cap("a b c", 3, 100) == ("a b c", False)
    assert cap("a b c d", 3, 100) == ("a b c [truncated]", True)
    text, cut = cap("word " * 300, 200, 1400)  # 200 words would fit in 1,400 chars here; the word cap binds
    assert cut and text == "word " * 199 + "word [truncated]"
    text, cut = cap("abcdefghij " * 150, 200, 1400)  # the character cap binds, at a word boundary
    assert cut and len(text) <= 1400 and text.endswith("abcdefghij [truncated]") and len(text.split()) == 126 + 1
    text, cut = cap("x" * 2000, 200, 1400)  # one giant word: cut inside it
    assert cut and len(text) == 1400 and text.endswith("x [truncated]")
    assert cap("  spaced   out  ", 2, 100) == ("  spaced   out  ", False)
    with pytest.raises(ConfigError):
        ExperimentConfig(notes_max_words=0)


async def test_long_notes_are_capped_in_memory_and_window(tmp_path):
    """Notes over 200 words / 1,400 characters are cut at a word boundary and marked; the firm sees only the cut
    version (memory and its windowed answer); the record counts truncations and keeps the full text."""
    long = " ".join(f"w{i}" for i in range(400))

    def answer(tag):
        if tag["phase"] == "price":
            return json.dumps({"notes": long if tag["firm"] == "A" else "short-B", "price": 100})
        return '{"notes": "n", "message": null}'

    exp = ExperimentConfig(part_lengths=(2, 1), post_game_probe=False)
    rec, client, _ = await play(spec(2, "NT-NT"), tmp_path, exp=exp, client=Scripted(answer))
    check_contexts(rec, client, exp)
    capped, _ = engine.cap_notes(long, 200, 1400)
    assert capped.endswith(" [truncated]") and len(capped) <= 1400 and len(capped.split()) <= 201
    d = rec["periods"][0]["decisions"]["A"]
    assert d["notes"] == capped and d["notes_truncated"] and d["notes_full"] == long
    assert any("notes truncated" in w for w in d["warnings"])
    assert "notes_truncated" not in rec["periods"][0]["decisions"]["B"]
    assert rec["totals"]["A"]["notes_truncated"] == 3 and rec["totals"]["B"]["notes_truncated"] == 0
    blob = json.dumps([c["messages"] for c in client.calls if c["tag"]["firm"] == "A"], ensure_ascii=False)
    assert long not in blob and f"Your notes (written by you at your last decision): {capped}" in blob
    a_answers = [json.loads(m["content"]) for m in rec["transcripts"]["A"] if m["role"] == "assistant"]
    assert a_answers[0] == {"notes": capped, "price": 100}


def test_reconstruct_merged_and_legacy_contexts():
    """Requests merge same-role messages (roles alternate); a context without `merged` (records from before the
    merge) is rebuilt unmerged, so older records still verify."""
    from oligopoly.llm import reconstruct_request

    rec = {"instructions": {"A": ["P1", "P2"]}, "transcripts": {"A": [
        {"role": "user", "content": "u1", "period": 1}, {"role": "assistant", "content": "a1", "period": 1},
        {"role": "user", "content": "u2", "period": 2}]}}
    ctx = {"instructions": 2, "memory": "mem", "window": [0, 3]}
    assert reconstruct_request(rec, "A", {**ctx, "merged": True}) == [
        {"role": "system", "content": "P1"}, {"role": "user", "content": "P2\n\nmem\n\nu1"},
        {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    assert [m["content"] for m in reconstruct_request(rec, "A", ctx)] == ["P1", "P2", "mem", "u1", "a1", "u2"]


async def test_context_is_bounded_and_old_play_is_gone(tmp_path):
    """Chat messages and results older than the window never reach a request; the notes do."""
    def answer(tag):
        mark = f"P{tag['part']}T{tag['period']}"
        if tag["phase"] == "price":
            return json.dumps({"notes": f"memo-{tag['firm']}-{mark}", "price": 100})
        if tag["phase"] == "probe":
            return json.dumps({"max_consumer_price": 100, "best_common_price": 100,
                               "last_period_prices": {"A": 100, "B": 100}, "deviations": "none"})
        return json.dumps({"notes": f"memo-{tag['firm']}-{mark}-chat", "message": f"msg-{tag['firm']}-{mark}"})

    exp = ExperimentConfig(part_lengths=(8, 8), chat_rounds=1, memory_window=3)
    rec, client, _ = await play(spec(2, "T-NT"), tmp_path, exp=exp, client=Scripted(answer))
    check_contexts(rec, client, exp)
    sizes = []
    for c in client.calls:
        t = c["tag"]
        g = 17 if t["phase"] == "probe" else t["period"] if t["part"] == 1 else 8 + t["period"]
        text = "\n".join(m["content"] for m in c["messages"])
        sizes.append(len(text))
        for q in range(1, 9):  # chat messages of Part 1 periods older than g - 3 are gone
            if q < g - 3:
                assert f"msg-A-P1T{q}" not in text and f"msg-B-P1T{q}" not in text
        if g > 1 and t["phase"] != "probe":
            prev = f"P{1 if g - 1 <= 8 else 2}T{g - 1 if g - 1 <= 8 else g - 9}"
            # memory = this firm's latest notes (the previous period's price notes, or this period's chat notes)
            assert f"Your notes (written by you at your last decision): memo-{t['firm']}-" in c["messages"][1][
                "content"]
            assert f"memo-{t['firm']}-{prev}" in text
            assert f"memo-{'B' if t['firm'] == 'A' else 'A'}-" not in text
    assert max(sizes[len(sizes) // 2:]) < 1.5 * max(sizes[:len(sizes) // 2])  # does not grow with the market


async def test_answer_without_notes_keeps_memory(tmp_path):
    def answer(tag):
        if tag["phase"] == "price" and tag["period"] % 2 == 0:
            return '{"price": 100}'  # no notes: the memory keeps the previous notes
        if tag["phase"] == "price":
            return json.dumps({"notes": f"odd-{tag['firm']}-{tag['part']}-{tag['period']}", "price": 100})
        return '{"notes": "", "message": null}' if tag["firm"] == "A" else '{"message": null}'

    exp = ExperimentConfig(part_lengths=(4, 3), memory_window=1, post_game_probe=False)
    rec, client, _ = await play(spec(2, "NT-T"), tmp_path, exp=exp, client=Scripted(answer))
    check_contexts(rec, client, exp)
    mem = {c["tag"]["call_id"]: c["messages"][1]["content"] for c in client.calls if c["tag"]["part"] == 1}
    assert "odd-B-1-1" in mem["mock__n2__NT-T__m0/B/p1/t3/price/a0"]  # period 2's answer had no notes
    assert mem["mock__n2__NT-T__m0/A/p1/t1/price/a0"].startswith("You have not written any notes yet.")
    # Part 2 period 1, after the chat: A's empty notes replaced its memory; B's chat answer had none
    p2 = {c["tag"]["firm"]: c for c in client.calls if (c["tag"]["part"], c["tag"]["period"], c["tag"]["phase"])
          == (2, 1, "price")}
    assert "\n\nYour notes (written by you at your last decision): \n" in p2["A"]["messages"][1]["content"]
    assert "\n\nYour notes (written by you at your last decision): odd-B-1-3" in p2["B"]["messages"][1]["content"]


# ---------------------------------------------------------------- structured outputs


@pytest.mark.parametrize("structured", [False, True])
async def test_structured_outputs_payload(tmp_path, structured):
    """With structured_outputs, chat and price calls send a strict json_schema response_format (and
    require_parameters); the probe and the default send none."""
    import httpx
    import respx

    from oligopoly.llm import BASE_URL, Budget, OpenRouterClient

    cfg = ModelConfig("tiny", "acme/tiny", "deepinfra/fp8", max_tokens=64, structured_outputs=structured)
    ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "pricing": {"prompt": "0.000001", "completion": "0"},
          "supported_parameters": ["seed", "response_format", "structured_outputs"]}
    bodies = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        last = body["messages"][-1]["content"]
        if "Please answer these questions" in last:
            content = json.dumps({"max_consumer_price": 100, "best_common_price": 100,
                                  "last_period_prices": {"A": 99, "B": 99}, "deviations": "no"})
        elif '"price": <integer' in last.splitlines()[-1]:
            content = '{"notes": "n", "price": 99}'
        else:
            content = '{"notes": "n", "message": null}'
        return httpx.Response(200, json={"id": "g", "model": "acme/tiny", "provider": "DeepInfra",
                                         "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0}})

    with respx.mock() as r:
        r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
            return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
        r.post(f"{BASE_URL}/chat/completions").mock(side_effect=reply)
        async with OpenRouterClient(tmp_path / "calls.jsonl", Budget(1.0), api_key="k") as client:
            s = MarketSpec("live", cfg, 2, "T-NT", 0, 1)
            await run_market(s, client, tmp_path, ExperimentConfig(part_lengths=(1, 1)))
    kinds = {"chat": [], "price": [], "probe": []}
    for b in bodies:
        last = b["messages"][-1]["content"]
        kind = ("probe" if "Please answer these questions" in last else
                "price" if '"price": <integer' in last.splitlines()[-1] else "chat")
        kinds[kind].append(b)
    assert all(kinds.values())
    for b in bodies:
        assert b["provider"]["require_parameters"] is True
    if not structured:
        assert not any("response_format" in b for b in bodies)
        return
    assert not any("response_format" in b for b in kinds["probe"])
    for b in kinds["chat"]:
        assert b["response_format"] == {"type": "json_schema", "json_schema": {
            "name": "chat_turn", "strict": True, "schema": {
                "type": "object", "properties": {"notes": {"type": "string"}, "message": {"type": ["string", "null"]}},
                "required": ["notes", "message"], "additionalProperties": False}}}
    for b in kinds["price"]:
        assert b["response_format"] == {"type": "json_schema", "json_schema": {
            "name": "price_decision", "strict": True, "schema": {
                "type": "object", "properties": {"notes": {"type": "string"},
                                                 "price": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["notes", "price"], "additionalProperties": False}}}
    log = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert {x["response_format"]["json_schema"]["name"] for x in log if x.get("response_format")} == \
        {"chat_turn", "price_decision"}


def test_structured_outputs_config():
    base = ModelConfig("m", "acme/m", "p", max_tokens=10)
    on = dataclasses.replace(base, structured_outputs=True)
    assert base.structured_outputs is False and load_models()["gpt-oss-20b"].structured_outputs is False
    s0, s1 = (MarketSpec("r", m, 2, "NT-NT", 0, 1) for m in (base, on))
    assert engine.market_fingerprint(s0, SHORT) != engine.market_fingerprint(s1, SHORT)
    with pytest.raises(ConfigError, match="require_parameters"):
        dataclasses.replace(on, require_parameters=False)
    assert engine.response_schema("price", False)["json_schema"]["schema"]["required"] == ["price"]


@pytest.mark.parametrize("stage", ["dumps", "write_text_atomic"])
async def test_cancel_during_checkpoint_save_keeps_paid_results(tmp_path, monkeypatch, stage):
    """A cancellation at either await of save_checkpoint (here at the boundary after period 2) must leave the
    new boundary or the old one plus every paid result of period 2: the resume makes no call twice."""
    s = spec(3, "NT-T")
    ref, ref_client, _ = await play(s, tmp_path / "ref")
    real, seen = asyncio.to_thread, []

    async def cancel_at(fn, *a, **k):
        seen.append(fn.__name__)
        if fn.__name__ == stage and seen.count(stage) == 3:  # boundaries: start, after period 1, after period 2
            asyncio.current_task().cancel()
            await asyncio.sleep(0)  # cancelled here: the thread's work never lands
        return await real(fn, *a, **k)

    out = tmp_path / "run"
    c1 = MockClient(record=True)
    monkeypatch.setattr(engine.asyncio, "to_thread", cancel_at)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(run_market(s, c1, out, SHORT))
    monkeypatch.undo()
    saved = json.loads(engine.checkpoint_path(out, s).read_text())
    period2 = {c["tag"]["call_id"] for c in c1.calls if (c["tag"]["part"], c["tag"]["period"]) == (1, 2)}
    assert period2
    if stage == "dumps":  # not serialised: the old boundary plus all of period 2's results
        assert saved["periods_completed"] == 1 and set(saved["partial_calls"]) == period2
    else:  # serialised but maybe not written: the new boundary
        assert saved["periods_completed"] == 2 and saved["partial_calls"] == {}
    c2 = MockClient(record=True)
    rec = json.loads((await run_market(s, c2, out, SHORT)).read_text())
    assert not {c["tag"]["call_id"] for c in c1.calls} & {c["tag"]["call_id"] for c in c2.calls}
    assert len(c1.calls) + len(c2.calls) == len(ref_client.calls)
    assert without_resume(rec) == without_resume(ref)
