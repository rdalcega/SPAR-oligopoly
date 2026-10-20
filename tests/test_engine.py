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
        assert c["tag"]["phase"].startswith("chat") == talk[c["tag"]["part"] - 1] or c["tag"]["phase"] == "price"

    # payoffs consistent with market.payoffs, totals consistent with periods
    for p in rec["periods"]:
        assert p["profits"] == market.payoffs(p["prices"])
    for L in letters:
        assert rec["totals"][L]["profit_total"] == pytest.approx(sum(p["profits"][L] for p in rec["periods"]))

    by_firm = calls_by_firm(client)
    for L, calls in by_firm.items():
        hist = rec["histories"][L]
        # Part 1 texts never mention Part 2, nor communication in a NoTalk Part 1
        i2 = next(i for i, m in enumerate(hist) if "Instructions – Part 2" in m["content"])
        part1 = [m["content"] for m in hist[:i2] if m["role"] != "assistant"]
        part1.append(hist[i2]["content"].split("Part 1 is over.")[0])
        for text in part1:
            low = text.lower()
            assert "part 2" not in low and "second part" not in low
            if not talk[0]:
                assert not any(w in low for w in COMM), text
        # append-only: each call extends the previous one, and the final history extends the last call
        seq = [c["messages"] for c in calls] + [hist]
        for prev, cur in zip(seq, seq[1:]):
            assert cur[:len(prev)] == prev and len(cur) > len(prev)
        assert all(m["role"] != "assistant" or "mock reasoning" not in m["content"] for m in hist)

    # simultaneity: no same-round message / same-period price is visible
    posted = {(p["part"], p["period"], r["round"]): r["messages"] for p in rec["periods"] for r in p["chat"]}
    for c in client.calls:
        t = c["tag"]
        text = "\n".join(m["content"] for m in c["messages"])
        if t["phase"] == "price":
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
                        assert note not in json.dumps(rec["histories"][F], ensure_ascii=False)

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
    hist = rec["histories"]["A"]
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
    text = "\n".join(m["content"] for m in rec["histories"]["A"])
    assert "Period 20 is over. The computer threw the virtual dice" in text
    assert "Period 22 is over. The computer threw the virtual dice and it showed 6" in text
    assert "Period 21 is over. The computer threw the virtual dice and it showed 6" in text
    assert "The experiment is over." in rec["histories"]["A"][-1]["content"]
    assert not {"21", "22"} & set(re.findall(r"(?<![\d,])\d+(?![\d,])", rec["histories"]["A"][0]["content"]))


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


async def test_atomic_write(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    engine.write_json_atomic(path, {"a": 1})

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(engine.json, "dump", boom)
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
        last = body["messages"][-1]["content"][-1]["text"]
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
            path = await run_market(s, client, tmp_path, ExperimentConfig(part_lengths=(2, 1)))
    rec = json.loads(path.read_text())
    assert [p["prices"] for p in rec["periods"]] == [{"A": 99, "B": 99}] * 3
    b = bodies[0]
    assert b["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert b["top_k"] == 20 and b["temperature"] == 0.6 and isinstance(b["seed"], int)
    assert "cache_control" in b["messages"][-1]["content"][-1]
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
        last = json.loads(request.content)["messages"][-1]["content"][-1]["text"]
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
    seen: dict[str, list] = {}
    for line in log:
        L = line["tag"]["firm"]
        msgs = reconstruct_request(rec, L, line["n_messages"], line["cache_upto"])
        assert messages_sha256(msgs) == line["messages_sha256"]
        assert line["prompt_bytes"] == sum(len(m["content"].encode()) for m in rec["histories"][L][:line["n_messages"]])
        full = L not in seen or (line["error"] is not None and not line.get("rate_limited"))
        assert ("request" in line) == full, line["tag"]
        if full:
            assert line["request"]["messages"] == msgs
        else:  # the appended messages extend the previous logged request to this one
            assert line["messages_from"] == len(seen[L])
            assert seen[L] + line["messages_new"] == rec["histories"][L][:line["n_messages"]]
            assert line["request_params"]["seed"] == line["seed"] and "messages" not in line["request_params"]
        seen[L] = rec["histories"][L][:line["n_messages"]]
    assert sum("request" in x for x in log) == 2 + 1  # one per firm, plus the HTTP 500


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
    a = [json.loads(m["content"]) for m in rec["histories"]["A"] if m["role"] == "assistant"]
    b = [json.loads(m["content"]) for m in rec["histories"]["B"] if m["role"] == "assistant"]
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
