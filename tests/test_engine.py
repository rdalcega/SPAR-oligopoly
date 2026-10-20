import json
import re

import pytest

from oligopoly import engine, market
from oligopoly.config import ConfigError, ExperimentConfig, ModelConfig, load_models
from oligopoly.engine import MarketSpec, environment_seed, market_path, run_market
from oligopoly.llm import BudgetExceeded, RetriesExhausted
from oligopoly.mock import MockClient
from oligopoly.run import build_specs, main, run_all

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


def test_models_config():
    models = load_models()
    assert "mock" in models and len(models) == 9
    live = [m for m in models.values() if not m.is_mock]
    assert all(m.provider_pending for m in live)
    with pytest.raises(ConfigError):
        live[0].check_live()
    assert models["qwen3.5-9b"].chat_kwargs()["extra_body"] == {"top_k": 20}


async def test_cli(tmp_path, capsys):
    with pytest.raises(ConfigError, match="provider"):
        await main(["--run-id", "x", "--models", "gpt-oss-20b", "--n", "2", "--budget-usd", "1",
                    "--out-root", str(tmp_path)])
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
        content = '{"notes": "n", "price": 99}' if '"price": <integer' in last else '{"notes": "n", "message": null}'
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
