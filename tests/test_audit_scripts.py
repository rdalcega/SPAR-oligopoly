import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_pilot as ap  # noqa: E402
import project_context as pctx  # noqa: E402

from oligopoly.llm import messages_sha256, with_cache_control  # noqa: E402


def test_audit_reads_compact_log_lines(tmp_path):
    sysm, u0 = {"role": "system", "content": "rules"}, {"role": "user", "content": "q0"}
    a0, u1 = {"role": "assistant", "content": "a0"}, {"role": "user", "content": "q1"}
    tag = {"run_id": "r", "market": "m", "firm": "A"}
    first = with_cache_control([sysm, u0], -1)
    second = with_cache_control([sysm, u0, a0, u1], -1)
    rows = [{"event": "budget_recovered_in_flight", "invocation": "i"},
            {"invocation": "i", "tag": tag, "cache_upto": -1, "request": {"messages": first}, "prompt_bytes": 7},
            {"event": "rate_limit_wait", "invocation": "i", "tag": tag},
            {"invocation": "i", "tag": tag, "cache_upto": -1, "messages_from": 2, "messages_new": [a0, u1],
             "messages_sha256": messages_sha256(second), "prompt_bytes": 11},
            {"invocation": "j", "tag": tag, "cache_upto": -1, "messages_from": 2, "messages_new": [a0, u1]}]
    log = tmp_path / "calls.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    calls = ap.load_calls(log)
    assert calls[3]["request"]["messages"] == second
    assert "reconstruct_error" in calls[4] and "request" not in calls[4]  # no full line in invocation j
    assert ap.request_bytes(calls[3]) == 11


async def test_context_projection_covers_the_probe():
    from collections import Counter

    from oligopoly.config import ExperimentConfig, ModelConfig

    sample = pctx.Sample(notes=["n"], messages=["hi"], prices=[100, 99], rounds_used=Counter({1: 1, 3: 1}),
                         passes=1, posts=1)
    cfg = ModelConfig("m", "acme/m", "p", max_tokens=10)
    # the probe (answered by the replay client) is the last and longest prompt
    assert await pctx.project(cfg, sample, 2, "NT-T", ExperimentConfig(part_lengths=(3, 2)), 1) > \
        await pctx.project(cfg, sample, 2, "NT-T", ExperimentConfig(part_lengths=(3, 2), post_game_probe=False), 1)


async def test_audit_rebuilds_windowed_requests_and_checks_the_log(tmp_path):
    """On a windowed-context run, the audit rebuilds every attempt's request from the record, matches it to the
    record's hash and to the call log's, and runs the leak checks on it; tampering is caught."""
    import httpx
    import respx

    from oligopoly.config import ExperimentConfig, ModelConfig
    from oligopoly.engine import MarketSpec, run_market
    from oligopoly.llm import BASE_URL, Budget, OpenRouterClient

    cfg = ModelConfig("tiny", "acme/tiny", "deepinfra/fp8", max_tokens=64)
    ep = {"tag": "deepinfra/fp8", "provider_name": "DeepInfra", "pricing": {"prompt": "0.000001", "completion": "0"},
          "supported_parameters": ["seed"]}

    def reply(request):
        body = json.loads(request.content)
        last, firm = body["messages"][-1]["content"], "A" if "You are Firm A" in str(body["messages"][0]) else "B"
        if "Please answer these questions" in last:
            content = json.dumps({"max_consumer_price": 100, "best_common_price": 100,
                                  "last_period_prices": {"A": 99, "B": 99}, "deviations": "none"})
        elif '"price": <integer' in last.splitlines()[-1]:
            content = json.dumps({"notes": f"secret plan number {len(last)} of firm {firm}, private", "price": 99})
        else:
            content = json.dumps({"notes": f"chat plan {len(last)} for firm {firm} kept private", "message": "hi all"})
        return httpx.Response(200, json={"id": "g", "model": "acme/tiny", "provider": "DeepInfra",
                                         "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0}})

    run = tmp_path / "runs" / "live"
    with respx.mock() as r:
        r.get(f"{BASE_URL}/models/acme/tiny/endpoints").mock(
            return_value=httpx.Response(200, json={"data": {"endpoints": [ep]}}))
        r.post(f"{BASE_URL}/chat/completions").mock(side_effect=reply)
        async with OpenRouterClient(run / "calls.jsonl", Budget(1.0), api_key="k") as client:
            path = await run_market(MarketSpec("live", cfg, 2, "T-NT", 0, 1), client, run,
                                    ExperimentConfig(part_lengths=(4, 2)))
    out = tmp_path / "audit.json"
    assert ap.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out), "live"]) == 0
    rep = json.loads(out.read_text())["live/" + path.stem]
    assert rep["engine"]["issues"] == [] and rep["calls"]["issues"] == []
    assert rep["calls"]["ok_calls"] == rep["calls"]["checked_against_log"] == rep["usage_record"]["calls"]

    rec = json.loads(path.read_text())
    calls = [c for c in ap.load_calls(run / "calls.jsonl") if "event" not in c]
    b_notes = rec["periods"][0]["decisions"]["B"]["notes"]
    rec["transcripts"]["A"][-2]["content"] += " " + b_notes  # tamper: B's private notes in A's last turns
    issues = ap.check_calls(rec, calls)["issues"]
    assert any("does not match its context hash" in x for x in issues)
    assert any("notes of B visible" in x for x in issues)


def test_worst_case_note_is_exactly_at_the_cap():
    from oligopoly.engine import cap_notes

    for words, chars in ((200, 1400), (200, 1000), (10, 1400)):
        note = pctx.cap_note(words, chars)
        assert len(note.split()) == words and len(note) == chars and not cap_notes(note, words, chars)[1]


def test_scripts_skip_checkpoints(tmp_path):
    markets = tmp_path / "markets"
    markets.mkdir()
    for name in ("x__m0.json", "x__m1.failed.json", "x__m2.ckpt.json"):
        (markets / name).write_text("{}")
    assert [f.name for f in ap.market_files(tmp_path)] == ["x__m0.json", "x__m1.failed.json"]
    assert pctx.load_samples([markets / "x__m1.failed.json", markets / "x__m2.ckpt.json"]) == {}
