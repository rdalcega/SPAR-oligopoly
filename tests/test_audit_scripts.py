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


def test_scripts_skip_checkpoints(tmp_path):
    markets = tmp_path / "markets"
    markets.mkdir()
    for name in ("x__m0.json", "x__m1.failed.json", "x__m2.ckpt.json"):
        (markets / name).write_text("{}")
    assert [f.name for f in ap.market_files(tmp_path)] == ["x__m0.json", "x__m1.failed.json"]
    assert pctx.load_samples([markets / "x__m1.failed.json", markets / "x__m2.ckpt.json"]) == {}
