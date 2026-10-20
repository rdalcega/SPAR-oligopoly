import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import project_cost as pc  # noqa: E402


def call(phase, prompt, out, latency, cached=0, content="x" * 100, model="acme/m", error=None):
    return {"model_requested": model, "error": error, "tag": {"phase": phase}, "latency_s": latency,
            "request": {"messages": [{"role": "user", "content": content}]},
            "usage": {"prompt_tokens": prompt, "completion_tokens": out, "cached_tokens": cached}}


def test_pilot_stats_and_pricing(tmp_path):
    log = tmp_path / "calls.jsonl"
    rows = [call("chat1", 30, 900, 6.0), call("price", 20, 600, 4.0), call("chat2", 10, 1, 1, error="429"),
            call("price", 99, 99, 9, model="other"), {"event": "rate_limit_wait"}]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    st = pc.pilot_stats([log], "acme/m")
    assert st.calls == 2 and st.tokens_per_byte == pytest.approx(50 / 200) and st.cached_share == 0
    assert st.output == {"chat": [900], "price": [600]} and st.latency == {"chat": [6.0], "price": [4.0]}
    calls = [{"bytes": 400, "bound": 300, "kind": "chat"}, {"bytes": 800, "bound": 500, "kind": "price"}]
    pricing = {"prompt": 1e-6, "completion": 2e-6, "input_cache_read": 1e-7}
    usd, ptok, otok = pc.price_calls(calls, st, pricing, bound_tpb=False, seed=1)
    assert (ptok, otok) == (300, 1500) and usd == pytest.approx(300e-6 + 1500 * 2e-6)
    usd_b, ptok_b, _ = pc.price_calls(calls, st, pricing, bound_tpb=True, seed=1)
    assert ptok_b == 800 and usd_b > usd
    st.cached_tokens = 25  # half the prompt tokens were cache hits: cache-read pricing applies to that share
    usd_c, _, _ = pc.price_calls(calls, st, pricing, bound_tpb=False, seed=1)
    assert usd_c == pytest.approx(150e-6 + 150e-7 + 1500 * 2e-6)


def test_steps_count_rounds_prices_and_reasks():
    c = [{"step": (1, 1, "chat1"), "attempt": 0}, {"step": (1, 1, "chat1"), "attempt": 0},
         {"step": (1, 1, "chat2"), "attempt": 2}, {"step": (1, 1, "price"), "attempt": 0},
         {"step": (1, 2, "price"), "attempt": 1}]
    assert pc.steps(c) == {"chat": 1 + 3, "price": 1 + 2}


def test_excluded_cells_not_projected():
    from oligopoly.config import ModelConfig

    cfg = ModelConfig("m", "acme/m", "p", exclude_cells=(("NT-T", 8), ("T-NT", 8)))
    todo, excluded = pc.cells_to_project(cfg, [4, 8], ["NT-T", "T-NT", "NT-NT"])
    assert todo == [(4, "NT-T"), (4, "T-NT"), (4, "NT-NT"), (8, "NT-NT")]
    assert excluded == [(8, "NT-T"), (8, "T-NT")]
