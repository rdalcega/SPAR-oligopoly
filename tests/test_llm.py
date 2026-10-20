import asyncio
import json

import httpx
import pytest
import respx

from oligopoly.llm import (
    BASE_URL,
    PROMPT_SLACK_TOKENS,
    Budget,
    BudgetExceeded,
    Endpoint,
    OpenRouterClient,
    PermanentError,
    RetriesExhausted,
    estimate_cost,
    with_cache_control,
)

MODEL = "acme/tiny"
CHAT = f"{BASE_URL}/chat/completions"
ENDPOINTS = f"{BASE_URL}/models/{MODEL}/endpoints"
MSGS = [{"role": "system", "content": "You are firm A."}, {"role": "user", "content": "Price?"}]


def endpoint(tag, name, seed=True, prompt="0.000001", completion="0.000002"):
    params = ["temperature", "max_tokens"] + (["seed"] if seed else [])
    return {"tag": tag, "provider_name": name, "context_length": 200_000, "max_completion_tokens": 500,
            "quantization": "fp8", "pricing": {"prompt": prompt, "completion": completion},
            "supported_parameters": params, "status": 0}


ENDPOINTS_BODY = {"data": {"id": MODEL, "endpoints": [endpoint("deepinfra/fp8", "DeepInfra"),
                                                      endpoint("novita", "Novita", seed=False)]}}


def completion(provider="DeepInfra", content="42", cost=0.0001, choices=True):
    return {
        "id": "gen-123", "model": MODEL, "provider": provider,
        "choices": [{"finish_reason": "stop", "native_finish_reason": "stop",
                     "message": {"role": "assistant", "content": content, "reasoning": "think"}}] if choices else [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": cost,
                  "prompt_tokens_details": {"cached_tokens": 4},
                  "completion_tokens_details": {"reasoning_tokens": 1}},
    }


@pytest.fixture
def mock():
    with respx.mock(assert_all_called=False) as r:
        r.get(ENDPOINTS).mock(return_value=httpx.Response(200, json=ENDPOINTS_BODY))
        yield r


@pytest.fixture
async def make(tmp_path):
    clients = []

    def _make(cap=1.0, **kw):
        c = OpenRouterClient(tmp_path / "calls.jsonl", Budget(cap, tmp_path / "budget.json"), api_key="sk-test",
                             **{"backoff_base_s": 0, "rate_limit_backoff_base_s": 0, **kw})
        clients.append(c)
        return c

    yield _make
    for c in clients:
        await c.aclose()


def lines(tmp_path, events=False):
    recs = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    return recs if events else [r for r in recs if "event" not in r]


async def test_success_payload_and_result(mock, make, tmp_path):
    route = mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    c = make()
    r = await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", temperature=0.7, top_p=0.9, max_tokens=50,
                     seed=7, reasoning={"effort": "low"}, tag={"market": 1, "firm": "A", "period": 3})
    sent = json.loads(route.calls[0].request.content)
    assert sent["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert sent["usage"] == {"include": True}
    assert sent["seed"] == 7 and sent["reasoning"] == {"effort": "low"} and sent["top_p"] == 0.9
    assert "response_format" not in sent
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
    assert (r.content, r.reasoning, r.provider, r.generation_id) == ("42", "think", "DeepInfra", "gen-123")
    assert r.usage.cached_tokens == 4 and r.usage.reasoning_tokens == 1
    assert r.cost_usd == pytest.approx(0.0001)
    assert not r.seed_possibly_ignored and not r.provider_mismatch


async def test_log_fields(mock, make, tmp_path):
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=50, seed=1, tag={"firm": "B"})
    [rec] = lines(tmp_path)
    for k in ("ts", "tag", "attempt", "model_requested", "model_served", "provider_requested", "provider_served",
              "sampling", "seed", "reasoning_setting", "request", "messages_sha256", "content", "reasoning",
              "usage", "cost_usd", "latency_s", "error", "seed_possibly_ignored", "provider_mismatch"):
        assert k in rec, k
    assert rec["tag"] == {"firm": "B"}
    assert rec["request"]["messages"] == MSGS
    assert rec["error"] is None and rec["attempt"] == 1
    assert "sk-test" not in (tmp_path / "calls.jsonl").read_text()


async def test_seed_flag_unsupported_endpoint(mock, make):
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion(provider="Novita")))
    r = await make().chat(MSGS, model=MODEL, provider="novita", max_tokens=10, seed=3)
    assert r.seed_possibly_ignored and not r.provider_mismatch


async def test_seed_flag_provider_mismatch(mock, make, tmp_path):
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion(provider="Novita")))
    r = await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10, seed=3)
    assert r.seed_possibly_ignored and r.provider_mismatch
    assert lines(tmp_path)[0]["seed_flag_reason"] == "served provider differs from pinned"


async def test_no_seed_no_flag(mock, make):
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion(provider="Novita")))
    r = await make().chat(MSGS, model=MODEL, provider="novita", max_tokens=10)
    assert not r.seed_possibly_ignored


async def test_retries_then_succeeds(mock, make, tmp_path):
    mock.post(CHAT).mock(side_effect=[
        httpx.Response(429, json={"error": {"code": 429, "message": "slow down"}}),
        httpx.Response(502, text="bad gateway"),
        httpx.ReadTimeout("timeout"),
        httpx.Response(200, json=completion(choices=False, cost=0)),
        httpx.Response(200, json=completion()),
    ])
    r = await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert r.attempts == 5
    recs = lines(tmp_path)
    assert [x["attempt"] for x in recs] == [1, 2, 3, 4, 5]
    assert all(x["error"] for x in recs[:4]) and recs[4]["error"] is None


async def test_retries_exhausted(mock, make):
    mock.post(CHAT).mock(return_value=httpx.Response(503, text="down"))
    with pytest.raises(RetriesExhausted):
        await make(max_retries=2).chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)


async def test_permanent_error_not_retried(mock, make, tmp_path):
    route = mock.post(CHAT).mock(return_value=httpx.Response(400, json={"error": {"code": 400, "message": "bad"}}))
    with pytest.raises(PermanentError):
        await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert route.call_count == 1
    assert lines(tmp_path)[0]["retryable"] is False


async def test_unknown_provider_tag(mock, make):
    with pytest.raises(PermanentError):
        await make().chat(MSGS, model=MODEL, provider="nope", max_tokens=10)


async def test_budget_refusal_and_persistence(mock, make, tmp_path):
    route = mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion(cost=0.006)))
    c = make(cap=0.01)
    await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)  # spent 0.012 >= cap
    with pytest.raises(BudgetExceeded):
        await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert route.call_count == 2
    assert json.loads((tmp_path / "budget.json").read_text())["spent_usd"] == pytest.approx(0.012)
    assert "BudgetExceeded" in lines(tmp_path)[-1]["error"]
    assert Budget(1.0, tmp_path / "budget.json").spent == pytest.approx(0.012)


async def test_budget_refuses_large_reservation(mock, make):
    route = mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    with pytest.raises(BudgetExceeded):  # 100k completion tokens * $2e-6 = $0.2 > cap
        await make(cap=0.05).chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=100_000)
    assert route.call_count == 0


def test_cache_control():
    out = with_cache_control(MSGS, 0)
    assert out[0]["content"] == [{"type": "text", "text": "You are firm A.",
                                  "cache_control": {"type": "ephemeral"}}]
    assert out[1] == MSGS[1] and MSGS[0]["content"] == "You are firm A."
    out = with_cache_control(MSGS, [0, -1], ttl="1h")
    assert out[1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_cache_upto_in_payload(mock, make):
    route = mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10, cache_upto=0)
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------- review regressions


def post(c, **kw):
    return c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", **{"max_tokens": 10, **kw})


async def test_cancel_charges_reservation_and_logs(mock, make, tmp_path):
    started = asyncio.Event()

    async def hang(request):
        started.set()
        await asyncio.sleep(60)

    mock.post(CHAT).mock(side_effect=hang)
    c = make()
    task = asyncio.create_task(post(c))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [rec] = lines(tmp_path)
    assert rec["error"] == "cancelled" and rec["charged_usd"] == rec["reserved_usd"] > 0
    assert not c.budget.in_flight and c.budget.spent == pytest.approx(rec["reserved_usd"])


async def test_unexpected_error_charged_logged_reraised(mock, make, tmp_path):
    mock.post(CHAT).mock(side_effect=RuntimeError("boom"))
    c = make()
    with pytest.raises(RuntimeError):
        await post(c)
    [rec] = lines(tmp_path)
    assert rec["error"] == "RuntimeError: boom" and rec["charged_usd"] == rec["reserved_usd"]
    assert not c.budget.in_flight


async def test_unsent_request_released(mock, make, tmp_path):
    mock.post(CHAT).mock(side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=completion(cost=0))])
    c = make()
    await post(c)
    assert lines(tmp_path)[0]["charged_usd"] == 0.0 and c.budget.spent == 0.0 and not c.budget.in_flight


def test_crashed_in_flight_charged_on_restart(tmp_path):
    b = Budget(1.0, tmp_path / "budget.json")
    b.reserve(0.25)
    b.settle(b.reserve(0.1), 0.05)
    assert json.loads((tmp_path / "budget.json").read_text())["in_flight"]  # the 0.25 is persisted
    b2 = Budget(1.0, tmp_path / "budget.json")  # the process died with 0.25 in flight
    assert b2.spent == pytest.approx(0.30) and b2.recovered_usd == pytest.approx(0.25) and not b2.in_flight
    assert not json.loads((tmp_path / "budget.json").read_text())["in_flight"]
    OpenRouterClient(tmp_path / "calls.jsonl", b2, api_key="k")
    assert lines(tmp_path, events=True)[0]["event"] == "budget_recovered_in_flight"


def test_budget_cap_counts_reservations():
    b = Budget(1.0)
    b.reserve(0.6)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.5)
    b.settle(b.reserve(0.4), 2.0)  # an actual cost above the reservation is charged in full
    assert b.spent == 2.0
    with pytest.raises(BudgetExceeded):
        b.reserve(0.0)


async def test_max_tokens_required(mock, make):
    with pytest.raises(ValueError, match="max_tokens"):
        await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8")


def test_estimate_is_byte_bound():
    ep = Endpoint.from_api(endpoint("x", "X", prompt="0.000001", completion="0.000002"))
    msgs = [{"role": "user", "content": "€" * 100}]  # 300 UTF-8 bytes
    n_bytes = len(json.dumps(msgs, ensure_ascii=False).encode())
    assert n_bytes > 300
    assert estimate_cost(ep, msgs, 50) == pytest.approx((n_bytes + PROMPT_SLACK_TOKENS) * 1e-6 + 50 * 2e-6)
    with pytest.raises(PermanentError):
        estimate_cost(Endpoint.from_api(endpoint("x", "X", prompt="-1")), msgs, 50)


async def test_unknown_pricing_fails_closed(make, tmp_path):
    body = {"data": {"endpoints": [{**endpoint("deepinfra/fp8", "DeepInfra"), "pricing": {"prompt": "0.000001"}}]}}
    with respx.mock(assert_all_called=False) as r:
        r.get(ENDPOINTS).mock(return_value=httpx.Response(200, json=body))
        route = r.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
        with pytest.raises(PermanentError, match="pricing"):
            await post(make())
    assert route.call_count == 0 and "pricing" in lines(tmp_path)[0]["error"]


async def test_reserve_inside_semaphore(mock, make):
    release = asyncio.Event()

    async def slow(request):
        await release.wait()
        return httpx.Response(200, json=completion(cost=0))

    mock.post(CHAT).mock(side_effect=slow)
    c = make(max_concurrency=1)
    tasks = [asyncio.create_task(post(c)) for _ in range(3)]
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(c.budget.in_flight) == 1  # calls waiting for the semaphore hold no reservation
    release.set()
    await asyncio.gather(*tasks)


async def test_timeouts(mock, make, tmp_path):
    seen = []

    def reply(request):
        seen.append(request.extensions["timeout"]["read"])
        if len(seen) == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=completion())

    mock.post(CHAT).mock(side_effect=reply)
    c = make()
    await post(c, timeout_s=5)
    await post(c)
    assert seen == [5, 5, 900]
    recs = lines(tmp_path)
    assert recs[0]["charged_usd"] == recs[0]["reserved_usd"]  # a timed-out request may have been billed
    assert [r["timeout_retry"] for r in recs] == [False, True, False]


async def test_malformed_200_retried(mock, make, tmp_path):
    mock.post(CHAT).mock(side_effect=[httpx.Response(200, json=[1, 2]), httpx.Response(200, json={"id": "x"}),
                                      httpx.Response(200, text="not json"), httpx.Response(200, json=completion())])
    r = await post(make())
    assert r.attempts == 4
    assert all("malformed" in x["error"] and x["retryable"] for x in lines(tmp_path)[:3])


async def test_unknown_provider_tag_logged(mock, make, tmp_path):
    with pytest.raises(PermanentError):
        await make().chat(MSGS, model=MODEL, provider="nope", max_tokens=10, tag={"firm": "A"})
    [rec] = lines(tmp_path)
    assert "nope" in rec["error"] and rec["tag"] == {"firm": "A"} and rec["retryable"] is False


def test_prompt_token_bound():
    from oligopoly.llm import TEMPLATE_TOKENS_PER_MESSAGE, prompt_token_bound

    msgs = [{"role": "system", "content": "x" * 1000},
            {"role": "user", "content": [{"type": "text", "text": "é" * 50}]}]  # 1000 + 100 UTF-8 bytes
    extra = 2 * TEMPLATE_TOKENS_PER_MESSAGE + PROMPT_SLACK_TOKENS
    assert prompt_token_bound(msgs) == 440 + extra
    assert prompt_token_bound(msgs, 2.0) == 550 + extra


async def test_context_overflow_is_not_sent(mock, make, tmp_path):
    from oligopoly.llm import ContextOverflow, prompt_token_bound

    route = mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    big = [{"role": "user", "content": "x" * 400_000}]  # bound ~160,000 tokens; the context is 200,000
    assert prompt_token_bound(big) + 30_000 < 200_000
    await make().chat(big, model=MODEL, provider="deepinfra/fp8", max_tokens=30_000)
    assert route.call_count == 1
    with pytest.raises(ContextOverflow):  # the same prompt no longer fits with a larger completion allowance
        await make().chat(big, model=MODEL, provider="deepinfra/fp8", max_tokens=40_000)
    with pytest.raises(ContextOverflow):  # nor under a lower bytes-per-token assumption
        await make().chat(big, model=MODEL, provider="deepinfra/fp8", max_tokens=30_000, bytes_per_token_min=2.0)
    assert route.call_count == 1
    logged = lines(tmp_path)
    assert [x["error"].startswith("ContextOverflow") for x in logged if x.get("error")] == [True, True]
    assert logged[-1]["context_guard"]["context_length"] == 200_000 and "status_code" not in logged[-1]


@pytest.fixture
def sleeps(monkeypatch):
    """Record the client's backoff sleeps without waiting."""
    import oligopoly.llm as llm
    real, waits = asyncio.sleep, []

    async def fake(d, *a, **kw):
        waits.append(d)
        await real(0)

    monkeypatch.setattr(llm.asyncio, "sleep", fake)
    return waits


async def test_rate_limits_retried_patiently_beyond_max_retries(mock, make, tmp_path, sleeps):
    overloaded = httpx.Response(503, json={"error": {"code": 503, "message": "engine_overloaded"}})
    mock.post(CHAT).mock(side_effect=[httpx.Response(429, json={"error": {"code": 429, "message": "slow"}},
                                                     headers={"Retry-After": "300"})]
                         + [overloaded] * 9 + [httpx.Response(502, text="bad"), httpx.Response(200, json=completion())])
    c = make(max_retries=1, rate_limit_backoff_base_s=5, rate_limit_backoff_max_s=120, rate_limit_max_wait_s=1e6)
    r = await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert r.attempts == 12
    assert sleeps[0] == 300  # Retry-After honoured even above the cap
    assert 5 <= sleeps[1] <= 10 and all(w <= 120 for w in sleeps[1:10]) and sleeps[9] >= 60  # exp. with jitter
    waits = [x for x in lines(tmp_path, events=True) if x.get("event") == "rate_limit_wait"]
    assert len(waits) == 10 and waits[-1]["rate_limit_retry"] == 10
    assert [x["rate_limited"] for x in lines(tmp_path)[:11]] == [True] * 10 + [False]


async def test_rate_limit_gives_up_after_total_wait(mock, make, tmp_path, sleeps):
    route = mock.post(CHAT).mock(return_value=httpx.Response(429, json={"error": {"code": 429, "message": "x"}}))
    c = make(max_retries=0, rate_limit_backoff_base_s=100, rate_limit_backoff_max_s=120, rate_limit_max_wait_s=500)
    with pytest.raises(RetriesExhausted, match="rate limited"):
        await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert sum(sleeps) == pytest.approx(500) and route.call_count == len(sleeps) + 1


async def test_non_rate_limit_errors_keep_max_retries(mock, make, sleeps):
    route = mock.post(CHAT).mock(return_value=httpx.Response(502, text="bad gateway"))
    with pytest.raises(RetriesExhausted):
        await make(max_retries=2).chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert route.call_count == 3


async def test_sampling_log_includes_extra_body(mock, make, tmp_path):
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    await make().chat(MSGS, model=MODEL, provider="deepinfra/fp8", temperature=0.6, top_p=0.95, max_tokens=10,
                      seed=3, extra_body={"top_k": 20, "presence_penalty": 1.5, "min_p": 0.0})
    s = lines(tmp_path)[0]["sampling"]
    assert s == {"temperature": 0.6, "top_p": 0.95, "max_tokens": 10, "top_k": 20, "presence_penalty": 1.5,
                 "min_p": 0.0}


async def test_error_body_200_without_cost_not_charged(mock, make, tmp_path):
    """A 200 whose body is only an error object (no choices, no usage.cost) is not billed by OpenRouter: each
    retry settles at 0 instead of its full reservation."""
    err = httpx.Response(200, json={"error": {"code": 502, "message": "upstream error"}})
    mock.post(CHAT).mock(side_effect=[err, err, httpx.Response(200, json=completion(cost=0.0001))])
    c = make(cap=1.0)
    r = await post(c)
    assert r.attempts == 3
    recs = lines(tmp_path)
    assert [x["charged_usd"] for x in recs[:2]] == [0.0, 0.0]
    assert [x["charge_reason"] for x in recs] == ["error_body", "error_body", "usage_cost"]
    assert c.budget.spent == pytest.approx(0.0001)


async def test_200_without_cost_still_charged_reservation(mock, make, tmp_path):
    body = completion()
    del body["usage"]["cost"]
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=body))
    c = make(cap=1.0)
    await post(c)
    [rec] = lines(tmp_path)
    assert rec["charge_reason"] == "cost_unknown_reservation" and rec["charged_usd"] == rec["reserved_usd"] > 0


async def test_rate_limit_total_attempts_capped(mock, make, tmp_path, sleeps):
    route = mock.post(CHAT).mock(return_value=httpx.Response(429, json={"error": {"code": 429, "message": "x"}},
                                                             headers={"Retry-After": "0"}))
    c = make(max_retries=0, rate_limit_max_wait_s=1e6, max_attempts=7)
    with pytest.raises(RetriesExhausted, match="7 attempts"):
        await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10)
    assert route.call_count == 7
    assert make().max_attempts == 60  # the default


async def test_admit_waits_under_reservation_pressure():
    b = Budget(1.0)
    first = b.reserve(0.6)
    waiter = asyncio.create_task(b.admit(0.5))  # fits after spending, not alongside the 0.6 reservation
    await asyncio.sleep(0)
    assert not waiter.done() and len(b.in_flight) == 1
    b.settle(first, 0.3)  # 0.3 spent + 0.5 <= 1.0: admitted
    rid = await asyncio.wait_for(waiter, 1)
    assert b.in_flight == {rid: 0.5} and b.spent + b.reserved <= b.cap
    # a waiter that turns out not to fit once the in-flight call settles is refused (true exhaustion)
    waiter = asyncio.create_task(b.admit(0.5))
    await asyncio.sleep(0)
    assert not waiter.done()
    b.settle(rid, 0.5)  # spent 0.8 + 0.5 > 1.0
    with pytest.raises(BudgetExceeded, match="exhausted"):
        await asyncio.wait_for(waiter, 1)
    with pytest.raises(BudgetExceeded):  # nothing in flight: refused at once
        await b.admit(0.3)
    assert not b.in_flight


async def test_near_cap_concurrent_calls_wait_instead_of_failing(mock, make, tmp_path):
    """Two concurrent calls whose reservations do not fit together, but whose actual costs do, both succeed;
    the cap is never exceeded by spent + reserved."""
    release = asyncio.Event()
    peak = []

    async def slow(request):
        peak.append(c.budget.spent + c.budget.reserved)
        await release.wait()
        return httpx.Response(200, json=completion(cost=0.00001))

    mock.post(CHAT).mock(side_effect=slow)
    est = estimate_cost(Endpoint.from_api(endpoint("deepinfra/fp8", "DeepInfra")), MSGS, 10)
    c = make(cap=1.5 * est)
    tasks = [asyncio.create_task(post(c)) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(c.budget.in_flight) == 1  # the others wait for room
    release.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), 5)
    assert all(r.content == "42" for r in results) and not c.budget.in_flight
    assert max(peak) <= c.budget.cap
    assert any(x.get("event") == "budget_wait" for x in lines(tmp_path, events=True))


# ---------------------------------------------------------------- budget fail-closed, admission bound, log chains


def near_cap_client(make, **kw):
    """A client whose cap fits one reservation at a time (so a second concurrent call waits in admit)."""
    est = estimate_cost(Endpoint.from_api(endpoint("deepinfra/fp8", "DeepInfra")), MSGS, 10)
    return make(cap=1.5 * est, **kw), est


def two_firm_market(client):
    from oligopoly import engine
    from oligopoly.config import ExperimentConfig, ModelConfig

    return engine.Market(engine.MarketSpec("t", ModelConfig("m", MODEL, "deepinfra/fp8"), 2, "NT-NT", 0, 1),
                         client, ExperimentConfig())


async def test_malformed_cost_settles_and_waiter_proceeds(mock, make, tmp_path):
    """usage.cost that is not a number is charged the full reservation (it may have been billed); the
    reservation is always settled, so a call waiting for budget room is not left hanging."""
    body = completion()
    body["usage"]["cost"] = "invalid"

    async def slow(request):
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=body)

    mock.post(CHAT).mock(side_effect=slow)
    c, est = near_cap_client(make)  # B waits for A; A is charged its reservation, so B no longer fits
    with pytest.raises(BudgetExceeded, match="exhausted"):
        await asyncio.wait_for(two_firm_market(c).gather({"A": post(c), "B": post(c)}), 5)
    assert any(x.get("event") == "budget_wait" for x in lines(tmp_path, events=True))
    ok, refused = lines(tmp_path)
    assert ok["charge_reason"] == "cost_invalid_reservation" and ok["cost_raw"] == "invalid"
    assert ok["charged_usd"] == ok["reserved_usd"] and ok["cost_usd"] is None and ok["error"] is None
    assert "exhausted" in refused["error"]
    assert not c.budget.in_flight and c.budget.spent == pytest.approx(est)
    c.budget.cap = 3 * est  # with room, the same response is accepted with cost_usd None
    assert (await post(c)).cost_usd is None


async def test_settle_persist_failure_fails_closed(mock, make, tmp_path):
    """If the budget file cannot be written when a call settles, the budget is marked unusable, waiters are
    woken and fail closed (BudgetUnusable is a BudgetExceeded, so the runner stops), and Market.gather exits."""
    from oligopoly.llm import BudgetUnusable

    release = asyncio.Event()

    async def slow(request):
        await release.wait()
        return httpx.Response(200, json=completion(cost=0.0))

    mock.post(CHAT).mock(side_effect=slow)
    c, _ = near_cap_client(make)
    task = asyncio.create_task(two_firm_market(c).gather({"A": post(c), "B": post(c)}))
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(c.budget.in_flight) == 1  # B waits for room

    def disk_full():
        raise OSError("disk full")

    c.budget._persist = disk_full
    release.set()
    with pytest.raises(BudgetUnusable):
        await asyncio.wait_for(task, 5)
    assert c.budget.broken and "disk full" in c.budget.broken and not c.budget.in_flight
    with pytest.raises(BudgetExceeded):
        await post(c)


def test_reserve_persist_failure_drops_reservation(tmp_path):
    from oligopoly.llm import BudgetUnusable

    b = Budget(1.0, tmp_path / "budget.json")

    def disk_full():
        raise OSError("disk full")

    b._persist = disk_full
    with pytest.raises(BudgetUnusable):
        b.reserve(0.1)
    assert not b.in_flight and b.broken
    b.settle("unknown", 1.0)  # never raises
    assert b.spent == 0.0


async def test_admit_wait_is_bounded_and_reports_wait_once():
    from oligopoly.llm import BudgetWaitTimeout

    b = Budget(1.0)
    b.reserve(0.6)  # never settled
    waits = []
    with pytest.raises(BudgetWaitTimeout, match="waited") as e:  # not exhaustion: the run keeps going
        await asyncio.wait_for(b.admit(0.5, max_wait_s=0.05, on_wait=lambda: waits.append(1)), 5)
    assert not isinstance(e.value, BudgetExceeded)
    assert waits == [1] and len(b.in_flight) == 1
    assert await b.admit(0.3, on_wait=lambda: waits.append(2)) and waits == [1]  # fits: no wait reported


async def test_admission_wait_outlasts_the_longest_deadline(mock, make, tmp_path):
    """The default admission backstop is 2 x the longest per-attempt deadline used (at least DEFAULT_MAX_WAIT_S),
    so a call never gives up waiting while another call's reservation can still legitimately be in flight."""
    c = make()
    assert c.timeout_s == 900 and Budget.DEFAULT_MAX_WAIT_S == 3600 and c.admission_wait_s() == 3600
    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion()))
    await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10, timeout_s=1500)
    assert c.admission_wait_s() == 6000  # deadline 2 x 1500
    await c.chat(MSGS, model=MODEL, provider="deepinfra/fp8", max_tokens=10, deadline_s=4000)
    assert c.admission_wait_s() == 8000
    assert make(budget_max_wait_s=5).admission_wait_s() == 5  # explicit setting wins


async def test_cancel_while_waiting_for_budget_is_logged(mock, make, tmp_path):
    release = asyncio.Event()

    async def slow(request):
        await release.wait()
        return httpx.Response(200, json=completion(cost=0.0))

    mock.post(CHAT).mock(side_effect=slow)
    c, _ = near_cap_client(make)
    first = asyncio.create_task(post(c, tag={"market": "m", "firm": "A"}))
    second = asyncio.create_task(post(c, tag={"market": "m", "firm": "B"}))
    for _ in range(10):
        await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    await first
    recs = lines(tmp_path, events=True)
    assert [x["event"] for x in recs if "event" in x] == ["budget_wait"]
    [cancelled] = [x for x in recs if x.get("tag", {}).get("firm") == "B" and "event" not in x]
    assert cancelled["error"] == "cancelled while waiting for budget room" and cancelled["charged_usd"] == 0.0
    assert "request" in cancelled and not c.budget.in_flight


async def test_log_chains_per_invocation(mock, make, tmp_path):
    """Compact lines chain from the latest full line of the same firm and invocation: a second process whose
    history diverges from what the first process logged (a mid-period resume) still reconstructs exactly."""
    from oligopoly.llm import messages_sha256, reconstruct_from_log, reconstruct_request

    mock.post(CHAT).mock(return_value=httpx.Response(200, json=completion(cost=0.0)))
    tag = {"run_id": "r", "market": "m", "firm": "A"}
    sysm = {"role": "system", "content": "rules"}
    u = [{"role": "user", "content": f"q{i}"} for i in range(4)]
    a = [{"role": "assistant", "content": f"a{i}"} for i in range(4)]

    async def send(client, msgs):
        await client.chat(msgs, model=MODEL, provider="deepinfra/fp8", max_tokens=10, tag=tag, cache_upto=-1)

    dead = make()
    await send(dead, [sysm, u[0]])
    await send(dead, [sysm, u[0], a[0], u[1]])
    await send(dead, [sysm, u[0], a[0], u[1], a[1], u[2]])  # the process dies; the resume replays after u[1]
    alive = make()
    final = [sysm, u[0], a[0], u[1], a[2], u[3]]
    await send(alive, final[:4])
    await send(alive, final)
    window = [sysm, {"role": "user", "content": "memory"}, *final[3:]]  # not a prefix extension: still compact
    await send(alive, window)
    log = lines(tmp_path)
    assert len({x["invocation"] for x in log}) == 2
    assert ["request" in x for x in log] == [True, False, False, True, False, False]
    assert log[5]["messages_ref"] == [0, {"role": "user", "content": "memory"}, 3, 4, 5]
    for i, x in enumerate(log):
        assert messages_sha256(reconstruct_from_log(log, i)) == x["messages_sha256"]
    # the record-based rebuild reads checkpoints too (instructions, transcript and a call's context)
    ck = {"firms": {"A": {"instructions": ["rules"], "transcript": [{**m, "period": 1} for m in final[1:]],
                          "pending": [], "notes": None}}}
    ctx = {"instructions": 1, "memory": "memory", "window": [2, 5]}
    assert messages_sha256(reconstruct_request(ck, "A", ctx, -1)) == log[5]["messages_sha256"]
    assert messages_sha256(reconstruct_request(ck, "A", ctx, -1)) != log[4]["messages_sha256"]
    with pytest.raises(ValueError):
        reconstruct_request(ck, "A", {**ctx, "window": [2, 9]})
    # a broken reference is reported, not guessed
    bad = [*log[:5], {**log[5], "messages_ref": [0, 99]}]
    with pytest.raises(ValueError, match="chain broken"):
        reconstruct_from_log(bad, 5)


async def test_trickling_response_hits_absolute_deadline(make, tmp_path):
    """httpx's float timeout bounds each read; a body trickling a byte at a time must still end at the
    absolute deadline (2 x timeout_s), charged and retried like a read timeout."""
    posts = []

    async def trickle():
        yield b"{"
        while True:
            await asyncio.sleep(0.02)
            yield b" "

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=ENDPOINTS_BODY)
        posts.append(request)
        if len(posts) == 1:
            return httpx.Response(200, content=trickle())
        return httpx.Response(200, json=completion())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    c = make(http=http)
    r = await asyncio.wait_for(post(c, timeout_s=0.2), 5)
    assert r.content == "42" and r.attempts == 2
    first, second = lines(tmp_path)
    assert first["error"].startswith("ReadTimeout: absolute deadline") and first["retryable"]
    assert first["charged_usd"] == first["reserved_usd"] > 0 and first["charge_reason"] == "sent_no_response"
    assert second["timeout_retry"] is True and second["error"] is None
    assert not c.budget.in_flight


def _hanging_after(trace_events, seen):
    """A transport that emits httpcore-style trace events for the first POST, then never answers."""
    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=ENDPOINTS_BODY)
        seen.append(request)
        if len(seen) > 1:
            return httpx.Response(200, json=completion())
        for name in trace_events:
            await request.extensions["trace"](name, {})
        await asyncio.sleep(3600)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_deadline_is_separate_and_larger_than_per_read_timeout(make, tmp_path):
    """httpx keeps timeout_s for connect/pool/read; the absolute deadline defaults to 2 x timeout_s."""
    seen = []
    c = make(http=_hanging_after(["http11.send_request_headers.started"], seen))
    r = await asyncio.wait_for(post(c, timeout_s=0.1), 5)
    assert r.attempts == 2
    assert seen[0].extensions["timeout"] == {"connect": 0.1, "read": 0.1, "write": 0.1, "pool": 0.1}
    first, _ = lines(tmp_path)
    assert first["timeout_s"] == 0.1 and first["deadline_s"] == 0.2
    assert first["latency_s"] >= 0.19
    # sent: charged like a read timeout and retried with timeout_retry
    assert first["error"].startswith("ReadTimeout: absolute deadline") and "(sent)" in first["error"]
    assert first["charged_usd"] == first["reserved_usd"] > 0 and first["charge_reason"] == "sent_no_response"


async def test_deadline_before_request_written_is_not_charged(make, tmp_path):
    """A deadline hit while still connecting (trace shows connection activity, no request write) is NOT_SENT."""
    seen = []
    c = make(http=_hanging_after(["connection.connect_tcp.started", "connection.connect_tcp.complete"], seen),
             deadline_s=0.15)
    r = await asyncio.wait_for(post(c, timeout_s=10), 5)
    assert r.attempts == 2
    first, second = lines(tmp_path)
    assert first["deadline_s"] == 0.15
    assert first["error"].startswith("ConnectTimeout: absolute deadline") and first["retryable"]
    assert first["charged_usd"] == 0.0 and first["charge_reason"] == "not_sent"
    assert second["timeout_retry"] is False
    assert not c.budget.in_flight


async def test_deadline_with_unknown_send_state_is_charged(make, tmp_path):
    """No trace at all (e.g. a pool wait or a transport without tracing): unknown, so charged conservatively."""
    seen = []
    c = make(http=_hanging_after([], seen))
    await asyncio.wait_for(post(c, timeout_s=10, deadline_s=0.1), 5)
    first, second = lines(tmp_path)
    assert "(unknown)" in first["error"] and first["charge_reason"] == "sent_no_response"
    assert second["timeout_retry"] is True
