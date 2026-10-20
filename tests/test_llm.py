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
    return {"tag": tag, "provider_name": name, "context_length": 1000, "max_completion_tokens": 500,
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
                             backoff_base_s=0, **kw)
        clients.append(c)
        return c

    yield _make
    for c in clients:
        await c.aclose()


def lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]


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
    assert lines(tmp_path)[0]["event"] == "budget_recovered_in_flight"


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
