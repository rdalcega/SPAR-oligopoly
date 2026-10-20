import pytest

from oligopoly.parse import extract_object, parse_message, parse_price, strip_reasoning


@pytest.mark.parametrize("content", [
    '{"notes": "hi", "price": 57}',
    '```json\n{"notes": "hi", "price": 57}\n```',
    'Sure! Here is my answer:\n{"notes": "hi", "price": 57}\nGood luck.',
    '<think>maybe {"price": 3}</think>{"notes": "hi", "price": 57}',
    'reasoning without an opening tag {"price": 2} </think>\n{"notes": "hi", "price": 57}',
    "{'notes': 'hi', 'price': 57}",
    '{"price": 10} on second thought: {"notes": "hi", "price": 57}',
    '{"notes": "I\'ll {keep} it", "price": 57}',
])
def test_price_extraction(content):
    p = parse_price(content)
    assert p.ok and p.value == 57 and p.notes is not None


def test_nested_object_only_as_fallback():
    assert extract_object('{"price": 5, "meta": {"price": 7}}', "price")["price"] == 5
    assert extract_object('{"answer": {"price": 7}}', "price")["price"] == 7


@pytest.mark.parametrize("raw,value,warn", [("57", 57, True), (57.0, 57, True), (0, 0, False), (100, 100, False)])
def test_price_coercion(raw, value, warn):
    import json
    p = parse_price(json.dumps({"notes": "x", "price": raw}))
    assert p.ok and p.value == value and bool(p.warnings) == warn


@pytest.mark.parametrize("content,needle", [
    ('{"price": 150}', "outside"), ('{"price": -1}', "outside"), ('{"price": 57.5}', "whole number"),
    ('{"price": true}', "whole number"), ('{"price": "fifty"}', "whole number"), ('{"price": null}', "whole number"),
    ("I pick 57", "no JSON"), ("", "empty"), (None, "empty"), ("<think>only thinking</think>", "empty"),
    ('{"message": "hi"}', "no JSON"), ('{"notes": "x", "price": ', "no JSON"),
])
def test_price_errors(content, needle):
    p = parse_price(content)
    assert not p.ok and needle in p.error and p.value is None


def test_missing_notes_is_a_warning_only():
    p = parse_price('{"price": 40}')
    assert p.ok and p.notes is None and "notes missing" in p.warnings


@pytest.mark.parametrize("raw", ['null', '""', '"pass"', '"  PASS "'])
def test_message_pass(raw):
    p = parse_message(f'{{"notes": "n", "message": {raw}}}')
    assert p.ok and p.value is None


def test_message_truncation_and_errors():
    p = parse_message('{"message": "' + "x" * 350 + '"}')
    assert p.ok and len(p.value) == 300 and any("truncated" in w for w in p.warnings)
    assert parse_message('{"message": " hello "}').value == "hello"
    assert not parse_message('{"message": ["a"]}').ok
    assert not parse_message('{"msg": "a"}').ok
    p = parse_message('{"message": 42}')
    assert p.ok and p.value == "42" and p.warnings


def test_strip_reasoning():
    assert strip_reasoning("<think>a</think> b <thinking>c</thinking>") == "b"
    assert strip_reasoning("abc </think> d") == "d"
