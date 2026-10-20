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
    ('{"message": "hi"}', "no JSON"), ('{"notes": "x", "price": ', "could not parse a JSON object"),
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


def test_malformed_json_is_reported_as_unparseable():
    # a curly quote that is not in a structural position is never straightened
    p = parse_message('{"notes": "ok", "message": "Let us all set 88.” please}')
    assert not p.ok and p.error.startswith("could not parse a JSON object")
    assert "could not parse" in parse_price('{"notes": "x", "price": 9O}').error
    assert "could not parse" in parse_price("{notes: x}").error
    # a well-formed object without the field is still reported as missing the field
    assert parse_message('{"notes": "x", "price": 88}').error == 'no JSON object with a "message" field was found'
    assert parse_price("I pick 57").error == 'no JSON object with a "price" field was found'


def test_structural_curly_quotes_coerced():
    # pilot 1, gpt-oss firm B: a string closed with a curly quote right before }
    p = parse_message('{"notes": "ok", "message": "Let us all set 88.”}')
    assert p.ok and p.value == "Let us all set 88." and p.notes == "ok"
    assert any("coerced" in w and "typographic" in w for w in p.warnings)
    p = parse_price('{"notes": "keep high” , "price": 99}')  # before a comma, with whitespace
    assert p.ok and p.value == 99 and p.notes == "keep high" and any("typographic" in w for w in p.warnings)
    p = parse_price('{"notes":  “keep high", "price": 99}')  # opening quote after a colon
    assert p.ok and p.notes == "keep high" and any("typographic" in w for w in p.warnings)


def test_curly_quotes_in_content_untouched():
    # a strictly valid answer keeps every curly quote, even next to commas, and has no coercion warning
    p = parse_price('{"notes": "they said “stay at 100”, so I will”", "price": 100}')
    assert p.ok and p.notes == "they said “stay at 100”, so I will”" and not p.warnings
    p = parse_message('{"notes": "n", "message": "“Deal”, everyone: “100”."}')
    assert p.ok and p.value == "“Deal”, everyone: “100”." and not p.warnings
    # the retry only straightens structural quotes: content quotes would break the object, so it is re-asked
    assert not parse_price('{"notes": "a “b”, c”, "price": 9}').ok
