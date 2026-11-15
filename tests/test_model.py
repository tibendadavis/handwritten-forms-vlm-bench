import base64
import io
from types import SimpleNamespace

import httpx
import openai
import pytest
from PIL import Image

from formbench import model
from formbench.model import encode_image, interpret, parse_json, read_json, response_schema

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "region": {"type": ["string", "null"]},
        "screened": {"type": ["string", "null"]},
    },
    "required": ["region", "screened"],
    "additionalProperties": False,
}


def decode(url):
    header, data = url.split(",", 1)
    return header, Image.open(io.BytesIO(base64.b64decode(data)))


def as_is(path, mime):
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("PNG", "image/png")])
def test_photo_within_max_side_is_sent_byte_for_byte(tmp_path, fmt, mime):
    path = tmp_path / f"photo.{fmt.lower()}"
    Image.new("RGB", (900, 1200), "white").save(path, fmt)
    assert encode_image(path) == as_is(path, mime)


def test_png_named_jpg_goes_as_png(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (900, 1200), "white").save(path, "PNG")
    assert encode_image(path) == as_is(path, "image/png")


def test_sideways_jpeg_is_turned_upright_and_re_encoded(tmp_path):
    path = tmp_path / "sideways.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # camera held sideways
    Image.new("RGB", (1200, 900), "white").save(path, "JPEG", exif=exif)
    url = encode_image(path)
    assert url != as_is(path, "image/jpeg")
    header, image = decode(url)
    assert header == "data:image/jpeg;base64" and image.size == (900, 1200)


def test_large_photo_is_shrunk_to_4032_by_default(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (5000, 3000), "white").save(path)
    header, image = decode(encode_image(path))
    assert header == "data:image/jpeg;base64" and max(image.size) == 4032


def test_large_photo_is_shrunk_to_max_side(tmp_path):
    path = tmp_path / "big.jpg"
    Image.new("RGB", (5000, 3000), "white").save(path)
    _, image = decode(encode_image(path, max_side=2000))
    assert image.size == (2000, 1200)


def test_photo_of_4000_by_3000_is_sent_as_is(tmp_path):
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (4000, 3000), "white").save(path)
    assert encode_image(path) == as_is(path, "image/jpeg")


def test_other_formats_are_re_encoded_as_jpeg(tmp_path):
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (300, 200), "white").save(path, "BMP")
    header, image = decode(encode_image(path))
    assert header == "data:image/jpeg;base64" and image.size == (300, 200)


@pytest.mark.parametrize(
    "text",
    [
        '{"region": "North", "screened": "4"}',
        '```json\n{"region": "North", "screened": "4"}\n```',
        'Here it is:\n```\n{"region": "North", "screened": "4"}\n```\nDone.',
    ],
)
def test_interpret_tolerates_fences_and_prose(text):
    result = interpret(text, SCHEMA)
    assert result["json_valid"] and result["errors"] == []
    assert result["values"] == {"region": "North", "screened": "4"}
    assert result["raw"] == text


@pytest.mark.parametrize("text", [None, "", "I cannot read this.", '{"region": ', "[1, 2]"])
def test_interpret_without_json(text):
    assert parse_json(text) is None
    result = interpret(text, SCHEMA)
    assert result == {
        "raw": text,
        "values": {"region": None, "screened": None},
        "json_valid": False,
        "errors": ["no JSON object in the reply"],
    }


def test_schema_violation_is_recorded_not_raised():
    result = interpret('{"region": "North", "screened": 4, "name": "x"}', SCHEMA)
    assert not result["json_valid"] and len(result["errors"]) == 2
    assert result["errors"] == sorted(result["errors"])
    assert result["values"] == {"region": "North", "screened": 4}


def test_response_schema_drops_only_the_header():
    out = response_schema(SCHEMA)
    assert "$schema" not in out and out["properties"] == SCHEMA["properties"]


def test_timeout_becomes_an_invalid_result(monkeypatch, tmp_path):
    def slow(*args, **kwargs):
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://localhost"))

    monkeypatch.setattr(model, "complete_json", slow)
    assert read_json(tmp_path / "x.jpg", SCHEMA, "prompt", "m") == {
        "raw": None,
        "values": {"region": None, "screened": None},
        "json_valid": False,
        "errors": ["model call timed out (no reply)"],
    }


def test_other_errors_are_raised(monkeypatch, tmp_path):
    def down(*args, **kwargs):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://localhost"))

    monkeypatch.setattr(model, "complete_json", down)
    with pytest.raises(openai.APIConnectionError):
        read_json(tmp_path / "x.jpg", SCHEMA, "prompt", "m")


class FakeClient:
    """Stands in for the traced OpenAI client and records the request."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"region": "North", "screened": null}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize("think", [False, True])
def test_complete_json_request(monkeypatch, tmp_path, think):
    clients = []

    def make(**kwargs):
        clients.append(FakeClient(**kwargs))
        return clients[-1]

    monkeypatch.setattr(model, "OpenAI", make)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    monkeypatch.setenv("VLM_TIMEOUT", "30")
    photo = tmp_path / "p.png"
    Image.new("RGB", (10, 10), "white").save(photo)
    prompt = object()
    reply = model.complete_json(
        photo,
        SCHEMA,
        "Read it.",
        "small:1b",
        think=think,
        max_side=4032,
        trace_name="tally-x",
        metadata={"prompt_version": 3},
        langfuse_prompt=prompt,
    )
    assert reply == '{"region": "North", "screened": null}'
    client = clients[0]
    assert client.kwargs == {
        "base_url": "http://ollama.test/v1",
        "api_key": "ollama",
        "timeout": 30.0,
        "max_retries": 0,
    }
    call = client.calls[0]
    assert (call["model"], call["temperature"], call["seed"], call["max_tokens"]) == (
        "small:1b",
        0,
        42,
        8192,
    )
    text, image = call["messages"][0]["content"]
    assert text == {"type": "text", "text": "Read it."}
    assert image["image_url"]["url"] == as_is(photo, "image/png")
    assert call["response_format"]["json_schema"] == {
        "name": "form_extraction",
        "strict": True,
        "schema": response_schema(SCHEMA),
    }
    assert call["name"] == "tally-x" and call["langfuse_prompt"] is prompt
    assert call["metadata"] == {"think": think, "prompt_version": 3}
    assert ("reasoning_effort" in call) is not think
    if not think:
        assert call["reasoning_effort"] == "none"
