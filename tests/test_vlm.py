import base64
import io

import pytest
from PIL import Image

from formbench.vlm import (
    MAX_SIDE,
    encode_image,
    interpret,
    parse_json,
    response_schema,
)

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "period": {"type": ["string", "null"], "description": "Period"},
        "bcg__lt_1y": {"type": ["integer", "null"], "description": "BCG doses given (<1y)"},
    },
    "required": ["period", "bcg__lt_1y"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "text",
    [
        '{"period": "202501", "bcg__lt_1y": 3}',
        '```json\n{"period": "202501", "bcg__lt_1y": 3}\n```',
        'Here is the data:\n{"period": "202501", "bcg__lt_1y": 3}\nHope that helps.',
    ],
)
def test_parse_json_tolerates_fences_and_prose(text):
    assert parse_json(text) == {"period": "202501", "bcg__lt_1y": 3}


@pytest.mark.parametrize("text", [None, "", "no json here", '{"period": ', "[1, 2]"])
def test_parse_json_returns_none_for_unusable_replies(text):
    assert parse_json(text) is None


def test_valid_reply():
    result = interpret('{"period": "202501", "bcg__lt_1y": 3}', SCHEMA)
    assert result["json_valid"] and result["values"] == {"period": "202501", "bcg__lt_1y": 3}


def test_schema_violation_is_recorded_not_raised():
    result = interpret('{"period": "202501", "bcg__lt_1y": "three", "extra": 1}', SCHEMA)
    assert not result["json_valid"]
    assert len(result["errors"]) == 2
    assert result["values"] == {"period": "202501", "bcg__lt_1y": "three"}


def test_unparseable_reply_gives_all_null():
    result = interpret("I cannot read this form.", SCHEMA)
    assert not result["json_valid"]
    assert result["values"] == {"period": None, "bcg__lt_1y": None}


def test_response_schema_drops_only_the_header():
    out = response_schema(SCHEMA)
    assert "$schema" not in out and out["properties"] == SCHEMA["properties"]


def test_photo_is_downscaled(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4000, 3000), "white").save(path)
    url = encode_image(path)
    assert url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(decoded.size) == MAX_SIDE


def test_in_memory_image_is_encoded_with_custom_size():
    url = encode_image(Image.new("RGB", (5000, 1000), "white"), max_side=3000)
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.size == (3000, 600)
