from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image, ImageOps

# Bump when the prompt text changes so runs stay comparable.
PROMPT_VERSION = "v1"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
SEED = 42
# Phone photos are 12MP+; a fixed downscale keeps latency sane and runs repeatable.
MAX_SIDE = 2048


def build_prompt(schema: dict[str, Any]) -> str:
    keys = "\n".join(
        f"- {key}: {prop.get('description', key)}" for key, prop in schema["properties"].items()
    )
    return (
        "You are reading a photo of a handwritten paper form.\n"
        "Return one JSON object with exactly these keys:\n"
        f"{keys}\n\n"
        "Rules:\n"
        "- Counts are integers, exactly as written in the box.\n"
        "- Use null when a box is empty or unreadable. Never guess a value.\n"
        "- Copy the facility name and period exactly as handwritten.\n"
        "- Output only the JSON object."
    )


def encode_image(source: Path | Image.Image, max_side: int = MAX_SIDE) -> str:
    """The photo as a JPEG data URL, upright and at most max_side pixels on its long side."""
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        with Image.open(source) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    # "$schema" is a document header, not part of the structure the server enforces.
    return {k: v for k, v in schema.items() if k != "$schema"}


def parse_json(text: str | None) -> dict[str, Any] | None:
    """The first JSON object in a model reply, tolerating code fences and prose around it."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def interpret(raw: str | None, schema: dict[str, Any]) -> dict[str, Any]:
    """Turn a model reply into values; invalid output is recorded, never raised."""
    keys = list(schema["properties"])
    obj = parse_json(raw)
    if obj is None:
        return {
            "raw": raw,
            "values": dict.fromkeys(keys),
            "json_valid": False,
            "errors": ["no JSON object in the reply"],
        }
    errors = sorted(e.message for e in Draft202012Validator(schema).iter_errors(obj))
    return {
        "raw": raw,
        "values": {k: obj.get(k) for k in keys},
        "json_valid": not errors,
        "errors": errors,
    }


def complete_json(
    image: Path | Image.Image,
    schema: dict[str, Any],
    prompt: str,
    model: str,
    *,
    think: bool = False,
    max_side: int = MAX_SIDE,
    name: str = "vlm-extract",
    metadata: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> str | None:
    """One schema-constrained call to an Ollama model with an image; traced by Langfuse."""
    from langfuse.openai import OpenAI  # type: ignore # drop-in wrapper: traces every call

    client = OpenAI(
        base_url=base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL), api_key="ollama"
    )
    # Ollama maps reasoning_effort "none" to think=false; omitting it keeps the model default.
    options: dict[str, Any] = {} if think else {"reasoning_effort": "none"}
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=SEED,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image(image, max_side)}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "form_extraction",
                "strict": True,
                "schema": response_schema(schema),
            },
        },
        name=name,
        metadata={"think": think, **(metadata or {})},
        **options,
    )
    return response.choices[0].message.content


def extract(
    image: Path,
    schema: dict[str, Any],
    model: str,
    *,
    think: bool = False,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Ask an Ollama model to fill the form schema from the photo."""
    raw = complete_json(
        image,
        schema,
        build_prompt(schema),
        model,
        think=think,
        metadata={"prompt_version": PROMPT_VERSION},
        base_url=base_url,
    )
    return interpret(raw, schema)
