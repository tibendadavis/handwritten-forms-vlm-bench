"""The one model call: a photo and a prompt in, a schema-shaped JSON reply out.

Every call goes to a local Ollama server through Langfuse's drop-in OpenAI wrapper, so each
generation is traced with its image and linked to its prompt version.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import openai
from jsonschema import Draft202012Validator
from langfuse.openai import OpenAI
from PIL import Image, ImageOps

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Temperature 0 and a fixed seed: the same photo and prompt give the same reply, so runs repeat.
SEED = 42
# A full register page is about 5k output tokens; the cap stops a reply that loops forever.
MAX_TOKENS = 8192
# One call may take minutes on a laptop; past this it has failed.
DEFAULT_TIMEOUT = 600.0
# A 12 MP phone photo as taken. Larger photos are shrunk; smaller ones are never enlarged.
DEFAULT_MAX_SIDE = 4032
# Formats sent byte for byte, by what is inside the file (a .jpg may hold a PNG).
AS_TAKEN = {"JPEG": "image/jpeg", "PNG": "image/png"}
ORIENTATION = 0x0112  # EXIF tag: how the camera was held


def encode_image(path: Path, max_side: int = DEFAULT_MAX_SIDE) -> str:
    """The photo as a data URL.

    A JPEG or PNG that is upright and at most `max_side` pixels on its long side is sent byte
    for byte, as taken. Otherwise it is turned upright, shrunk to `max_side` and sent as JPEG.
    Re-encoding a photo, even invisibly, changes what a model reads, so it only happens when
    the photo cannot be sent as it is.
    """
    with Image.open(path) as opened:
        mime = AS_TAKEN.get(opened.format or "")
        upright = opened.getexif().get(ORIENTATION, 1) == 1
        if mime and upright and max(opened.size) <= max_side:
            return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema as sent to the server: "$schema" is a document header, not structure."""
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
    """A model reply as values plus validity. Invalid output is recorded, never raised: a bad
    reply is a result to score, not a crash."""
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


def make_client() -> Any:
    """The traced OpenAI client pointed at Ollama. No retries: a retry of a deterministic call
    repeats the same answer and only multiplies the wait."""
    return OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL,
        api_key="ollama",
        timeout=float(os.environ.get("VLM_TIMEOUT") or DEFAULT_TIMEOUT),
        max_retries=0,
    )


def complete_json(
    image: Path,
    schema: dict[str, Any],
    prompt_text: str,
    model: str,
    *,
    think: bool,
    max_side: int,
    trace_name: str,
    metadata: dict[str, Any],
    langfuse_prompt: Any,
) -> str | None:
    """One schema-constrained call to an Ollama model with one photo; returns the reply text.

    `langfuse_prompt` links the traced generation to the prompt version it was compiled from.
    A reply cut off at MAX_TOKENS is returned as is; it then fails parsing and is scored as
    not usable.
    """
    # Ollama maps reasoning_effort "none" to "no thinking"; leaving it out keeps the default.
    thinking: dict[str, Any] = {} if think else {"reasoning_effort": "none"}
    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": encode_image(image, max_side)}},
    ]
    response = make_client().chat.completions.create(
        model=model,
        temperature=0,
        seed=SEED,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "form_extraction",
                "strict": True,
                "schema": response_schema(schema),
            },
        },
        name=trace_name,
        metadata={"think": think, **metadata},
        langfuse_prompt=langfuse_prompt,
        **thinking,
    )
    return response.choices[0].message.content


def read_json(
    image: Path, schema: dict[str, Any], prompt_text: str, model: str, **options: Any
) -> dict[str, Any]:
    """complete_json, then interpret: {"raw", "values", "json_valid", "errors"}.

    A timeout is recorded as an invalid reply, so one stuck photo cannot stop a whole run.
    Any other error (for example Ollama being down) is raised: that must be loud.
    `options` are complete_json's keyword arguments.
    """
    try:
        raw = complete_json(image, schema, prompt_text, model, **options)
    except openai.APITimeoutError:
        return {
            "raw": None,
            "values": dict.fromkeys(schema["properties"]),
            "json_valid": False,
            "errors": ["model call timed out (no reply)"],
        }
    return interpret(raw, schema)
