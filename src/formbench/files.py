"""Small helpers for file names and JSON files written by every command."""

from __future__ import annotations

import datetime as dt
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    """Lowercase ASCII with underscores: 'RI v2' -> 'ri_v2', 'ollama:qwen3.5:4b' -> 'ollama_qwen3_5_4b'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def stamp() -> str:
    """UTC time for file names, e.g. 20261030T101500Z."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    return path
