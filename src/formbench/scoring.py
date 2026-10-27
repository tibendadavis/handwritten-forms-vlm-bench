from __future__ import annotations

from typing import Any


def is_numeric(prop: dict[str, Any]) -> bool:
    types = prop.get("type", [])
    types = [types] if isinstance(types, str) else types
    return "integer" in types or "number" in types


def normalize(value: Any, numeric: bool) -> Any:
    """Comparable form of a value: trimmed, "" -> None, numeric strings -> numbers.

    Text is casefolded with whitespace collapsed (exact match after normalization).
    """
    if isinstance(value, str):
        value = " ".join(value.split())
        if not value:
            return None
        if not numeric:
            return value.casefold()
        try:
            value = float(value)
        except ValueError:
            return value
    if numeric and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def score(
    prediction: dict[str, Any], truth: dict[str, Any], schema: dict[str, Any], json_valid: bool
) -> dict[str, Any]:
    """Per-field results and summary metrics for one extraction.

    Rates whose denominator is empty (e.g. hallucination_rate with no blank cells) are None.
    """
    fields = []
    for key, prop in schema["properties"].items():
        numeric = is_numeric(prop)
        t, p = normalize(truth.get(key), numeric), normalize(prediction.get(key), numeric)
        fields.append(
            {
                "key": key,
                "numeric": numeric,
                "truth": truth.get(key),
                "predicted": prediction.get(key),
                "correct": t == p,
                "hallucinated": t is None and p is not None,
                "omitted": t is not None and p is None,
            }
        )

    filled = [f for f in fields if normalize(f["truth"], f["numeric"]) is not None]
    blank = [f for f in fields if normalize(f["truth"], f["numeric"]) is None]
    numeric_filled = [f for f in filled if f["numeric"]]
    metrics = {
        "json_valid": int(json_valid),
        "field_accuracy": _rate(sum(f["correct"] for f in fields), len(fields)),
        "numeric_accuracy": _rate(sum(f["correct"] for f in numeric_filled), len(numeric_filled)),
        "hallucination_rate": _rate(sum(f["hallucinated"] for f in blank), len(blank)),
        "omission_rate": _rate(sum(f["omitted"] for f in filled), len(filled)),
    }
    return {"metrics": metrics, "fields": fields}
