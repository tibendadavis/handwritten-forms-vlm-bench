"""The scores, and how they are built from counts.

Every score is listed once in SCORES with a plain label, a description, what it means and its
formula. Terminal tables, Langfuse score configs and the comment on every Langfuse score all
come from that list, so they always agree.

Scores are built from counts: each photo adds its counts, and each rate is taken once over the
sum. A folder (or several runs pooled) is never an average of per-photo scores, which would
let a photo with two cells weigh as much as one with two hundred.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    name: str  # the Langfuse score name
    label: str
    description: str  # for the Langfuse score config and `formbench scores`
    meaning: str  # one plain sentence, the start of every comment
    formula: str  # the division in words; the comment adds the numbers
    higher_is_better: bool = True
    max_value: float | None = 1.0  # None: no upper bound (seconds)
    places: int = 4

    @property
    def is_rate(self) -> bool:
        return self.max_value is not None

    @property
    def heading(self) -> str:
        """The label with the way that is better: ↑ or ↓."""
        return f"{self.label} {'↑' if self.higher_is_better else '↓'}"


SCORES = (
    Score(
        "overall_accuracy",
        "Overall accuracy",
        "THE MAIN SCORE. Right cells ÷ cells with a value on the form or in the reading: of "
        "every cell that has a value on the form or in the model's output, the share with the "
        "right value under the right column.",
        "How many cells were read correctly",
        "right cells ÷ cells with a value on the form or in the reading",
    ),
    Score(
        "completeness",
        "Completeness",
        "Entries read right ÷ entries written on the form. Low: the model misses or misreads "
        "what is there.",
        "How much of what is written was captured",
        "entries read right ÷ entries written on the form",
    ),
    Score(
        "reliability",
        "Reliability",
        "Values right ÷ values the model reported. Low: the model invents values or puts them "
        "under the wrong column.",
        "How much of what the model reported is true",
        "values right ÷ values the model reported",
    ),
    Score(
        "rows_detected",
        "Rows detected",
        "Registers only. Rows found by serial number ÷ rows on the page. Low: the model skips "
        "rows or misreads serial numbers.",
        "How many rows were found",
        "rows found by serial number ÷ rows on the page",
    ),
    Score(
        "processing_success_rate",
        "Processing success rate",
        "Usable answers ÷ photos: photos for which the model returned JSON in the required "
        "shape. Low: some photos produced nothing usable (bad JSON or a timeout).",
        "How many photos gave a usable answer",
        "usable answers ÷ photos",
    ),
    Score(
        "processing_time",
        "Processing time (seconds per photo)",
        "Seconds ÷ photos: how long one photo takes to read on this machine. High: the model is "
        "slow here.",
        "How long one photo takes to read",
        "seconds ÷ photos",
        higher_is_better=False,
        max_value=None,
        places=1,
    ),
)
BY_NAME = {s.name: s for s in SCORES}


def sum_counts(counts: Iterable[dict[str, float]]) -> dict[str, float]:
    """Counts added key by key, over photos or over runs."""
    out: dict[str, float] = {}
    for c in counts:
        for key, value in c.items():
            out[key] = out.get(key, 0) + value
    return out


def divisions(counts: dict[str, float], latency: float | None) -> dict[str, tuple[float, float]]:
    """Each score as the division it is: (numerator, denominator)."""
    c = counts.get
    tp = c("field_tp", 0)
    return {
        "overall_accuracy": (c("field_correct", 0), c("field_cells", 0)),
        "completeness": (tp, tp + c("field_fn", 0)),
        "reliability": (tp, tp + c("field_fp", 0)),
        "rows_detected": (c("rows_found", 0), c("rows", 0)),
        "processing_success_rate": (c("json_valid", 0), c("pages", 0)),
        "processing_time": (latency, c("pages", 0)) if latency is not None else (0, 0),
    }


def metrics(counts: dict[str, float], latency: float | None = None) -> dict[str, float]:
    """Scores from summed counts, in SCORES order. A score whose denominator is 0 cannot be
    worked out from these counts and is left out."""
    parts = divisions(counts, latency)
    out = {}
    for score in SCORES:
        numerator, denominator = parts[score.name]
        if denominator:
            out[score.name] = round(numerator / denominator, score.places)
    return out


def number(x: float) -> str:
    """A count as text: whole numbers without decimals, others with one."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def comments(counts: dict[str, float], latency: float | None = None) -> dict[str, str]:
    """For each score in metrics(): what it means and how it was worked out, e.g.
    "How many cells were read correctly: right cells ÷ ... = 432 ÷ 485"."""
    parts = divisions(counts, latency)
    out = {}
    for score in SCORES:
        numerator, denominator = parts[score.name]
        if denominator:
            out[score.name] = (
                f"{score.meaning}: {score.formula} = {number(numerator)} ÷ {number(denominator)}"
            )
    return out
