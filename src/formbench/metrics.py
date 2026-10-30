"""Text and structure metrics for extraction outputs: CER, WER, ANLS, ANLS for lists,
field F1 and an ANLS-weighted tree edit distance. Pure functions, no dependencies beyond zss.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import zss

ANLS_THRESHOLD = 0.5


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Edit distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def normalized_distance(a: str, b: str) -> float:
    """Levenshtein distance divided by the longer length; 0 when both are empty."""
    longest = max(len(a), len(b))
    return levenshtein(a, b) / longest if longest else 0.0


def anls(truth: str, predicted: str, threshold: float = ANLS_THRESHOLD) -> float:
    """ANLS similarity of one field: 1 - normalized distance, or 0 at/above the threshold."""
    distance = normalized_distance(truth, predicted)
    return 1 - distance if distance < threshold else 0.0


def char_edits(truth: str, predicted: str) -> tuple[int, int]:
    """(edits, truth characters) for corpus-level CER."""
    return levenshtein(truth, predicted), len(truth)


def word_edits(truth: str, predicted: str) -> tuple[int, int]:
    """(edits, truth words) for corpus-level WER."""
    words = truth.split()
    return levenshtein(words, predicted.split()), len(words)


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one assignment for a rectangular cost matrix (rows x columns).

    Classic O(n^3) Kuhn-Munkres with potentials; returns (row, column) pairs.
    """
    if not cost or not cost[0]:
        return []
    transposed = len(cost) > len(cost[0])
    matrix = [list(r) for r in zip(*cost)] if transposed else [list(r) for r in cost]
    n, m = len(matrix), len(matrix[0])
    inf = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    owner, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0], j0 = i, 0
        minv, used = [inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j], way[j] = current, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    pairs = [(owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j]]
    return sorted((c, r) for r, c in pairs) if transposed else sorted(pairs)


def match_lists(
    truth: list[Any], predicted: list[Any], similarity: Callable[[Any, Any], float]
) -> list[tuple[int, int, float]]:
    """Pair truth and predicted items to maximise total similarity: (i, j, similarity)."""
    if not truth or not predicted:
        return []
    scores = [[similarity(t, p) for p in predicted] for t in truth]
    pairs = hungarian([[1 - s for s in row] for row in scores])
    return [(i, j, scores[i][j]) for i, j in pairs]


def anls_list(truth: list[Any], predicted: list[Any], similarity: Callable[[Any, Any], float]) -> float:  # fmt: skip
    """ANLS for lists (as in ANLS*): optimal matching, divided by the longer list, so missing
    and invented items both cost. Two empty lists score 1."""
    if not truth and not predicted:
        return 1.0
    matched = match_lists(truth, predicted, similarity)
    return sum(s for _, _, s in matched) / max(len(truth), len(predicted))


def f1(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


class _Node:
    __slots__ = ("label", "children", "leaf")

    def __init__(self, label: str, children: list[_Node] | None = None, leaf: bool = False):
        self.label, self.children, self.leaf = label, children or [], leaf


def to_tree(value: Any, label: str = "<root>") -> _Node:
    """JSON-like value -> ordered labelled tree; dict keys sorted, None/"" leaves dropped."""
    if isinstance(value, dict):
        return _Node(label, [to_tree(value[k], k) for k in sorted(value) if value[k] not in (None, "")])  # fmt: skip
    if isinstance(value, list):
        return _Node(label, [to_tree(v, "<item>") for v in value])
    return _Node(label, [_Node(str(value), leaf=True)])


def _size(node: _Node) -> int:
    return 1 + sum(_size(c) for c in node.children)


def tree_distance(truth: _Node, predicted: _Node) -> float:
    """Tree edit distance; inserting or removing a node costs 1, relabelling a leaf costs its
    normalized edit distance (ANLS-style), relabelling a structural node costs 0 or 1."""

    def update(a: _Node, b: _Node) -> float:
        if a.leaf and b.leaf:
            return normalized_distance(a.label, b.label)
        return 0.0 if a.label == b.label and a.leaf == b.leaf else 1.0

    return zss.distance(truth, predicted, lambda n: n.children, lambda n: 1, lambda n: 1, update)


def tree_accuracy(truth: Any, predicted: Any) -> float:
    """1 - TED(truth, predicted) / size(truth), floored at 0 (nTED, as used for Donut)."""
    t, p = to_tree(truth), to_tree(predicted)
    total = _size(t) - 1
    return (
        1.0 if total == 0 and _size(p) == 1 else max(0.0, 1 - tree_distance(t, p) / max(total, 1))
    )
