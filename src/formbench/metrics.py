"""Text similarity and list matching, used to pair the model's rows with the truth's rows.
Pure functions, no dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

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
