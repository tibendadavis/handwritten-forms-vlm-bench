import itertools
import random

import pytest

from formbench.metrics import anls, hungarian, levenshtein, normalized_distance


@pytest.mark.parametrize(
    "a, b, d", [("", "", 0), ("abc", "", 3), ("kitten", "sitting", 3), ("12-06-25", "12-08-25", 1)]
)
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d == levenshtein(b, a)


def test_anls_threshold():
    assert anls("abcd", "abcd") == 1.0
    assert anls("abcd", "abce") == pytest.approx(0.75)
    assert anls("abcd", "wxyz") == 0.0
    assert anls("", "") == 1.0
    assert normalized_distance("ab", "") == 1.0


def test_hungarian_is_optimal_on_random_matrices():
    rng = random.Random(0)
    for n, m in [(3, 3), (4, 2), (2, 5), (6, 6)]:
        cost = [[rng.random() for _ in range(m)] for _ in range(n)]
        pairs = hungarian(cost)
        assert len(pairs) == min(n, m)
        best = min(
            sum(cost[i][j] for i, j in zip(rows, cols))
            for rows in itertools.permutations(range(n), min(n, m))
            for cols in itertools.permutations(range(m), min(n, m))
        )
        assert sum(cost[i][j] for i, j in pairs) == pytest.approx(best)
