import itertools
import random

import pytest

from formbench.textmetrics import (
    anls,
    anls_list,
    char_edits,
    f1,
    hungarian,
    levenshtein,
    match_lists,
    normalized_distance,
    tree_accuracy,
    word_edits,
)


@pytest.mark.parametrize(
    "a, b, d", [("", "", 0), ("abc", "", 3), ("kitten", "sitting", 3), ("12-06-25", "12-08-25", 1)]
)
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d == levenshtein(b, a)


def test_cer_wer_counts():
    assert char_edits("Amani North", "Amani Nort") == (1, 11)
    assert word_edits("Amani North Clinic", "Amani Clinic") == (1, 3)


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


def test_anls_list_penalizes_missing_and_invented_items():
    same = lambda a, b: anls(a, b)  # noqa: E731
    assert anls_list(["a", "b"], ["b", "a"], same) == 1.0
    assert anls_list(["a", "b"], ["a"], same) == 0.5
    assert anls_list(["a"], ["a", "zz"], same) == 0.5
    assert anls_list([], [], same) == 1.0
    assert [(i, j) for i, j, _ in match_lists(["x", "y"], ["y", "x"], same)] == [(0, 1), (1, 0)]


def test_f1():
    assert f1(8, 2, 2) == pytest.approx(0.8)
    assert f1(0, 0, 0) is None


def test_tree_accuracy():
    truth = {"cohort": "June", "rows": [{"serial": "1", "opv1": "12-06-25"}]}
    assert tree_accuracy(truth, truth) == 1.0
    close = {"cohort": "June", "rows": [{"serial": "1", "opv1": "12-08-25"}]}
    assert 0.9 < tree_accuracy(truth, close) < 1.0
    moved = {"cohort": "June", "rows": [{"serial": "1", "opv2": "12-06-25"}]}
    assert tree_accuracy(truth, moved) < tree_accuracy(truth, close)
    assert tree_accuracy(truth, {}) == 0.0
