import pytest

from formbench.matching import anls, hungarian, levenshtein, match_lists, normalized_distance


def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "") == 3


def test_normalized_distance_of_two_empty_strings_is_zero():
    assert normalized_distance("", "") == 0.0
    assert anls("", "") == 1.0


def test_anls_threshold():
    assert anls("abcd", "abcd") == 1.0
    assert anls("abcd", "abcx") == pytest.approx(0.75)
    assert anls("abcd", "abxy") == 0.0  # distance 0.5 is at the threshold: no credit
    assert anls("abcdefghij", "abcdefxyzw") == pytest.approx(0.6)


def test_hungarian_square():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pairs = hungarian(cost)
    assert pairs == [(0, 1), (1, 0), (2, 2)]
    assert sum(cost[i][j] for i, j in pairs) == 5


def test_hungarian_more_rows_than_columns():
    cost = [[9, 1], [1, 9], [5, 5]]
    assert hungarian(cost) == [(0, 1), (1, 0)]


def test_hungarian_more_columns_than_rows():
    cost = [[9, 9, 1, 9], [1, 9, 9, 9]]
    assert hungarian(cost) == [(0, 2), (1, 0)]


def test_hungarian_empty():
    assert hungarian([]) == [] and hungarian([[]]) == []


def test_match_lists_pairs_by_similarity():
    truth = ["apple", "banana", "cherry"]
    predicted = ["cherry", "aple"]
    pairs = match_lists(truth, predicted, lambda a, b: anls(a, b))
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (2, 0)]
    assert match_lists([], predicted, anls) == [] and match_lists(truth, [], anls) == []
