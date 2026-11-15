from formbench.scoring import BY_NAME, SCORES, comments, metrics, number, sum_counts


def photo(cells, correct, tp, fp, fn, valid=1.0, **extra):
    return {
        "pages": 1,
        "json_valid": valid,
        "field_cells": cells,
        "field_correct": correct,
        "field_tp": tp,
        "field_fp": fp,
        "field_fn": fn,
        **extra,
    }


def test_scores_in_order_with_labels():
    assert [s.name for s in SCORES] == [
        "overall_accuracy",
        "completeness",
        "reliability",
        "rows_detected",
        "processing_success_rate",
        "processing_time",
    ]
    assert BY_NAME["overall_accuracy"].heading == "Overall accuracy ↑"
    assert BY_NAME["processing_time"].heading == "Processing time (seconds per photo) ↓"
    assert BY_NAME["overall_accuracy"].description.startswith("THE MAIN SCORE.")
    assert "invents values" in BY_NAME["reliability"].description
    assert "misses or misreads" in BY_NAME["completeness"].description
    assert BY_NAME["processing_time"].max_value is None and BY_NAME["reliability"].max_value == 1


def test_counts_are_summed_before_rates_are_taken():
    small = photo(2, 0, 0, 2, 2)  # 0% on 2 cells
    large = photo(198, 198, 198, 0, 0)  # 100% on 198 cells
    total = sum_counts([small, large])
    assert metrics(total)["overall_accuracy"] == 0.99
    averaged = (metrics(small)["overall_accuracy"] + metrics(large)["overall_accuracy"]) / 2
    assert averaged == 0.5  # what averaging per photo would wrongly give


def test_metrics_and_rounding():
    counts = sum_counts([photo(3, 2, 2, 1, 1, rows=3, rows_found=2), photo(0, 0, 0, 0, 0, 0.0)])
    assert metrics(counts, latency=10.0) == {
        "overall_accuracy": 0.6667,
        "completeness": 0.6667,
        "reliability": 0.6667,
        "rows_detected": 0.6667,
        "processing_success_rate": 0.5,
        "processing_time": 5.0,
    }
    assert metrics(counts, latency=10.06)["processing_time"] == 5.0


def test_scores_with_a_zero_denominator_are_left_out():
    counts = photo(0, 0, 0, 0, 0)
    assert metrics(counts) == {"processing_success_rate": 1.0}
    assert list(comments(counts)) == ["processing_success_rate"]
    assert metrics({}) == {} and comments({}, latency=3.0) == {}


def test_comments_text_exactly():
    counts = photo(485, 432, 432, 23, 53)
    notes = comments(sum_counts([counts]), latency=359.6)
    assert notes["overall_accuracy"] == (
        "How many cells were read correctly: right cells ÷ cells with a value on the form or "
        "in the reading = 432 ÷ 485"
    )
    assert notes["completeness"] == (
        "How much of what is written was captured: entries read right ÷ entries written on "
        "the form = 432 ÷ 485"
    )
    assert notes["reliability"] == (
        "How much of what the model reported is true: values right ÷ values the model "
        "reported = 432 ÷ 455"
    )
    assert notes["processing_success_rate"] == (
        "How many photos gave a usable answer: usable answers ÷ photos = 1 ÷ 1"
    )
    assert notes["processing_time"] == (
        "How long one photo takes to read: seconds ÷ photos = 359.6 ÷ 1"
    )
    assert "rows_detected" not in notes


def test_number():
    assert (number(37.0), number(4), number(359.64)) == ("37", "4", "359.6")
