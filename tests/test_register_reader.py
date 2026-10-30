from formbench.register.reader import clean_rows


def test_clean_rows_drops_extras_and_empty_rows():
    rows = clean_rows(
        [
            {"serial": 3, "opv1": " 12-06-25 ", "child_name": "should not survive"},
            {"serial": None, "opv1": ""},
            "not a row",
        ]
    )
    assert len(rows) == 1
    assert rows[0]["serial"] == "3" and rows[0]["opv1"] == "12-06-25"
    assert "child_name" not in rows[0]
    assert clean_rows(None) == []


def test_a_timed_out_call_is_an_invalid_reading_not_a_crash(monkeypatch):
    import httpx
    import openai

    import formbench.vlm as vlm
    from formbench.register.prompts import page_schema
    from formbench.register.reader import _call

    def stuck(*args, **kwargs):
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://localhost"))

    monkeypatch.setattr(vlm, "complete_json", stuck)
    result = _call(None, page_schema(), "prompt", "model", False, 100, "test")
    assert result["json_valid"] is False and result["values"]["rows"] is None
    assert "timed out" in result["errors"][0]


def test_clean_rows_handles_a_failed_reading():
    assert clean_rows(None) == []
