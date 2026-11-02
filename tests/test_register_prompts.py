import pytest
from jsonschema import Draft202012Validator

from formbench.register.prompts import (
    cover_schema,
    page_prompt,
    page_schema,
    rows_schema,
)
from formbench.register.spec import DOSE_COLUMNS

ROW = {
    "serial": "1",
    "card_number": "C-104",
    "sex": "F",
    "birth_date": "03-04-25",
    "age": "6/52",
    **{c.key: None for c in DOSE_COLUMNS},
}


def test_page_reply_validates():
    reply = {"cohort_month": "April", "cohort_year": "2025", "rows": [ROW]}
    assert not list(Draft202012Validator(page_schema()).iter_errors(reply))


def test_personal_data_cannot_be_returned():
    reply = {"rows": [{**ROW, "child_name": "Jane Lee Brown"}]}
    errors = list(Draft202012Validator(rows_schema()).iter_errors(reply))
    assert errors and "child_name" in errors[0].message
    for schema in (page_schema(), cover_schema()):
        text = str(schema).lower()
        assert "phone" not in text and "guardian" not in text and "village" not in text


def test_every_dose_column_is_requested():
    row = rows_schema()["properties"]["rows"]["items"]
    assert {c.key for c in DOSE_COLUMNS} <= set(row["required"])
    prompt = page_prompt()
    assert all(f"- {c.key}: column {c.letter} " in prompt for c in DOSE_COLUMNS)
    assert "Do not transcribe child names" in prompt


class FakePrompts:
    """Langfuse prompt management in memory: versions of one prompt name, with labels."""

    def __init__(self, versions=(), labels=None, reachable=True):
        self.versions, self.labels, self.reachable, self.created = list(versions), labels or {}, reachable, []  # fmt: skip

    def _prompt(self, n):
        from types import SimpleNamespace

        return SimpleNamespace(version=n, compile=lambda: self.versions[n - 1])

    def get_prompt(self, name, *, version=None, label=None, **_):
        from langfuse.api import NotFoundError

        if not self.reachable:
            raise ConnectionError("down")
        labels = {**self.labels, "latest": len(self.versions)} if self.versions else self.labels
        n = version if version else labels.get(label)
        if not n or n > len(self.versions):
            raise NotFoundError(body={"message": "not found"})
        return self._prompt(n)

    def create_prompt(self, *, name, prompt, labels, **_):
        self.versions.append(prompt)
        self.created.append((name, labels))
        return self._prompt(len(self.versions))


def test_the_prompt_in_code_becomes_version_1_in_langfuse():
    from formbench.register.prompts import load_prompt

    langfuse = FakePrompts()
    prompt = load_prompt(langfuse, "page")
    assert prompt.version == 1 and prompt.compile() == page_prompt()
    assert langfuse.created == [("register-page", ["production"])]
    langfuse.labels["production"] = 1
    assert load_prompt(langfuse, "page").version == 1 and len(langfuse.created) == 1


def test_a_run_picks_a_prompt_version_by_number_or_label():
    from formbench.register.prompts import load_prompt

    langfuse = FakePrompts(["v1", "v2", "v3"], {"production": 1, "candidate": 3})
    assert load_prompt(langfuse, "page").compile() == "v1"
    assert load_prompt(langfuse, "page", "2").compile() == "v2"
    assert load_prompt(langfuse, "page", "candidate").compile() == "v3"
    for missing in ("9", "typo"):
        with pytest.raises(SystemExit, match="no register-page prompt"):
            load_prompt(langfuse, "page", missing)
    assert not langfuse.created


def test_an_asked_for_version_is_never_replaced_by_another():
    from formbench.register.prompts import load_prompt

    with pytest.raises(SystemExit, match="'production'"):  # production label moved away
        load_prompt(FakePrompts(["v1"], {}), "page")
    with pytest.raises(SystemExit):  # nothing saved yet: a label does not create version 1
        load_prompt(FakePrompts(), "page", "candidate")
    with pytest.raises(SystemExit, match="not reachable"):
        load_prompt(FakePrompts(reachable=False), "page", "2")
    assert load_prompt(FakePrompts(reachable=False), "page") is None  # the prompt in code


def test_the_model_gets_the_prompt_text_and_the_trace_its_version(monkeypatch):
    import formbench.vlm as vlm
    from formbench.register.reader import read_page

    calls = []
    monkeypatch.setattr(vlm, "complete_json", lambda *a, **k: calls.append((a, k)) or '{"rows": []}')  # fmt: skip
    prompt = FakePrompts(["Read the page."])._prompt(1)
    read_page(None, "m", prompt=prompt)
    read_page(None, "m")
    (args, kwargs), (code_args, code_kwargs) = calls
    assert args[2] == "Read the page." and kwargs["langfuse_prompt"] is prompt
    assert kwargs["metadata"]["prompt_version"] == 1 and kwargs["name"] == "register-page"
    assert code_args[2] == page_prompt() and code_kwargs["langfuse_prompt"] is None
