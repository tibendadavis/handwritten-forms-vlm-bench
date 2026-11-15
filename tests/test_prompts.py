import pytest

from formbench.prompts import (
    PAGE_PROMPT,
    REGISTER_PAGE_PROMPT,
    SECTION_PROMPT,
    PromptError,
    compile_prompt,
    fetch_prompt,
    prompt_version,
)
from tests.conftest import FakeLangfuse

NAME = "tally-weekly-tally"


@pytest.fixture
def stocked():
    langfuse = FakeLangfuse()
    langfuse.add_prompt(NAME, "v1 {{section}}", ["old"])
    langfuse.add_prompt(NAME, "v2 {{section}}", ["production"])
    langfuse.add_prompt(NAME, "v3 {{section}}", ["candidate"])
    return langfuse


def test_production_is_used_by_default(stocked):
    prompt = fetch_prompt(stocked, NAME, "code", None)
    assert prompt_version(prompt) == 2
    assert stocked.calls[0] == (
        "get_prompt",
        {
            "name": NAME,
            "version": None,
            "label": "production",
            "cache_ttl_seconds": 0,
            "max_retries": 0,
        },
    )


def test_a_number_picks_that_version(stocked):
    assert fetch_prompt(stocked, NAME, "code", "1").version == 1
    assert stocked.calls[0][1]["version"] == 1 and stocked.calls[0][1]["label"] is None


@pytest.mark.parametrize("label, version", [("candidate", 3), ("latest", 3), ("old", 1)])
def test_a_label_picks_that_label(stocked, label, version):
    assert fetch_prompt(stocked, NAME, "code", label).version == version


def test_missing_prompt_without_choice_creates_v1_from_code(langfuse):
    prompt = fetch_prompt(langfuse, "register-page", REGISTER_PAGE_PROMPT)
    assert prompt.version == 1 and prompt.prompt == REGISTER_PAGE_PROMPT
    created = [kw for name, kw in langfuse.calls if name == "create_prompt"]
    assert created == [
        {
            "name": "register-page",
            "prompt": REGISTER_PAGE_PROMPT,
            "labels": ["production"],
            "tags": ["formbench"],
            "commit_message": "The prompt in code (formbench), first version",
        }
    ]


@pytest.mark.parametrize("choice", ["7", "candidate"])
def test_missing_prompt_with_a_choice_stops(langfuse, choice):
    with pytest.raises(PromptError, match=f"no register-page prompt '{choice}'"):
        fetch_prompt(langfuse, "register-page", "code", choice)
    assert not [c for c in langfuse.calls if c[0] == "create_prompt"]


@pytest.mark.parametrize("choice", ["9", "nope"])
def test_missing_version_stops_and_never_falls_back(stocked, choice):
    with pytest.raises(PromptError) as error:
        fetch_prompt(stocked, NAME, "code", choice)
    assert str(error.value) == (
        f"Langfuse has no {NAME} prompt '{choice}'. "
        f"Prompts > {NAME} in Langfuse lists its versions and labels."
    )
    assert not [c for c in stocked.calls if c[0] == "create_prompt"]


def test_existing_prompt_without_production_label_stops():
    langfuse = FakeLangfuse()
    langfuse.add_prompt(NAME, "v1", ["candidate"])
    with pytest.raises(PromptError, match="'production'"):
        fetch_prompt(langfuse, NAME, "code")


@pytest.mark.parametrize("choice", [None, "3"])
def test_langfuse_unreachable_stops(choice):
    with pytest.raises(PromptError, match="not reachable"):
        fetch_prompt(FakeLangfuse(down=True), NAME, "code", choice)


def test_compile_fills_variables(stocked):
    assert compile_prompt(fetch_prompt(stocked, NAME, "code"), {"section": "B"}) == "v2 B"


def test_code_defaults_have_their_variables():
    assert "{{" not in REGISTER_PAGE_PROMPT
    for name in ("{{section}}", "{{title}}", "{{fields}}"):
        assert name in SECTION_PROMPT
    assert "{{fields}}" in PAGE_PROMPT and "{{section}}" not in PAGE_PROMPT
