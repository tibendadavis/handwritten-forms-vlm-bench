"""Prompts versioned in Langfuse prompt management, shared by every kind of form.

The text in code is only the first version: the first run that finds no prompt saves it with
the production label. After that, prompts are edited in Langfuse and each run picks a version.
"""

from __future__ import annotations

from typing import Any


def prompt_version(prompt: Any) -> int | None:
    """The Langfuse version of a prompt; None for the prompt in code."""
    return None if prompt is None else prompt.version


def fetch_prompt(langfuse: Any, name: str, default: str, choice: str | None = None) -> Any:
    """The prompt `name` from Langfuse: `choice` is a version number or a label (default: the
    version labelled production). The first time, `default` is saved as version 1 with the
    production label. Returns None when Langfuse is not reachable and no version was asked
    for: the prompt in code is used, linked to no version."""
    from langfuse.api import NotFoundError

    version = int(choice) if choice and choice.isdigit() else None
    label = None if version else choice or "production"
    try:
        return langfuse.get_prompt(name, version=version, label=label, cache_ttl_seconds=0, max_retries=0)  # fmt: skip
    except NotFoundError:
        pass
    except Exception as error:  # no keys, or the server is down
        if choice:
            raise SystemExit(f"Langfuse not reachable for prompt {name} {choice} ({error})") from None  # fmt: skip
        return None
    try:
        langfuse.get_prompt(name, label="latest", cache_ttl_seconds=0, max_retries=0)
    except NotFoundError:
        if not choice:  # never saved: the prompt in code becomes version 1
            return langfuse.create_prompt(
                name=name, prompt=default, labels=["production"], tags=["formbench"],
                commit_message="The prompt in code (formbench), first version",
            )  # fmt: skip
    # What was asked for does not exist: never run another version instead.
    raise SystemExit(f"Langfuse has no {name} prompt {label or version!r}. Prompts > {name} in "
                     "Langfuse lists its versions and labels.")  # fmt: skip
