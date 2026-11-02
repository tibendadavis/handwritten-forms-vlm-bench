# ADR 0008: Prompts versioned in Langfuse

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

With the models and photos fixed, the prompt is the next thing to tune. A prompt edited in
code leaves no record of which wording produced which scores, and a hand-bumped version
string (`r2`) is easy to forget. Langfuse already holds the runs, traces and scores.

## Decision

**The page and cover prompts live in Langfuse prompt management** as `register-page` and
`register-cover` (text prompts). The text in `register/prompts.py` is only the first version:
the first run that finds no prompt saves it with the `production` label. After that, prompts
are edited in Langfuse, each edit a new version with a commit message.

**Each run names its prompt version.** `--prompt` on `bench` and `eval` takes a version number
or a label; the default is `production`. Every model call is linked to that version (the
OpenAI wrapper's `langfuse_prompt`), so Langfuse lists the calls and scores per version. The
dataset run is named after the version, and the saved run records `prompt_version`, so
`formbench compare` shows one column per model and prompt version.

**Never a different prompt than asked for.** A version or label that does not exist stops the
run; so does an unreachable Langfuse when a version was asked for. Only with no `--prompt` and
no Langfuse does a run fall back to the text in code, recorded as no version.

**The schema stays in code.** A prompt changes what the model is told, not what it must
return: keys, descriptions and the absence of personal fields remain in the JSON schema.

**Weekly runs use `production`**, so promoting a version in Langfuse changes what is used.

## Consequences

- Prompt experiments need no code change or commit; Langfuse keeps every version.
- Prompt versions live on the Langfuse server, not in git: back up its volumes.
- Runs saved before this change have no prompt version and show under the model name only.
- `REGISTER_PROMPT_VERSION` is removed.
