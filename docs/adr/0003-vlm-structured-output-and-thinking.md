# ADR 0003: Schema-constrained VLM output with thinking off by default

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The VLM must return values for the keys of the form's generated JSON Schema, and nothing else.
It never sees or emits DHIS2 UIDs; `mapping.yaml` maps keys to UIDs in plain code. Runs must be
repeatable and traced in Langfuse through `langfuse.openai`, which wraps the OpenAI client. So
the preferred transport is Ollama's **OpenAI-compatible API** (`/v1/chat/completions`), with
Ollama's native `/api/chat` as a fallback.

Two open questions:

1. Can the OpenAI-compatible API enforce our JSON Schema?
2. Can it turn "thinking" off for reasoning models? Thinking adds latency and tokens to what is
   a transcription task.

## Findings (Ollama 0.40.11, openai SDK 2.48.0, langfuse 3.7.0)

Tested directly against a local Ollama:

| Request | Result |
| --- | --- |
| `response_format: {type: "json_schema", json_schema: {strict: true, schema}}` | Output follows the schema (types, required keys, no extras). |
| qwen3.5:4b, no `reasoning_effort` | Thinks: ~2,480 completion tokens, ~9.6k chars of reasoning. |
| qwen3.5:4b, `reasoning_effort: "none"` | No reasoning, 20 completion tokens, same JSON. |
| granite-vision-4.1-4b (not a thinking model), `reasoning_effort: "none"` | Accepted, no error. |

`"none"` is a valid `ReasoningEffort` value in the installed openai SDK. The Langfuse wrapper
passes `response_format` through and records the generation.

## Decision

- Use the **OpenAI-compatible API through `langfuse.openai.OpenAI`** for all VLM calls. The
  native `/api/chat` fallback is not needed.
- Send the subset JSON Schema as a **strict `json_schema` `response_format`**, with the
  `$schema` header removed. The prompt also lists every `key: description`, so models that
  ignore constraints still know the keys.
- **Thinking is off by default** (`reasoning_effort="none"`). `extract --think` omits the
  parameter, which leaves the model's default (thinking on for thinking models). The setting
  is recorded in the run file and the Langfuse trace metadata.
- Decoding is fixed: `temperature=0`, `seed=42`. The photo is downscaled to at most 2048 px on
  its long side before encoding. `PROMPT_VERSION` is bumped whenever the prompt changes.
- **Never trust the constraint alone.** Replies are parsed defensively (code fences stripped,
  first JSON object extracted) and validated with `jsonschema` (Draft 2020-12).
  - Unparseable output scores as `json_valid = 0` with every field `null`.
  - Parseable but schema-invalid output scores as `json_valid = 0`, and its fields are still
    scored.
  - Neither case crashes the run.

## Consequences

- With constrained decoding, `json_valid` should be near 1 on Ollama. It stays a metric
  because other backends (llama.cpp, MLX, on-device runtimes) may not support constraints.
- Constrained decoding can hide a model's uncertainty: it has to emit *some* value or `null`
  for every key. Hallucination and omission rates capture the result.
- If a future Ollama drops `reasoning_effort` mapping, the fallback is native `/api/chat` with
  `think: false` and `format: <schema>`. That would need manual Langfuse instrumentation.
