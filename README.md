# blackbox_explainer

A Python tool for providing contextual explanations behind LLM responses.

## What This Does

Language models generate answers without exposing the reasoning that led to them.

`blackbox_explainer` helps analyze a model's output and surface:

- Supporting phrases or evidence within the response
- Structured reasoning patterns
- Prompt influence signals
- Output formatting logic

The goal is not to retrain or modify the model, but to provide transparency into how a response may have been constructed.

---

## How It Works

1. A response is generated using a selected local LLM backend (LM Studio or Ollama).
2. The explanation pipeline analyzes the output.
3. Structured highlighting and formatting logic extract reasoning cues.
4. The result is returned in a consistent, inspectable format.

---

## Structure

- `app.py` — Entry point
- `config.py` — Configuration settings
- `llm/` — Model client implementations
- `explain/` — Explanation and analysis logic
- `utils/` — Shared utilities

---

## Requirements

- Python 3.12+
- Local LLM backend (LM Studio or Ollama)
