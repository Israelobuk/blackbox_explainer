from typing import Any, Dict, List
import json

from explain.prompts import SYSTEM_PROMPT, SCHEMA_INSTRUCTIONS, build_user_prompt, build_critique_prompt
from explain.schemas import default_result, normalize_result
from explain.highlight import verify_evidence_claims, add_question_relevance, adjust_confidence, build_highlighted_context
from utils.logging import build_trace_log


def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found.")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("No balanced JSON object found.")


def get_json_from_text(text: str) -> Dict[str, Any]:
    # First try direct JSON parsing.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # If model wrapped JSON in extra text, pull the first balanced JSON object.
    try:
        candidate = _extract_balanced_json_object(text)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, json.JSONDecodeError):
        pass

    raise ValueError("Could not parse JSON object from LLM output.")


def combine_unique_items(base: List[str], extra: List[str]) -> List[str]:
    # Keep order, remove duplicates (case-insensitive).
    seen = set()
    out: List[str] = []
    for item in base + extra:
        text = item.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


class ExplainerPipeline:
    def __init__(self, client):
        self.client = client

    def run(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        critique_pass: bool = False,
    ) -> Dict[str, Any]:
        steps = [
            "llm_primary_call",
            "parse_json",
            "normalize_schema",
            "verify_evidence",
            "adjust_confidence",
        ]
        raw_text = ""

        try:
            # 1) Ask model for structured JSON answer.
            primary_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ]
            raw_text = self.client.chat(primary_messages, temperature=temperature, max_tokens=max_tokens)

            try:
                result = normalize_result(get_json_from_text(raw_text))
            except ValueError:
                # Retry once by asking model to convert prior output into strict JSON only.
                steps.append("llm_json_repair_call")
                repair_messages = [
                    {
                        "role": "system",
                        "content": "Convert text to STRICT valid JSON only. No markdown, no comments, no extra text.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Return valid JSON matching this schema exactly:\n"
                            f"{SCHEMA_INSTRUCTIONS}\n\n"
                            "TEXT TO CONVERT:\n"
                            f"{raw_text}"
                        ),
                    },
                ]
                repaired = self.client.chat(repair_messages, temperature=0.0, max_tokens=max_tokens)
                result = normalize_result(get_json_from_text(repaired))

            if critique_pass:
                steps.append("llm_critique_call")
                # 2) Optional second pass to improve assumptions/uncertainty.
                critique_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_critique_prompt(
                            question,
                            context,
                            json.dumps(result, ensure_ascii=False),
                        ),
                    },
                ]
                critique_raw = self.client.chat(critique_messages, temperature=temperature, max_tokens=max_tokens)
                critique = normalize_result(get_json_from_text(critique_raw))

                result["assumptions"] = combine_unique_items(
                    result.get("assumptions", []), critique.get("assumptions", [])
                )
                result["uncertainty"] = combine_unique_items(
                    result.get("uncertainty", []), critique.get("uncertainty", [])
                )
                result["followups"] = combine_unique_items(
                    result.get("followups", []), critique.get("followups", [])
                )
                if critique.get("evidence_claims"):
                    result["evidence_claims"] = critique["evidence_claims"]
                if critique.get("answer"):
                    result["answer"] = critique["answer"]
                if critique.get("black_box_explanation"):
                    result["black_box_explanation"] = critique["black_box_explanation"]
                if critique.get("confidence") in {"low", "medium", "high"}:
                    result["confidence"] = critique["confidence"]
                if critique.get("confidence_reason"):
                    result["confidence_reason"] = critique["confidence_reason"]

            # 3) Deterministic checks: verify evidence + question relevance + adjust confidence.
            result = verify_evidence_claims(result, context)
            result = add_question_relevance(result, question)
            result = adjust_confidence(result)

        except Exception as exc:
            # If anything fails, return a safe low-confidence response.
            result = default_result()
            result["answer"] = "Unable to produce a reliable answer from the local model."
            result["uncertainty"] = [f"Pipeline error: {exc}"]
            result["confidence"] = "low"
            result["confidence_reason"] = "Local model call or JSON parsing failed."
            result["evidence_claims"] = []

        # 4) Prepare UI extras.
        result["highlighted_context"] = build_highlighted_context(context, result.get("evidence_claims", []))
        result["trace_log"] = build_trace_log(
            backend_meta=self.client.metadata(),
            temperature=temperature,
            max_tokens=max_tokens,
            steps=steps,
            raw_preview=raw_text[:500] if raw_text else "",
        )
        return result
