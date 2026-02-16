from typing import Any, Dict, List


ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def default_result() -> Dict[str, Any]:
    return {
        "answer": "",
        "black_box_explanation": "",
        "assumptions": [],
        "evidence_claims": [],
        "uncertainty": [],
        "confidence": "low",
        "confidence_reason": "No valid model output parsed.",
        "followups": [],
    }


def to_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = default_result()

    out["answer"] = str(raw.get("answer", "")).strip()
    out["black_box_explanation"] = str(raw.get("black_box_explanation", "")).strip()
    out["assumptions"] = to_string_list(raw.get("assumptions"))
    out["uncertainty"] = to_string_list(raw.get("uncertainty"))
    out["followups"] = to_string_list(raw.get("followups"))

    confidence = str(raw.get("confidence", "low")).strip().lower()
    out["confidence"] = confidence if confidence in ALLOWED_CONFIDENCE else "low"
    out["confidence_reason"] = str(raw.get("confidence_reason", "")).strip()

    claims = raw.get("evidence_claims", [])
    clean_claims: List[Dict[str, Any]] = []
    if isinstance(claims, list):
        for item in claims:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            support_reason = str(item.get("support_reason", "")).strip()
            quote = str(item.get("quote", "")).strip()
            start = item.get("start") if isinstance(item.get("start"), int) else None
            end = item.get("end") if isinstance(item.get("end"), int) else None

            if claim or quote or support_reason:
                clean_claims.append(
                    {
                        "claim": claim,
                        "support_reason": support_reason,
                        "quote": quote,
                        "start": start,
                        "end": end,
                        "verified": False,
                    }
                )

    out["evidence_claims"] = clean_claims
    return out
