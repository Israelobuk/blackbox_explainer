from typing import Any, Dict, List, Optional, Tuple
import html
import re

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


def get_quote_position(context: str, quote: str) -> Tuple[Optional[int], Optional[int]]:
    # Try exact match first, then case-insensitive match.
    if not quote:
        return None, None

    start = context.find(quote)
    if start != -1:
        return start, start + len(quote)

    start = context.lower().find(quote.lower())
    if start != -1:
        return start, start + len(quote)

    # Normalize punctuation/whitespace differences.
    def norm(text: str) -> str:
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    norm_context = norm(context)
    norm_quote = norm(quote)
    start = norm_context.lower().find(norm_quote.lower())
    if start != -1:
        first_word = next((w for w in norm_quote.split(" ") if w), "")
        if first_word:
            raw_start = context.lower().find(first_word.lower())
            if raw_start != -1:
                return raw_start, min(len(context), raw_start + len(quote))

    # Fuzzy fallback for small formatting drift.
    if fuzz is not None:
        try:
            align = fuzz.partial_ratio_alignment(quote, context, score_cutoff=88)
            if align is not None:
                return align.dest_start, align.dest_end
        except Exception:
            pass

    return None, None


def verify_evidence_claims(result: Dict[str, Any], context: str) -> Dict[str, Any]:
    # Check every quote and mark whether it was really found in the context.
    checked: List[Dict[str, Any]] = []

    for claim in result.get("evidence_claims", []):
        quote = str(claim.get("quote", "")).strip()
        start, end = get_quote_position(context, quote)

        if start is None or end is None:
            checked.append(
                {
                    "claim": claim.get("claim", ""),
                    "support_reason": claim.get("support_reason", ""),
                    "quote": "EVIDENCE_NOT_FOUND",
                    "start": None,
                    "end": None,
                    "verified": False,
                }
            )
        else:
            checked.append(
                {
                    "claim": claim.get("claim", ""),
                    "support_reason": claim.get("support_reason", ""),
                    "quote": context[start:end],
                    "start": start,
                    "end": end,
                    "verified": True,
                }
            )

    result["evidence_claims"] = checked
    return result


def _keyword_tokens(text: str) -> set:
    # Small stopword list so overlap focuses on meaningful words.
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
        "and", "or", "it", "this", "that", "with", "as", "at", "by", "from", "why", "what",
        "how", "when", "where", "who", "which", "does", "do", "did", "can", "could", "would",
        "should", "will", "you", "your", "i", "we", "they", "he", "she", "them", "his", "her",
    }
    words = []
    for raw in text.lower().split():
        clean = "".join(ch for ch in raw if ch.isalnum())
        if len(clean) >= 3 and clean not in stop:
            words.append(clean)
    return set(words)


def add_question_relevance(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    # Mark each evidence claim as relevant/weak based on keyword overlap with the question.
    q_tokens = _keyword_tokens(question)
    for claim in result.get("evidence_claims", []):
        claim_text = str(claim.get("claim", ""))
        quote_text = str(claim.get("quote", ""))
        c_tokens = _keyword_tokens(claim_text + " " + quote_text)
        overlap = len(q_tokens.intersection(c_tokens))

        if overlap > 0:
            claim["question_relevance"] = "relevant"
            claim["relevance_reason"] = "This evidence shares key terms with your question."
        else:
            claim["question_relevance"] = "weak"
            claim["relevance_reason"] = "This evidence may be true, but it does not clearly address your question."
    return result


def adjust_confidence(result: Dict[str, Any]) -> Dict[str, Any]:
    # Lower confidence when evidence is missing or weak.
    claims = result.get("evidence_claims", [])
    if not claims:
        result["confidence"] = "low"
        msg = result.get("confidence_reason", "")
        result["confidence_reason"] = (msg + " No evidence claims were provided.").strip()
        return result

    verified = sum(1 for c in claims if c.get("verified"))
    total = len(claims)

    if verified == 0:
        result["confidence"] = "low"
        msg = result.get("confidence_reason", "")
        result["confidence_reason"] = (msg + " None of the evidence quotes were found in context.").strip()
    elif verified < total and result.get("confidence") == "high":
        result["confidence"] = "medium"
        msg = result.get("confidence_reason", "")
        result["confidence_reason"] = (msg + " Some evidence quotes could not be verified.").strip()

    weak_relevance = sum(1 for c in claims if c.get("question_relevance") == "weak")
    if weak_relevance == len(claims) and len(claims) > 0:
        result["confidence"] = "low"
        msg = result.get("confidence_reason", "")
        result["confidence_reason"] = (
            msg + " Evidence quotes were found, but they do not clearly answer the question."
        ).strip()

    return result


def build_highlighted_context(context: str, evidence_claims: List[Dict[str, Any]]) -> str:
    # Build a context string where evidence spans are wrapped in <mark>...</mark>.
    spans: List[Tuple[int, int]] = []
    for claim in evidence_claims:
        start = claim.get("start")
        end = claim.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context):
            spans.append((start, end))

    if not spans:
        return html.escape(context)

    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(html.escape(context[cursor:start]))
        parts.append("<mark>" + html.escape(context[start:end]) + "</mark>")
        cursor = end

    parts.append(html.escape(context[cursor:]))
    return "".join(parts)
