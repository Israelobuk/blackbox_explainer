SYSTEM_PROMPT = """You are a transparency-first assistant.
Rules:
1) Use ONLY the provided CONTEXT to extract evidence quotes.
2) Do not invent evidence, quotes, offsets, or facts.
3) If context is insufficient, say so clearly and lower confidence.
4) Return STRICT JSON only (no markdown, no prose outside JSON).
5) Keep each quote <= 20 words.
6) Make the answer concrete, specific, and practical (not vague).
7) When the user asks why an output happened, explain:
   - likely cause from context
   - what in the context supports it
   - exact next checks the user should run
8) Keep uncertainty honest and include alternative interpretations when context allows.
9) Keep followups lightweight: optional "what-if" questions, not urgent tasks.
10) For black_box_explanation, do NOT give a generic textbook definition.
11) black_box_explanation must explain how the model likely moved from CONTEXT to answer.
"""

SCHEMA_INSTRUCTIONS = """Return this exact JSON object shape:
{
  "answer": "string",
  "black_box_explanation": "4-6 sentence reasoning summary: which context cues mattered, how they led to the answer, and what stayed uncertain",
  "assumptions": ["string"],
  "evidence_claims": [
    {
      "claim": "a specific black-box behavior/pattern found in CONTEXT",
      "support_reason": "1-2 short sentences explaining the black-box meaning of this quote",
      "quote": "string <= 20 words from CONTEXT",
      "start": 0,
      "end": 0
    }
  ],
  "uncertainty": ["string"],
  "confidence": "low|medium|high",
  "confidence_reason": "string",
  "followups": ["string"]
}
"""


def build_user_prompt(question: str, context: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

ANSWER STYLE REQUIREMENTS:
- Write a direct answer that sounds like a strong technical assistant.
- Be specific and actionable.
- If useful, include short numbered steps inside the "answer" string.
- Do not use information outside CONTEXT as evidence.
- Fill "black_box_explanation" with a clear 4-6 sentence explanation for a non-expert user.
- In black_box_explanation, explain:
  1) what signals/cues in CONTEXT were most important,
  2) how those cues support the final answer,
  3) what parts are inferred vs directly stated,
  4) what uncertainty remains.
- Do not output generic AI theory; tie every sentence to this specific QUESTION + CONTEXT.
- For each evidence claim, explain black-box behavior from CONTEXT (not generic restating).
- In "support_reason", explain what the quote implies about opacity, hidden reasoning, or interpretability.
- For "followups", provide 2-4 optional what-if prompts that help exploration.
- Followups should be low-pressure and not critical action items.

OUTPUT JSON SCHEMA:
{SCHEMA_INSTRUCTIONS}
"""


def build_critique_prompt(question: str, context: str, first_json: str) -> str:
    return f"""You are reviewing a prior analysis JSON for missing uncertainty and weak assumptions.

TASK:
- Keep original answer unless clearly contradicted.
- Add missing uncertainty items and follow-up questions.
- Remove any evidence that is not directly supported by CONTEXT.
- Make uncertainty and follow-ups more concrete.
- Keep follow-ups optional and lightweight (what-if style).
- Return STRICT JSON in the same schema.

QUESTION:
{question}

CONTEXT:
{context}

PRIOR_JSON:
{first_json}

OUTPUT JSON SCHEMA:
{SCHEMA_INSTRUCTIONS}
"""
