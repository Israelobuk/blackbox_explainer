import requests
import streamlit as st

from config import default_for_backend
from llm import create_client
from explain.pipeline import ExplainerPipeline

FOLLOWUP_SYSTEM_PROMPT = """
You are a serious technical assistant.
Rules:
1) Be direct, professional, and practical. No hype language.
2) Start with a direct answer first (1-2 sentences).
3) If the user asks a normal factual question, answer it plainly and stop.
4) Only use "supported/speculative/safer recommendation" when the user is explicitly asking to evaluate claims from context.
5) If information is weak, speculative, risky, or misleading, say that clearly.
6) Do not overstate certainty.
7) Keep answers concise and useful.
"""

st.set_page_config(page_title="Black Box Explainer", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --bg-main: #ffffff;
      --bg-card: #ffffff;
      --bg-panel: #f8fafc;
      --bg-input: #ffffff;
      --border: #263244;
      --text-main: #111827;
      --text-soft: #6b7280;
      --ok: #16a34a;
      --warn: #d97706;
      --bad: #dc2626;
      --brand: #2563eb;
    }

    .stApp {
      background: var(--bg-main);
      color: var(--text-main);
    }

    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
      background: #000000;
      border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
      color: #e5e7eb;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
      color: #d1d5db !important;
    }

    /* Dark inputs only inside sidebar */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox [role="combobox"] {
      background: #111827 !important;
      color: #f9fafb !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px !important;
    }

    /* Tabs + containers */
    [data-testid="stTabs"] button {
      color: #475569 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
      color: #0f172a !important;
      border-bottom-color: #3b82f6 !important;
    }
    [data-testid="stVerticalBlock"] [data-testid="stContainer"] {
      border-radius: 12px;
    }

    .hero {
      padding: 18px 20px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, #0a0f1a 0%, #070b13 100%);
      color: #f8fafc;
      margin-bottom: 14px;
    }
    .hero h1 {
      margin: 0;
      font-size: 1.9rem;
      line-height: 1.2;
      letter-spacing: .2px;
    }
    .hero p {
      margin: 6px 0 0;
      color: #cbd5e1;
    }

    .subtle {
      color: var(--text-soft);
      font-size: .92rem;
    }

    .metric-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 6px 0 4px;
    }
    .metric-chip {
      border: 1px solid #d1d5db;
      background: #ffffff;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: .86rem;
      color: #111827;
    }

    .pill {
      display: inline-block;
      padding: 5px 11px;
      border-radius: 999px;
      font-weight: 700;
      color: #fff;
      font-size: .82rem;
    }
    .pill-high { background: #166534; }
    .pill-medium { background: #b45309; }
    .pill-low { background: #991b1b; }

    .proof-ok {
      color: var(--ok);
      font-weight: 600;
      font-size: .9rem;
    }
    .proof-miss {
      color: var(--bad);
      font-weight: 600;
      font-size: .9rem;
    }

    .section-title {
      margin-top: .2rem;
    }
    </style>
    <div class="hero">
      <h1>Black Box Explainer</h1>
      <p>Local transparency dashboard for answer quality, evidence, uncertainty, and follow-up chat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def init_state():
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""
    if "followup_chat_history" not in st.session_state:
        st.session_state.followup_chat_history = []


def confidence_badge(conf: str) -> str:
    conf = (conf or "").lower()
    if conf == "high":
        css = "pill pill-high"
    elif conf == "medium":
        css = "pill pill-medium"
    else:
        css = "pill pill-low"
    return f"<span class='{css}'>{conf.upper() or 'LOW'}</span>"


def check_backend_ready(backend: str, base_url: str, model: str, timeout_seconds: int):
    if not base_url.strip() or not model.strip():
        return False, "Base URL and Model are required."

    timeout = min(max(timeout_seconds, 5), 30)
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
        if model not in models:
            return False, f"Ollama is running, but model '{model}' is not found. Run: ollama pull {model}"
        return True, f"Connected to Ollama. Model ready: {model}"
    except Exception as exc:
        return False, f"Cannot connect to Ollama at {base_url}. {exc}"


def render_bullet_list(items, empty_msg="- None listed."):
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.markdown(empty_msg)


def render_result(result: dict):
    claims = result.get("evidence_claims", [])
    verified_count = sum(1 for claim in claims if claim.get("verified"))

    st.markdown("### Result")
    c1, c2, c3 = st.columns([1.2, 1.3, 2.5])
    with c1:
        st.markdown("**Confidence**")
        st.markdown(confidence_badge(result.get("confidence", "low")), unsafe_allow_html=True)
    with c2:
        st.markdown("**Evidence Match**")
        st.markdown(f"`{verified_count}/{len(claims)}`")
    with c3:
        st.markdown("**Confidence reason**")
        st.markdown(result.get("confidence_reason", ""))

    st.markdown(
        "<p class='subtle'>Confidence reflects support from your provided context. "
        "High = strong support, Medium = partial support, Low = weak or missing support.</p>",
        unsafe_allow_html=True,
    )

    tab_answer, tab_blackbox, tab_evidence, tab_context, tab_risks = st.tabs(
        ["Answer", "Black Box", "Evidence", "Context", "Risks & Follow-ups"]
    )

    with tab_answer:
        st.markdown("#### Direct Answer")
        st.write(result.get("answer", ""))

    with tab_blackbox:
        st.markdown("#### Black Box Explanation (How The Model Got There)")
        black_box_text = result.get("black_box_explanation", "").strip()
        if black_box_text:
            st.write(black_box_text)
        else:
            st.info("The model did not provide a black-box explanation for this run.")

    with tab_evidence:
        st.markdown("#### Model's Black-Box Explanation (From Your Context)")
        st.markdown("<p class='subtle'>Model interpretation + exact quote verification.</p>", unsafe_allow_html=True)
        if claims:
            for i, claim in enumerate(claims, start=1):
                claim_text = claim.get("claim", "") or "Claim"
                quote = claim.get("quote", "")
                support_reason = claim.get("support_reason", "")

                with st.container(border=True):
                    st.markdown(f"**{i}. {claim_text}**")
                    st.markdown(f"> {quote}")
                    if support_reason:
                        st.caption(f"Model explanation: {support_reason}")
                    if claim.get("verified"):
                        st.markdown("<span class='proof-ok'>Found in your context</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<span class='proof-miss'>No matching quote found in your context</span>",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No supporting snippets were returned.")

    with tab_context:
        st.markdown("#### Context with Highlights")
        st.markdown(result.get("highlighted_context", ""), unsafe_allow_html=True)

    with tab_risks:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Assumptions")
            render_bullet_list(result.get("assumptions", []), "- None listed.")
            st.markdown("#### Uncertainty / What Could Be Wrong")
            render_bullet_list(result.get("uncertainty", []), "- None listed.")
        with c_right:
            st.markdown("#### Helpful What-If Questions")
            render_bullet_list(result.get("followups", []), "- None.")


def render_chat(backend: str, base_url: str, model: str, timeout_seconds: int, temperature: float, max_tokens: int):
    st.markdown("### Talk to the Model")
    st.markdown("<p class='subtle'>Follow-up conversation using the same local backend and model.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.followup_chat_history = []
            st.rerun()

    for msg in st.session_state.followup_chat_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg.get("content", ""))

    chat_input = st.chat_input("Ask a follow-up...")

    if chat_input:
        user_text = chat_input.strip()
        if not user_text:
            return

        st.session_state.followup_chat_history.append({"role": "user", "content": user_text})

        client = create_client(
            backend=backend,
            base_url=base_url,
            model=model,
            timeout_seconds=int(timeout_seconds),
        )

        chat_messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Original question:\n"
                    f"{st.session_state.last_question}\n\n"
                    "Context:\n"
                    f"{st.session_state.last_context}\n\n"
                    "Follow-up question:\n"
                    f"{user_text}\n\n"
                    "Instruction:\n"
                    "If this follow-up is unrelated to the context, answer directly without forcing context analysis."
                ),
            },
        ]

        with st.spinner("Getting follow-up response..."):
            reply = client.chat(messages=chat_messages, temperature=temperature, max_tokens=max_tokens)
        st.session_state.followup_chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


init_state()

with st.sidebar:
    st.markdown("## Backend Settings")
    backend = "ollama"
    st.selectbox("Provider", ["ollama"], index=0, disabled=True)
    defaults = default_for_backend(backend)

    base_url = st.text_input("Base URL", value=defaults.base_url)
    model = st.text_input("Model", value=defaults.model)
    temperature = st.slider(
        "Response creativity",
        0.0,
        1.0,
        float(defaults.temperature),
        0.05,
        help="Lower = consistent and factual. Higher = more varied wording.",
    )
    timeout_seconds = st.number_input("Timeout (seconds)", min_value=5, max_value=600, value=120, step=5)
    critique_pass = st.toggle("Critique pass (second model call)", value=False)

    ready, status = check_backend_ready(backend, base_url, model, int(timeout_seconds))
    if ready:
        st.success(status)
    else:
        st.error(status)

st.markdown("### Input")
question = st.text_input("Question", placeholder="Ask a specific question...")
context = st.text_area("Context", height=300, placeholder="Paste source/context text here...")

action_col1, action_col2 = st.columns([1.2, 5])
with action_col1:
    run = st.button("Explain", type="primary", use_container_width=True)
with action_col2:
    st.markdown("<p class='subtle'>Run analysis after backend shows connected status in sidebar.</p>", unsafe_allow_html=True)

if run:
    if not question.strip():
        st.error("Question is required.")
    elif not context.strip():
        st.error("Context is required.")
    elif not ready:
        st.error("Backend is not ready. Fix backend settings in the sidebar first.")
    else:
        try:
            client = create_client(
                backend=backend,
                base_url=base_url,
                model=model,
                timeout_seconds=int(timeout_seconds),
            )
            pipeline = ExplainerPipeline(client)

            with st.spinner("Running local explainer..."):
                result = pipeline.run(
                    question=question.strip(),
                    context=context,
                    temperature=float(temperature),
                    max_tokens=int(defaults.max_tokens),
                    critique_pass=bool(critique_pass),
                )

            st.session_state.last_result = result
            st.session_state.last_question = question.strip()
            st.session_state.last_context = context
            st.session_state.followup_chat_history = []
        except Exception as exc:
            st.error(
                "Failed to run local backend. Ensure Ollama is running and base URL/model are correct."
            )
            st.exception(exc)

if st.session_state.last_result is not None:
    st.divider()
    render_result(st.session_state.last_result)
    st.divider()
    if ready:
        render_chat(
            backend=backend,
            base_url=base_url,
            model=model,
            timeout_seconds=int(timeout_seconds),
            temperature=float(temperature),
            max_tokens=int(defaults.max_tokens),
        )
    else:
        st.info("Fix backend connection in the sidebar to use follow-up chat.")
