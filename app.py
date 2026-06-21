import streamlit as st
from main import ask, MODELS, PRICES

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multi-Model Comparison Tool", layout="wide")

st.title("Multi-Model Comparison Tool")
st.caption(
    "Ask one question to multiple LLMs via OpenRouter "
    "and compare their answers, speed, and cost side by side."
)

st.divider()

# ── Build a name→id lookup from the shared MODELS list ───────────────────────

MODEL_MAP = {name: model_id for name, model_id in MODELS}

# ── Inputs ────────────────────────────────────────────────────────────────────

question = st.text_area(
    "Your question",
    placeholder="e.g. What is the capital of France, and what is it famous for?",
    height=100,
)

selected_names = st.multiselect(
    "Models to compare",
    options=list(MODEL_MAP.keys()),
    default=list(MODEL_MAP.keys()),
    help="Choose one or more models. All are free-tier via OpenRouter.",
)

running = st.session_state.get("running", False)
run = st.button(
    "Compare models",
    type="primary",
    disabled=not question or not selected_names or running,
)

# ── Run ───────────────────────────────────────────────────────────────────────

if run:
    if not question.strip():
        st.warning("Please enter a question before comparing models.")
    else:
        st.session_state["running"] = True
        results = []

        with st.spinner(f"Querying {len(selected_names)} model(s)…"):
            for name in selected_names:
                model_id = MODEL_MAP[name]
                try:
                    answer, latency, in_tok, out_tok, cost = ask(question, model_id)
                    results.append({
                        "name":    name,
                        "answer":  answer,
                        "latency": latency,
                        "in_tok":  in_tok,
                        "out_tok": out_tok,
                        "cost":    cost,
                        "error":   None,
                    })
                except Exception as e:
                    results.append({
                        "name":    name,
                        "answer":  None,
                        "latency": None,
                        "in_tok":  None,
                        "out_tok": None,
                        "cost":    None,
                        "error":   str(e),
                    })

        st.session_state["running"] = False
        st.session_state["results"] = results

# ── Display stored results (persists across reruns) ───────────────────────────

if "results" in st.session_state:
    st.divider()
    st.subheader("Results")

    results = st.session_state["results"]
    cols = st.columns(len(results))

    for col, r in zip(cols, results):
        with col:
            st.markdown(f"### {r['name']}")

            if r["error"]:
                st.error(r["error"])
            else:
                st.write(r["answer"])
                st.metric("Latency", f"{r['latency']:.2f} s")
                st.metric("Cost", f"${r['cost']:.6f}")
                st.caption("Prices are illustrative; all models shown are free-tier.")
