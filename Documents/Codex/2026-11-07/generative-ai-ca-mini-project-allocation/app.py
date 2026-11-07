"""Streamlit UI for Next-Token Probability Lab."""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

from models import (attention_and_perplexity, bert_predictions, generate_text,
                    is_unsafe, load_bert, load_gpt2, next_token_probabilities,
                    render_text_image)


st.set_page_config(page_title="Next-Token Probability Lab", page_icon="🧪", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: linear-gradient(135deg,#f8fbff 0%,#eef4fb 100%); color: #14213d; }
.block-container { max-width: 1400px; padding-top: 2rem; }
.glass { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.9); border-radius: 22px;
         padding: 1.2rem 1.4rem; box-shadow: 0 12px 36px rgba(31,55,85,.10); margin-bottom: 1rem; }
.glass h2, .glass h3, .glass p, .glass li, .glass label,
.stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown li, .stCaption { color: #14213d !important; }
h1 { color: #14213d !important; letter-spacing: -0.05em; }
h2, h3 { color: #14213d !important; letter-spacing: -0.03em; }
div[data-baseweb="textarea"] textarea { background: #ffffff !important; color: #14213d !important; caret-color: #14213d !important; }
div[data-baseweb="textarea"] textarea::placeholder { color: #6b7890 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading GPT-2…")
def gpt2(): return load_gpt2()


@st.cache_resource(show_spinner="Loading BERT…")
def bert(): return load_bert()


st.title("🧪 Next-Token Probability Lab")
st.caption("Explore uncertainty, attention, sampling, and safety in small language models.")
with st.sidebar:
    st.header("Lab controls")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    top_k = st.slider("Top-k", 1, 100, 10)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    max_new = st.slider("Generated tokens", 1, 120, 40)

prompt = st.text_area("Prompt", "The future of artificial intelligence is", height=100)
try:
    model = gpt2()
except Exception as exc:
    st.error(f"GPT-2 could not be loaded: {exc}")
    st.stop()

st.markdown('<div class="glass"><h2>U1 · Probability Lab</h2>', unsafe_allow_html=True)
if prompt.strip():
    try:
        rows = next_token_probabilities(model, prompt, top_k, temperature)
        chart = go.Figure(go.Bar(x=[r["probability"] for r in rows][::-1],
                                 y=[repr(r["token"]) for r in rows][::-1], orientation="h",
                                 marker_color="#4f8cff", text=[f'{r["probability"]:.2%}' for r in rows][::-1],
                                 textposition="auto"))
        chart.update_layout(height=440, margin=dict(l=10, r=20, t=10, b=20),
                            xaxis_title="Probability", yaxis_title="Candidate token",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#14213d"),
                            xaxis=dict(title_font=dict(color="#14213d"), tickfont=dict(color="#14213d")),
                            yaxis=dict(title_font=dict(color="#14213d"), tickfont=dict(color="#14213d")))
        st.plotly_chart(chart, use_container_width=True)
    except Exception as exc: st.warning(str(exc))
else: st.info("Enter a prompt to inspect next-token probabilities.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass"><h2>U2 · Attention & Perplexity</h2>', unsafe_allow_html=True)
if prompt.strip():
    try:
        tokens, attention, ppl = attention_and_perplexity(model, prompt)
        fig, ax = plt.subplots(figsize=(min(12, max(6, len(tokens) * .45)), 5))
        sns.heatmap(attention.numpy(), xticklabels=tokens, yticklabels=tokens, cmap="mako", ax=ax)
        ax.set_xlabel("Key tokens"); ax.set_ylabel("Query tokens"); plt.xticks(rotation=60, ha="right")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.metric("GPT-2 perplexity", f"{ppl:,.2f}")
        st.markdown("**Baseline context — standard LSTM:** An LSTM language model also reports perplexity as the exponentiated average negative log-likelihood. Lower is better, but values are only directly comparable when the tokenization, corpus, and evaluation split are the same. GPT-2's contextual self-attention often captures longer dependencies than a standard LSTM, though that does not guarantee lower perplexity on every dataset.")
    except Exception as exc: st.warning(str(exc))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass"><h2>U3 · BERT vs GPT-2</h2>', unsafe_allow_html=True)
bert_text = st.text_input("BERT sentence with [MASK]", "The capital of France is [MASK].")
if st.button("Compare predictions"):
    try:
        b = bert(); b_rows = bert_predictions(b, bert_text)
        g_rows = next_token_probabilities(model, bert_text.replace("[MASK]", ""), 5, temperature)
        left, right = st.columns(2)
        with left:
            st.subheader("BERT · masked context")
            for token, prob in b_rows: st.write(f"`{token}` — **{prob:.2%}**")
        with right:
            st.subheader("GPT-2 · left-to-right context")
            for row in g_rows: st.write(f"`{row['token']}` — **{row['probability']:.2%}**")
    except Exception as exc: st.warning(str(exc))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass"><h2>U4 · PyTorch Sampling & Text-to-Image</h2>', unsafe_allow_html=True)
if st.button("Generate with current controls"):
    try:
        generated = generate_text(model, prompt, max_new, temperature, top_k, top_p)
        unsafe, reason = is_unsafe(generated)
        if unsafe: st.warning(f"Output blocked by safety filter ({reason}).")
        else:
            st.code(generated, language="text")
            st.image(render_text_image(generated), use_container_width=True)
    except Exception as exc: st.warning(str(exc))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass"><h2>U5 · Safety & Bias Notes</h2><p>Safety filters are safeguards, not guarantees. Review generated content before sharing or deploying it.</p><p><b>Temperature and factuality:</b> Low temperatures concentrate probability on likely continuations, which can make text more consistent but may repeat confident errors. High temperatures broaden sampling, improving diversity while increasing the chance of unsupported claims, contradictions, and hallucinated facts. Treat fluent output as a hypothesis and verify important claims against reliable sources.</p></div>', unsafe_allow_html=True)
