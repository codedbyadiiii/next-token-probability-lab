# 🧪 Next-Token Probability Lab

An interactive Streamlit application for exploring how language models predict, attend to, and generate text.

The lab uses **GPT-2**, **BERT**, **PyTorch**, and **Hugging Face Transformers** to make token probabilities, self-attention, perplexity, sampling, and safety considerations easy to inspect in one clean interface.

## Features

### U1 · Probability Lab

- Enter any text prompt and inspect the most likely next tokens.
- Adjust temperature, top-k, and top-p from the sidebar.
- View predictions in an interactive horizontal Plotly bar chart.

### U2 · Attention & Perplexity

- Extract final-layer GPT-2 self-attention weights.
- Visualize token-to-token attention with a heatmap.
- Calculate and display GPT-2 perplexity.
- Explain how GPT-2 perplexity compares with a standard LSTM baseline.

### U3 · BERT vs GPT-2

- Enter a sentence containing a literal `[MASK]` token.
- Compare BERT's top five masked-token predictions with GPT-2's next-token predictions.
- See the difference between bidirectional masked-language modeling and left-to-right causal language modeling.

### U4 · PyTorch Sampling & Text-to-Image

- Generate text using a manually implemented PyTorch loop.
- Apply temperature scaling, top-k filtering, nucleus/top-p filtering, and multinomial sampling.
- Render the generated result onto a dynamically created Pillow image canvas.

### U5 · Safety & Bias

- Apply a lightweight safety filter before displaying generated output.
- Block outputs matching conservative violent-threat or targeted-abuse patterns.
- Explain why higher temperature can increase factual errors and hallucinations.

## Project structure

```text
.
├── app.py              # Streamlit interface, layout, charts, and controls
├── models.py           # Model loading, inference, sampling, safety, and image rendering
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Requirements

- Python 3.9 or newer
- Internet access for the first model download
- Approximately 2–4 GB of free disk space for downloaded model files

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

The first launch downloads `gpt2` and, when the comparison section is used, `bert-base-uncased` from Hugging Face. Later launches reuse the local cache.

## How to use

1. Enter a prompt in the main text box.
2. Adjust temperature, top-k, top-p, and generated-token settings in the sidebar.
3. Explore the U1 probability chart and U2 attention heatmap.
4. Use a sentence such as `The capital of France is [MASK].` in U3.
5. Click **Generate with current controls** to sample text and create the image output.

## Deployment

The app can be deployed to Streamlit Community Cloud, Render, Railway, or another Python hosting service.

For Streamlit Community Cloud:

1. Push `app.py`, `models.py`, `requirements.txt`, and `README.md` to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io/).
3. Select the repository and branch.
4. Set the main file to `app.py`.
5. Deploy.

No API keys or application secrets are required.

## Important notes

- Model downloads require network access during first use.
- CPU inference works, but generation and attention visualization may be slower than on a GPU.
- The safety filter is a lightweight fallback and is not a complete content-moderation system.
- Generated language-model output can be inaccurate, biased, or unsafe. Review important outputs before relying on or sharing them.
- Perplexity values should only be compared when the tokenizer, dataset, and evaluation procedure are equivalent.

## Technology stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Plotly
- Matplotlib and Seaborn
- Pillow

## License

This project is intended for educational and academic use. Add a project-specific license before redistributing it commercially.
