# 🧪 Next-Token Probability Lab

An interactive Streamlit application for exploring token probabilities, attention, perplexity, text generation, and safety in modern language models.

Built using Python, PyTorch, Hugging Face Transformers, Streamlit, Plotly, Matplotlib, Seaborn, and Pillow.

## Features

### U1 — Probability Lab

- Calculates next-token probabilities using GPT-2.
- Interactive controls for:
  - Temperature
  - Top-k
  - Top-p
- Displays the most probable next tokens using a horizontal Plotly bar chart.

### U2 — Attention & Perplexity

- Extracts final-layer self-attention weights from GPT-2.
- Displays an attention heatmap for input tokens.
- Calculates GPT-2 perplexity.
- Explains the comparison between GPT-2 and a standard LSTM language model.

### U3 — BERT vs GPT-2

- Accepts text containing a `[MASK]` token.
- Uses BERT to predict the masked word.
- Compares BERT's top five predictions with GPT-2's next-token predictions.
- Demonstrates the difference between masked and causal language modeling.

### U4 — PyTorch Sampling & Text-to-Image

- Implements GPT-2 text generation manually using PyTorch.
- Uses temperature scaling, top-k filtering, top-p filtering, and `torch.multinomial`.
- Does not use the Hugging Face `pipeline` for core generation.
- Renders generated text onto a dynamically created Pillow image.

### U5 — Safety & Bias

- Includes a lightweight keyword-based safety filter.
- Blocks outputs containing selected violent threats or targeted abuse.
- Discusses how high and low temperatures affect factual accuracy, creativity, and hallucinations.

## Project Structure

```text
next-token-probability-lab/
├── app.py
├── models.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/codedbyadiiii/next-token-probability-lab.git
cd next-token-probability-lab
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

## Example Usage

For the probability and generation sections, try:

```text
The future of artificial intelligence is
```

For the BERT comparison section, use:

```text
The capital of France is [MASK].
```

## Models Used

- `gpt2`
- `bert-base-uncased`

The models are downloaded automatically from Hugging Face during the first run.

## Technology Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Plotly
- Matplotlib
- Seaborn
- Pillow

## Safety Notice

The safety filter is a lightweight educational safeguard and is not a complete content-moderation system. Language-model outputs may be inaccurate, biased, offensive, or unsafe. Important information should always be verified independently.

## Deployment

This application can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the GitHub repository.
4. Select `app.py` as the main file.
5. Deploy the application.

No API keys are required.

## License

This project was created for educational and academic use.
