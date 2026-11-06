"""Inference and visualization-ready utilities for Next-Token Probability Lab."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


@dataclass
class ModelBundle:
    tokenizer: object
    model: object
    device: torch.device


def load_gpt2() -> ModelBundle:
    """Load GPT-2 once per Streamlit session (the caller should cache this function)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", output_attentions=True)
    model.to(device).eval()
    return ModelBundle(tokenizer, model, device)


def load_bert() -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model.to(device).eval()
    return ModelBundle(tokenizer, model, device)


def _validate_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Please enter a non-empty prompt.")
    return prompt


@torch.inference_mode()
def next_token_probabilities(bundle: ModelBundle, prompt: str, top_k: int = 10,
                             temperature: float = 1.0) -> List[Dict[str, float]]:
    prompt = _validate_prompt(prompt)
    temperature = max(float(temperature), 1e-3)
    encoded = bundle.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024).to(bundle.device)
    logits = bundle.model(**encoded).logits[:, -1, :] / temperature
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, k=min(int(top_k), probabilities.shape[-1]), dim=-1)
    return [{"token": bundle.tokenizer.decode([int(idx)], clean_up_tokenization_spaces=False),
             "probability": float(value)} for value, idx in zip(values[0].cpu(), indices[0].cpu())]


@torch.inference_mode()
def nucleus_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    filtered.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    return filtered


@torch.inference_mode()
def generate_text(bundle: ModelBundle, prompt: str, max_new_tokens: int = 60,
                  temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95) -> str:
    prompt = _validate_prompt(prompt)
    if max_new_tokens < 1:
        return prompt
    input_ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=1024).input_ids.to(bundle.device)
    for _ in range(min(int(max_new_tokens), 256)):
        context = input_ids[:, -1024:]
        logits = bundle.model(input_ids=context).logits[:, -1, :] / max(float(temperature), 1e-3)
        if top_k > 0:
            kth = torch.topk(logits, min(int(top_k), logits.shape[-1]), dim=-1).values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < kth, float("-inf"))
        logits = nucleus_filter(logits, min(max(float(top_p), 0.01), 1.0))
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        if next_id.item() == bundle.tokenizer.eos_token_id:
            break
    return bundle.tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=True)


@torch.inference_mode()
def attention_and_perplexity(bundle: ModelBundle, text: str) -> Tuple[List[str], torch.Tensor, float]:
    text = _validate_prompt(text)
    encoded = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(bundle.device)
    outputs = bundle.model(**encoded, labels=encoded.input_ids, output_attentions=True)
    attention = outputs.attentions[-1][0].mean(dim=0).detach().cpu()
    ids = encoded.input_ids[0].detach().cpu()
    tokens = [bundle.tokenizer.decode([int(i)], clean_up_tokenization_spaces=False) for i in ids]
    return tokens, attention, float(torch.exp(outputs.loss.detach().cpu()).item())


@torch.inference_mode()
def bert_predictions(bundle: ModelBundle, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
    text = _validate_prompt(text)
    mask_id = bundle.tokenizer.mask_token_id
    encoded = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(bundle.device)
    locations = (encoded.input_ids == mask_id).nonzero(as_tuple=False)
    if locations.numel() == 0:
        raise ValueError("BERT input must contain the literal [MASK] token.")
    logits = bundle.model(**encoded).logits[locations[0, 0], locations[0, 1]]
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=top_n)
    return [(bundle.tokenizer.decode([int(i)]).strip(), float(v)) for v, i in zip(values.cpu(), indices.cpu())]


def is_unsafe(text: str) -> Tuple[bool, str]:
    """Dependency-light safety fallback. It is intentionally conservative and non-diagnostic."""
    patterns = {
        "hate or targeted abuse": r"\b(?:kill|murder|exterminate|genocide|racial slur)\b",
        "violent threat": r"\b(?:shoot|bomb|attack)\s+(?:you|them|him|her)\b",
    }
    lowered = text.lower()
    for label, pattern in patterns.items():
        if re.search(pattern, lowered):
            return True, label
    return False, ""


def render_text_image(text: str, width: int = 1200, height: int = 650) -> Image.Image:
    image = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        color = (15 + int(20 * y / height), 23 + int(30 * y / height), 42 + int(45 * y / height))
        draw.line((0, y, width, y), fill=color)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
        small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    draw.rounded_rectangle((55, 55, width - 55, height - 55), radius=28,
                           outline=(120, 190, 255), width=2, fill=(25, 38, 66))
    draw.text((90, 85), "NEXT-TOKEN PROBABILITY LAB", font=small, fill=(150, 210, 255))
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] > width - 220:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, current in enumerate(lines[:10]):
        draw.text((90, 170 + index * 52), current, font=font, fill=(240, 247, 255))
    return image
