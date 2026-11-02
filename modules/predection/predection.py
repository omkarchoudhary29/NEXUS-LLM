
"""
NEXUS LLM — Module 4
Next Token Prediction Lab

Features:
1. Load tokenizer and causal language model
2. Tokenization
3. Logits extraction
4. Softmax probabilities
5. Top token predictions
6. Greedy decoding
7. Temperature
8. Top-K filtering
9. Top-P filtering
10. Token sampling
11. Probability visualization
"""

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_MODEL_NAME = "distilbert/distilgpt2"


# ============================================================
# 2. DEVICE
# ============================================================

def get_device():
    """Select GPU if available, otherwise CPU."""

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# 3. LOAD MODEL AND TOKENIZER
# ============================================================

def load_prediction_model(
    model_name=DEFAULT_MODEL_NAME,
    device=None
):
    """Load tokenizer and causal language model."""

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# 4. TOKENIZATION
# ============================================================

def tokenize_text(text, tokenizer, device):
    """Convert text into token IDs and readable tokens."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    )

    return inputs, tokens

