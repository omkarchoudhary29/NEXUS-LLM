
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


# ============================================================
# 5. EXTRACT LOGITS
# ============================================================

def get_next_token_logits(model, inputs):
    """Extract full logits and final-position logits."""

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    next_token_logits = logits[:, -1, :]

    return logits, next_token_logits


# ============================================================
# 6. SOFTMAX PROBABILITIES
# ============================================================

def logits_to_probabilities(logits, temperature=1.0):
    """Convert logits into probabilities using Softmax."""

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    probabilities = F.softmax(
        logits / temperature,
        dim=-1
    )

    return probabilities


# ============================================================
# 7. TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    probabilities,
    tokenizer,
    top_n=10
):
    """Return top predicted tokens in a DataFrame."""

    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    top_n = min(top_n, probabilities.shape[-1])

    top_probs, top_ids = torch.topk(
        probabilities[0],
        k=top_n
    )

    results = []

    for token_id, probability in zip(
        top_ids.tolist(),
        top_probs.tolist()
    ):

        results.append({
            "token_id": token_id,
            "token": repr(tokenizer.decode([token_id])),
            "probability": probability,
            "probability_percent": probability * 100
        })

    return pd.DataFrame(results)


# ============================================================
# 8. GREEDY DECODING
# ============================================================

def greedy_decode(logits, tokenizer):
    """Select the highest-scoring next token."""

    token_id = torch.argmax(
        logits,
        dim=-1
    ).item()

    token_text = tokenizer.decode([token_id])

    return token_id, token_text


# ============================================================
# 9. TOP-K FILTERING
# ============================================================

def apply_top_k(logits, k=50):
    """Keep only the k highest-scoring candidates."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    k = min(k, logits.shape[-1])

    top_values, _ = torch.topk(
        logits,
        k=k,
        dim=-1
    )

    threshold = top_values[..., -1, None]

    filtered_logits = logits.masked_fill(
        logits < threshold,
        float("-inf")
    )

    return filtered_logits


# ============================================================
# 10. TOP-P FILTERING
# ============================================================

def apply_top_p(logits, top_p=0.9):
    """Apply nucleus sampling filtering."""

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1.")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1
    )

    sorted_probs = F.softmax(
        sorted_logits,
        dim=-1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    sorted_remove = cumulative_probs > top_p

    # Retain the token that crosses the threshold.
    sorted_remove[..., 1:] = (
        sorted_remove[..., :-1].clone()
    )

    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(
        logits,
        dtype=torch.bool
    )

    remove_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_remove
    )

    filtered_logits = logits.masked_fill(
        remove_mask,
        float("-inf")
    )

    return filtered_logits


# ============================================================
# 11. SAMPLE NEXT TOKEN
# ============================================================

def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    """Apply decoding filters and sample one token."""

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    filtered_logits = logits / temperature

    if top_k is not None:

        filtered_logits = apply_top_k(
            filtered_logits,
            top_k
        )

    if top_p is not None:

        filtered_logits = apply_top_p(
            filtered_logits,
            top_p
        )

    probabilities = F.softmax(
        filtered_logits,
        dim=-1
    )

    sampled_ids = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return sampled_ids, probabilities


# ============================================================
# 12. COMPLETE NEXT TOKEN PREDICTION
# ============================================================

def predict_next_token(
    text,
    tokenizer,
    model,
    device,
    top_n=10,
    temperature=1.0
):
    """Run the complete next-token prediction pipeline."""

    inputs, tokens = tokenize_text(
        text,
        tokenizer,
        device
    )

    full_logits, next_logits = get_next_token_logits(
        model,
        inputs
    )

    probabilities = logits_to_probabilities(
        next_logits,
        temperature
    )

    greedy_id, greedy_text = greedy_decode(
        next_logits,
        tokenizer
    )

    top_predictions = get_top_predictions(
        probabilities,
        tokenizer,
        top_n
    )

    return {
        "inputs": inputs,
        "tokens": tokens,
        "full_logits": full_logits,
        "next_token_logits": next_logits,
        "probabilities": probabilities,
        "greedy_token_id": greedy_id,
        "greedy_token": greedy_text,
        "top_predictions": top_predictions
    }


# ============================================================
# 13. PROBABILITY VISUALIZATION
# ============================================================

def plot_top_predictions(
    probabilities,
    tokenizer,
    top_n=10,
    title="Next Token Probability Distribution"
):
    """Visualize the top predicted token probabilities."""

    top_n = min(top_n, probabilities.shape[-1])

    top_probs, top_ids = torch.topk(
        probabilities[0],
        k=top_n
    )

    labels = [
        repr(tokenizer.decode([token_id]))
        for token_id in top_ids.tolist()
    ]

    values = (
        top_probs.detach().cpu().numpy() * 100
    )

    plt.figure(figsize=(11, 5))

    plt.bar(labels, values)

    plt.title(title)
    plt.xlabel("Candidate Token")
    plt.ylabel("Probability (%)")

    plt.xticks(
        rotation=40,
        ha="right"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 14. MAIN FUNCTION
# ============================================================

def main():
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("NEXUS LLM - NEXT TOKEN PREDICTION LAB")
    print("=" * 60)

    # Load model
    tokenizer, model, device = load_prediction_model()

    print("\nDevice:", device)
    print("Vocabulary:", model.config.vocab_size)

    # Input
    text = "Artificial intelligence is"

    # Prediction
    result = predict_next_token(
        text,
        tokenizer,
        model,
        device,
        top_n=10,
        temperature=1.0
    )

    print("\nInput:", text)
    print("Tokens:", result["tokens"])

    print(
        "Full logits shape:",
        tuple(result["full_logits"].shape)
    )

    print(
        "Next-token logits shape:",
        tuple(result["next_token_logits"].shape)
    )

    print(
        "Probability sum:",
        result["probabilities"].sum().item()
    )

    print(
        "Greedy prediction:",
        repr(result["greedy_token"])
    )

    print("\nTop predictions:")

    print(
        result["top_predictions"].to_string(index=False)
    )

    # Sampling experiment
    sampled_ids, sampled_probs = sample_next_token(
        result["next_token_logits"],
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

    sampled_id = sampled_ids.item()

    print(
        "\nSampled token:",
        repr(tokenizer.decode([sampled_id]))
    )

    print(
        "Sampled probability sum:",
        sampled_probs.sum().item()
    )

    print("\nModule execution completed.")


# ============================================================
# 15. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
