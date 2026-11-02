
"""
NEXUS LLM — Module 5: Generation Lab

Features:
1. Load tokenizer and causal language model
2. Tokenize prompts
3. Greedy decoding
4. Sampling and temperature
5. Top-K and Top-P
6. Repetition penalty
7. No-repeat n-gram
8. EOS stopping
9. KV cache
10. Generation strategy comparison
"""

import torch
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

def load_generation_model(
    model_name=DEFAULT_MODEL_NAME,
    device=None
):
    """Load tokenizer and causal language model."""

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT-2 tokenizer has no dedicated PAD token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# 4. TOKENIZE PROMPT
# ============================================================

def tokenize_prompt(prompt, tokenizer, device):
    """Convert prompt into token IDs."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    return inputs


# ============================================================
# 5. TEXT GENERATION
# ============================================================

def generate_text(
    prompt,
    tokenizer,
    model,
    device,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0,
    return_full_text=True
):
    """
    Generate text using a pretrained causal language model.

    do_sample=False:
        Greedy decoding.

    do_sample=True:
        Sampling with optional temperature, Top-K and Top-P.

    return_full_text=False:
        Return only the generated continuation.
    """

    # Validate input
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0.")

    if do_sample and temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative or None.")

    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1.")

    if repetition_penalty <= 0:
        raise ValueError("Repetition penalty must be positive.")

    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size cannot be negative.")

    # Tokenize prompt
    inputs = tokenize_prompt(
        prompt,
        tokenizer,
        device
    )

    # Generation settings
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True
    }

    # Sampling-specific parameters
    if do_sample:

        generation_kwargs["temperature"] = temperature

        if top_k is not None:
            generation_kwargs["top_k"] = top_k

        if top_p is not None:
            generation_kwargs["top_p"] = top_p

    # Generate tokens
    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            **generation_kwargs
        )

    # Return full text or only continuation
    if return_full_text:

        result_ids = output_ids[0]

    else:

        prompt_length = inputs["input_ids"].shape[1]

        result_ids = output_ids[0, prompt_length:]

    generated_text = tokenizer.decode(
        result_ids,
        skip_special_tokens=True
    )

    return generated_text


# ============================================================
# 6. COMPARE GENERATION STRATEGIES
# ============================================================

def compare_generation_strategies(
    prompt,
    tokenizer,
    model,
    device,
    max_new_tokens=50
):
    """Compare different decoding strategies."""

    strategies = {

        "Greedy": {
            "do_sample": False
        },

        "Temperature Sampling": {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": None,
            "top_p": None
        },

        "Top-K": {
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": None
        },

        "Top-P": {
            "do_sample": True,
            "temperature": 0.8,
            "top_k": None,
            "top_p": 0.9
        },

        "Top-K + Top-P": {
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9
        }
    }

    results = {}

    for name, settings in strategies.items():

        generated = generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **settings
        )

        results[name] = generated

    return results


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("NEXUS LLM - GENERATION LAB")
    print("=" * 60)

    # Load model
    tokenizer, model, device = load_generation_model()

    print("\nDevice:", device)
    print("Vocabulary:", model.config.vocab_size)

    print("EOS token ID:", tokenizer.eos_token_id)
    print("PAD token ID:", tokenizer.pad_token_id)

    # Input prompt
    prompt = "The future of artificial intelligence"

    # Generate text
    generated = generate_text(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=0,
        return_full_text=True
    )

    print("\nPrompt:")
    print(prompt)

    print("\nGenerated text:")
    print(generated)

    print("\nModule execution completed.")


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
