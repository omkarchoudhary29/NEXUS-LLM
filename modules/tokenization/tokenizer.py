
from transformers import AutoTokenizer


# ==========================================
# 1. TOKENIZER CONFIGURATION
# ==========================================

DEFAULT_MODEL_NAME = "distilbert/distilgpt2"


def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    """
    Load a pretrained Hugging Face tokenizer.

    For GPT-2-style tokenizers, configure EOS as PAD
    when no padding token exists.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                "Tokenizer has no padding token or EOS token."
            )

    return tokenizer


# ==========================================
# 2. BASIC TOKENIZATION
# ==========================================

def tokenize_text(text, tokenizer):
    """
    Convert text into a list of tokens.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    return tokenizer.tokenize(text)


def encode_text(text, tokenizer, add_special_tokens=False):
    """
    Convert text into numerical token IDs.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    return tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens
    )


def decode_tokens(token_ids, tokenizer, skip_special_tokens=True):
    """
    Convert token IDs back into readable text.
    """
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens
    )


def get_token_count(text, tokenizer):
    """
    Count tokens without adding special tokens.
    """
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# ==========================================
# 3. TOKEN AND ID MAPPING
# ==========================================

def get_token_id(token, tokenizer):
    """
    Convert an individual token into its vocabulary ID.
    """
    return tokenizer.convert_tokens_to_ids(token)


def get_token_from_id(token_id, tokenizer):
    """
    Convert a vocabulary ID into its token.
    """
    return tokenizer.convert_ids_to_tokens(token_id)


def get_token_mapping(text, tokenizer):
    """
    Return tokens and their corresponding IDs.
    """
    tokens = tokenizer.tokenize(text)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return [
        {"token": token, "token_id": token_id}
        for token, token_id in zip(tokens, token_ids)
    ]


# ==========================================
# 4. VOCABULARY INSPECTION
# ==========================================

def get_vocabulary_info(tokenizer):
    """
    Return tokenizer vocabulary information.
    """
    vocabulary = tokenizer.get_vocab()

    return {
        "vocabulary_size": len(vocabulary),
        "model_max_length": tokenizer.model_max_length,
        "special_tokens": tokenizer.special_tokens_map
    }


def get_vocabulary_sample(tokenizer, limit=20):
    """
    Return a sample of vocabulary tokens and IDs.
    """
    vocabulary = tokenizer.get_vocab()

    sorted_vocab = sorted(
        vocabulary.items(),
        key=lambda item: item[1]
    )

    return [
        {"token": token, "token_id": token_id}
        for token, token_id in sorted_vocab[:limit]
    ]


# ==========================================
# 5. SPECIAL TOKEN INSPECTION
# ==========================================

def get_special_tokens(tokenizer):
    """
    Inspect BOS, EOS, PAD, UNK and MASK tokens.
    """
    return {
        "BOS": {
            "token": tokenizer.bos_token,
            "id": tokenizer.bos_token_id
        },
        "EOS": {
            "token": tokenizer.eos_token,
            "id": tokenizer.eos_token_id
        },
        "PAD": {
            "token": tokenizer.pad_token,
            "id": tokenizer.pad_token_id
        },
        "UNK": {
            "token": tokenizer.unk_token,
            "id": tokenizer.unk_token_id
        },
        "MASK": {
            "token": tokenizer.mask_token,
            "id": tokenizer.mask_token_id
        }
    }


# ==========================================
# 6. BATCH PREPARATION
# ==========================================

def prepare_batch(
    texts,
    tokenizer,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
):
    """
    Prepare a batch of text for a Transformer model.

    Returns:
        input_ids
        attention_mask
        other tokenizer-generated fields
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        raise ValueError("At least one text is required.")

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Every batch item must be a string.")

    if truncation and max_length is None:
        raise ValueError(
            "Provide max_length when truncation is enabled."
        )

    return tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors
    )


# ==========================================
# 7. TOKENIZER COMPARISON
# ==========================================

def compare_tokenizers(text, tokenizers):
    """
    Compare tokenization across multiple tokenizers.

    Example:
        {
            "GPT-2": gpt_tokenizer,
            "BERT": bert_tokenizer
        }
    """
    results = {}

    for name, tokenizer in tokenizers.items():

        tokens = tokenizer.tokenize(text)

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        results[name] = {
            "tokens": tokens,
            "token_count": len(token_ids),
            "token_ids": token_ids
        }

    return results


# ==========================================
# 8. TEST THE TOKENIZATION MODULE
# ==========================================

if __name__ == "__main__":

    print("Loading Tokenizer...")

    tokenizer = load_tokenizer()

    text = "I am learning Large Language Models."

    print("\nOriginal Text:")
    print(text)

    print("\nTokens:")
    print(tokenize_text(text, tokenizer))

    print("\nToken IDs:")
    print(encode_text(text, tokenizer))

    print("\nToken Count:")
    print(get_token_count(text, tokenizer))

    print("\nDecoded Text:")
    print(decode_tokens(
        encode_text(text, tokenizer),
        tokenizer
    ))

    print("\nSpecial Tokens:")
    print(get_special_tokens(tokenizer))

    print("\nTokenizer Loaded Successfully!")