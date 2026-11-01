"""
NEXUS LLM — Module 2: Embedding Explorer

Reusable embedding utilities for:
- DistilGPT2 token embeddings and contextual token representations.
- DistilGPT2 learned positional embeddings.
- Sentence embeddings with Sentence Transformers.
- Cosine similarity, similarity matrices, PCA and t-SNE.

Install:
    pip install torch transformers sentence-transformers numpy scikit-learn matplotlib

The notebook is the place for experiments. This file contains reusable functions
that can later be called from the NEXUS Gradio application.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_NAME = "distilbert/distilgpt2"
DEFAULT_SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_device() -> str:
    """Return CUDA when available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gpt2(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> Tuple[Any, torch.nn.Module]:
    """
    Load a compatible tokenizer and Transformer model.

    Returns:
        (tokenizer, model)

    GPT-2 tokenizers normally have no separate PAD token. For batch processing,
    EOS is assigned as PAD. Always pass the tokenizer's attention_mask to the
    model so padded positions can be distinguished from real tokens.
    """
    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a PAD token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def tokenize_text(
    text: str,
    tokenizer: Any,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize one text and return PyTorch tensors on the selected device."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    device = device or get_device()
    kwargs = {"return_tensors": "pt"}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})

    encoded = tokenizer(text, **kwargs)
    return {key: value.to(device) for key, value in encoded.items()}


def get_token_embeddings(
    text: str,
    tokenizer: Any,
    model: torch.nn.Module,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieve the initial learned token vectors for the tokens in text.

    These are lookup-table vectors, before Transformer contextual processing.
    Returns tokens, token IDs, input tensors, and a (sequence, hidden) NumPy array.
    """
    device = next(model.parameters()).device
    inputs = tokenize_text(text, tokenizer, device=device, max_length=max_length)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        vectors = model.get_input_embeddings()(input_ids)

    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "inputs": inputs,
        "embeddings": vectors[0].detach().cpu().numpy(),
    }


def get_contextual_embeddings(
    text: str,
    tokenizer: Any,
    model: torch.nn.Module,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return the final Transformer hidden state for each token.

    Shape of embeddings: (sequence_length, hidden_dimension).
    These are contextual token representations, not automatically sentence vectors.
    """
    device = next(model.parameters()).device
    inputs = tokenize_text(text, tokenizer, device=device, max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)

    token_ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "last_hidden_state": outputs.last_hidden_state[0].detach().cpu().numpy(),
    }


def get_embedding_matrix(model: torch.nn.Module) -> np.ndarray:
    """Return the full pretrained token embedding matrix as a NumPy array."""
    return model.get_input_embeddings().weight.detach().cpu().numpy()


def get_position_embedding_matrix(model: torch.nn.Module) -> np.ndarray:
    """
    Return GPT-2's learned absolute position embedding matrix.

    This helper intentionally targets GPT-2-style models exposing `wpe`.
    It raises a clear error for architectures that use a different position scheme.
    """
    position_layer = getattr(model, "wpe", None)
    if position_layer is None:
        raise AttributeError(
            "This model does not expose a GPT-2 `wpe` learned position embedding layer."
        )
    return position_layer.weight.detach().cpu().numpy()


def get_position_embeddings(
    model: torch.nn.Module,
    sequence_length: int,
) -> np.ndarray:
    """Return the learned position vectors for positions 0..sequence_length-1."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    position_layer = getattr(model, "wpe", None)
    if position_layer is None:
        raise AttributeError("This model does not expose a GPT-2 `wpe` layer.")

    max_positions = position_layer.num_embeddings
    if sequence_length > max_positions:
        raise ValueError(
            f"Requested {sequence_length} positions, but model supports {max_positions}."
        )

    device = next(model.parameters()).device
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)

    with torch.no_grad():
        position_vectors = position_layer(position_ids)

    return position_vectors[0].detach().cpu().numpy()


def load_sentence_model(
    model_name: str = DEFAULT_SENTENCE_MODEL_NAME,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """Load a sentence embedding model trained for semantic similarity tasks."""
    return SentenceTransformer(model_name, device=device or get_device())


def encode_sentences(
    sentences: Sequence[str],
    sentence_model: SentenceTransformer,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of sentences into one vector per sentence.

    The default MiniLM checkpoint returns 384-dimensional sentence vectors.
    """
    if not sentences:
        raise ValueError("sentences must contain at least one sentence.")
    if any(not isinstance(sentence, str) or not sentence.strip() for sentence in sentences):
        raise ValueError("Every sentence must be a non-empty string.")

    return sentence_model.encode(
        list(sentences),
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )


def calculate_similarity(
    sentence_a: str,
    sentence_b: str,
    sentence_model: SentenceTransformer,
) -> float:
    """Return cosine similarity between two sentences using sentence embeddings."""
    vectors = encode_sentences([sentence_a, sentence_b], sentence_model, normalize=True)
    return float(np.dot(vectors[0], vectors[1]))


def calculate_similarity_matrix(
    sentences: Sequence[str],
    sentence_model: SentenceTransformer,
) -> np.ndarray:
    """Return an N x N cosine similarity matrix for a sequence of sentences."""
    vectors = encode_sentences(sentences, sentence_model, normalize=True)
    return cosine_similarity(vectors)


def reduce_with_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """Reduce an embedding matrix with PCA. Rows must represent samples."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[0] < n_components or embeddings.shape[1] < n_components:
        raise ValueError("Not enough samples or dimensions for the requested PCA components.")

    return PCA(n_components=n_components).fit_transform(embeddings)


def reduce_with_tsne(
    embeddings: np.ndarray,
    perplexity: float = 5.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce an embedding matrix with t-SNE.

    Perplexity must be lower than the number of samples. For very small datasets,
    the resulting plot is only a demonstration, not a robust semantic analysis.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[0] < 3:
        raise ValueError("t-SNE requires at least 3 samples.")
    if perplexity <= 0 or perplexity >= embeddings.shape[0]:
        raise ValueError("perplexity must be greater than 0 and smaller than sample count.")

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def build_token_embedding_table(
    token_data: Dict[str, Any],
    preview_dimensions: int = 8,
):
    """Create a pandas DataFrame showing token IDs and a short vector preview."""
    import pandas as pd

    vectors = np.asarray(token_data["embeddings"])
    tokens = token_data["tokens"]
    token_ids = token_data["token_ids"]

    if len(tokens) != len(token_ids) or len(tokens) != vectors.shape[0]:
        raise ValueError("Token metadata and embedding rows do not have matching lengths.")

    preview_dimensions = max(1, min(preview_dimensions, vectors.shape[1]))

    return pd.DataFrame({
        "token": tokens,
        "token_id": token_ids,
        "embedding_dimensions": [vectors.shape[1]] * len(tokens),
        "vector_preview": [
            np.round(vector[:preview_dimensions], 4).tolist()
            for vector in vectors
        ],
    })


if __name__ == "__main__":
    # Small smoke test: run `python -m modules.embeddings` from the project root
    # if this file is saved as modules/embeddings.py.
    tokenizer, model = load_gpt2()
    sample = "Embeddings represent tokens as numerical vectors."

    token_result = get_token_embeddings(sample, tokenizer, model)
    contextual_result = get_contextual_embeddings(sample, tokenizer, model)

    print("Tokens:", token_result["tokens"])
    print("Token embedding shape:", token_result["embeddings"].shape)
    print("Contextual embedding shape:", contextual_result["last_hidden_state"].shape)

    sentence_model = load_sentence_model()
    score = calculate_similarity(
        "A dog is playing outside.",
        "A puppy is running outdoors.",
        sentence_model,
    )
    print("Example sentence similarity:", round(score, 4))
