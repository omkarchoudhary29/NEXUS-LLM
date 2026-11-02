
"""NEXUS LLM — Language Intelligence & Transformer Analysis Workbench

Run from the project root:
    python app.py

The workspace prompt is shared by all five labs and remains editable while
switching tabs. Models are loaded once and reused for the active app process.
"""

from __future__ import annotations

import html
import math
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CONFIGURATION AND SESSION-WIDE RESOURCES
# =============================================================================

MODEL_NAME = "distilbert/distilgpt2"
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_INPUT_TOKENS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TOKENIZER = None
_MODEL = None
_SENTENCE_MODEL = None

BG = "#0b1020"
PANEL = "#121b2e"
GRID = "#26334a"
TEXT = "#e8edf7"
MUTED = "#9eacc4"
ACCENT = "#9b8cff"
CYAN = "#4dd6d0"
BLUE = "#63a6ff"
ORANGE = "#ffb86b"
PINK = "#f18bb5"
GREEN = "#74d99f"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": GRID,
    "font.size": 9,
    "axes.titleweight": "semibold",
    "savefig.facecolor": BG,
})


def get_resources():
    """Load DistilGPT2 once; all tabs reuse this same model/tokenizer."""
    global _TOKENIZER, _MODEL

    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

        if _TOKENIZER.pad_token_id is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

        try:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                attn_implementation="eager",
            )
        except (TypeError, ValueError):
            _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        _MODEL.to(DEVICE)
        _MODEL.eval()

    return _TOKENIZER, _MODEL, DEVICE


def load_model_status():
    tokenizer, model, device = get_resources()

    layers = getattr(model.config, "n_layer", 6)
    heads = getattr(model.config, "n_head", 12)
    hidden = getattr(model.config, "n_embd", 768)

    return (
        "<div class='status-good'>● MODEL READY</div>"
        f"<div class='status-detail'>{MODEL_NAME} &nbsp;·&nbsp; {device} &nbsp;·&nbsp; "
        f"{layers} layers &nbsp;·&nbsp; {heads} heads &nbsp;·&nbsp; "
        f"{hidden} hidden dimensions</div>"
    )


def prepare_text(text: str, tokenizer=None, max_length: int = MAX_INPUT_TOKENS):

    if not isinstance(text, str) or not text.strip():
        raise gr.Error(
            "Enter text in the shared Workspace Prompt at the top of the dashboard."
        )

    if tokenizer is None:
        tokenizer, _, _ = get_resources()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


def token_label(tokenizer, token_id: int) -> str:
    """Decode one token into a readable display label."""

    piece = tokenizer.decode(
        [int(token_id)],
        clean_up_tokenization_spaces=False,
        skip_special_tokens=False,
    )

    piece = piece.replace("\n", "↵").replace("\t", "⇥")

    if piece == " ":
        return "␠ (space)"

    if not piece:
        return "∅"

    return piece


def safe_plot_message(message: str, width: float = 7, height: float = 2.2):

    fig, ax = plt.subplots(figsize=(width, height))

    ax.axis("off")

    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color=TEXT,
        wrap=True,
    )

    fig.tight_layout()

    return fig


def style_axis(ax, title=None, xlabel=None, ylabel=None):

    if title:
        ax.set_title(title, pad=12)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(alpha=0.28, linewidth=0.7)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(GRID)


# =============================================================================
# MODULE 1 — TOKENIZATION STUDIO
# =============================================================================

def run_tokenization(text: str):

    tokenizer, _, _ = get_resources()

    inputs = prepare_text(text, tokenizer)

    ids = inputs["input_ids"][0].detach().cpu().tolist()

    raw_tokens = tokenizer.convert_ids_to_tokens(ids)

    attention_mask = inputs.get("attention_mask")

    mask_values = (
        attention_mask[0].detach().cpu().tolist()
        if attention_mask is not None
        else [1] * len(ids)
    )

    rows = []
    chips = []

    for position, (raw_token, token_id) in enumerate(
        zip(raw_tokens, ids)
    ):

        readable = token_label(tokenizer, token_id)

        rows.append({
            "Position": position,
            "Readable token": readable,
            "Raw tokenizer token": raw_token,
            "Token ID": int(token_id),
            "Mask": int(mask_values[position]),
        })

        chips.append(
            "<span class='token-chip'>"
            f"<small>POS {position}</small>"
            f"<strong>{html.escape(readable)}</strong>"
            f"<small>ID {token_id}</small>"
            "</span>"
        )

    token_html = (
        "<div class='viz-label'>"
        "TEXT → SUBWORD TOKENS → TOKEN IDS"
        "</div>"
        "<div class='token-flow'>"
        + "".join(chips)
        + "</div>"
    )

    # Token IDs by sequence position

    show_n = min(len(ids), 40)

    fig_ids, ax_ids = plt.subplots(figsize=(9, 3.2))

    positions = np.arange(show_n)

    ax_ids.bar(
        positions,
        ids[:show_n],
        color=ACCENT,
        alpha=0.9,
    )

    ax_ids.set_xticks(positions)
    ax_ids.set_xticklabels([str(x) for x in positions])

    style_axis(
        ax_ids,
        "Token ID at Each Sequence Position",
        "Token position",
        "Vocabulary ID",
    )

    fig_ids.tight_layout()

    # Attention mask visualization

    fig_mask, ax_mask = plt.subplots(figsize=(9, 1.8))

    mask_array = np.array(
        mask_values[:show_n],
        dtype=float,
    ).reshape(1, -1)

    ax_mask.imshow(
        mask_array,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    ax_mask.set_yticks([0])
    ax_mask.set_yticklabels(["Attention mask"])

    ax_mask.set_xticks(np.arange(show_n))
    ax_mask.set_xticklabels([
        str(x) for x in range(show_n)
    ])

    ax_mask.set_xlabel(
        "Token position · 1 = real token, 0 = padding"
    )

    ax_mask.set_title("Sequence Mask Visualization")

    fig_mask.tight_layout()

    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=False,
    )

    special_text = "\n".join(
        f"{key}: {value}"
        for key, value in tokenizer.special_tokens_map.items()
    ) or "No special tokens reported."

    info = (
        f"Token count: {len(ids)}\n"
        f"Vocabulary size: {len(tokenizer):,}\n"
        f"Input limit used by dashboard: {MAX_INPUT_TOKENS} tokens\n"
        "GPT-2 byte-level BPE may split one word into multiple subword tokens."
    )

    table = pd.DataFrame(rows)

    return (
        token_html,
        table,
        fig_ids,
        fig_mask,
        decoded,
        info,
        special_text,
    )


# =============================================================================
# MODULE 2 — EMBEDDING EXPLORER
# =============================================================================

def run_embeddings(text: str):

    tokenizer, model, _ = get_resources()

    inputs = prepare_text(text, tokenizer)

    with torch.inference_mode():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        input_vectors = model.get_input_embeddings()(
            inputs["input_ids"]
        )[0]

        contextual_vectors = outputs.hidden_states[-1][0]

    ids = inputs["input_ids"][0].detach().cpu().tolist()

    labels = [
        token_label(tokenizer, token_id)
        for token_id in ids
    ]

    input_np = input_vectors.detach().cpu().numpy()

    contextual_np = (
        contextual_vectors.detach().cpu().numpy()
    )

    n_tokens, hidden_size = contextual_np.shape

    rows = []

    for i, label in enumerate(labels):

        rows.append({
            "Position": i,
            "Token": label,
            "Input vector norm": float(
                np.linalg.norm(input_np[i])
            ),
            "Context vector norm": float(
                np.linalg.norm(contextual_np[i])
            ),
            **{
                f"Context dim {j+1}": float(
                    contextual_np[i, j]
                )
                for j in range(min(6, hidden_size))
            },
        })

    table = pd.DataFrame(rows)

    # Input versus contextual embedding heatmap

    dims = min(32, hidden_size)
    tokens_to_show = min(24, n_tokens)

    fig_heat, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.2),
        sharex=True,
    )

    im0 = axes[0].imshow(
        input_np[:tokens_to_show, :dims],
        aspect="auto",
        cmap="coolwarm",
    )

    axes[0].set_title("Input Token Embeddings")
    axes[0].set_ylabel("Token")

    axes[0].set_yticks(range(tokens_to_show))
    axes[0].set_yticklabels(
        labels[:tokens_to_show],
        fontsize=7,
    )

    fig_heat.colorbar(
        im0,
        ax=axes[0],
        fraction=0.018,
        pad=0.01,
    )

    im1 = axes[1].imshow(
        contextual_np[:tokens_to_show, :dims],
        aspect="auto",
        cmap="coolwarm",
    )

    axes[1].set_title(
        "Contextual Representations After Transformer Layers"
    )

    axes[1].set_ylabel("Token")
    axes[1].set_xlabel(
        "Embedding dimension (first 32 shown)"
    )

    axes[1].set_yticks(range(tokens_to_show))
    axes[1].set_yticklabels(
        labels[:tokens_to_show],
        fontsize=7,
    )

    fig_heat.colorbar(
        im1,
        ax=axes[1],
        fraction=0.018,
        pad=0.01,
    )

    fig_heat.tight_layout()

    # PCA projection

    if n_tokens >= 2 and hidden_size >= 2:

        from sklearn.decomposition import PCA

        points = PCA(
            n_components=2
        ).fit_transform(contextual_np)

        fig_pca, ax_pca = plt.subplots(
            figsize=(9, 4.2)
        )

        ax_pca.scatter(
            points[:, 0],
            points[:, 1],
            s=70,
            c=CYAN,
            edgecolors=TEXT,
            linewidths=0.5,
        )

        for i, label in enumerate(labels[:40]):

            ax_pca.annotate(
                label[:18],
                (points[i, 0], points[i, 1]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

        style_axis(
            ax_pca,
            "Contextual Token Space · PCA Projection",
            "Principal component 1",
            "Principal component 2",
        )

        fig_pca.tight_layout()

    else:

        fig_pca = safe_plot_message(
            "PCA requires at least two tokens."
        )

    # Input and contextual vector magnitudes

    input_norms = np.linalg.norm(input_np, axis=1)

    contextual_norms = np.linalg.norm(
        contextual_np,
        axis=1,
    )

    fig_norm, ax_norm = plt.subplots(
        figsize=(9, 3.5)
    )

    x = np.arange(n_tokens)

    ax_norm.plot(
        x,
        input_norms,
        marker="o",
        label="Input embedding",
        color=BLUE,
    )

    ax_norm.plot(
        x,
        contextual_norms,
        marker="o",
        label="Contextual representation",
        color=ORANGE,
    )

    ax_norm.set_xticks(x)

    ax_norm.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=7,
    )

    style_axis(
        ax_norm,
        "Vector Magnitude Through the Model",
        "Token",
        "L2 vector norm",
    )

    ax_norm.legend(
        facecolor=PANEL,
        edgecolor=GRID,
    )

    fig_norm.tight_layout()

    info = (
        f"Input embedding tensor: {tuple(input_vectors.shape)}\n"
        f"Contextual representation tensor: {tuple(contextual_vectors.shape)}\n"
        f"Hidden dimensions: {hidden_size}\n"
        "The heatmap compares the first 32 dimensions. PCA is a 2D projection "
        "of contextual vectors, not the original embedding space."
    )

    return table, fig_heat, fig_pca, fig_norm, info


def compare_sentence_similarity(text_a: str, text_b: str):

    global _SENTENCE_MODEL

    if not text_a.strip() or not text_b.strip():
        raise gr.Error("Enter both sentences to compare.")

    try:

        if _SENTENCE_MODEL is None:

            from sentence_transformers import SentenceTransformer

            _SENTENCE_MODEL = SentenceTransformer(
                SENTENCE_MODEL_NAME,
                device=str(DEVICE),
            )

        vectors = _SENTENCE_MODEL.encode(
            [text_a, text_b],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarity = float(
            np.dot(vectors[0], vectors[1])
        )

    except ImportError:

        raise gr.Error(
            "Install sentence-transformers: "
            "pip install sentence-transformers"
        )

    except Exception as exc:

        raise gr.Error(
            f"Sentence similarity failed: {exc}"
        )

    fig, ax = plt.subplots(figsize=(7, 2.8))

    ax.barh(
        ["Cosine similarity"],
        [similarity],
        color=CYAN,
        height=0.45,
    )

    ax.set_xlim(-1, 1)
    ax.axvline(0, color=MUTED, linewidth=0.8)

    style_axis(
        ax,
        "Sentence Semantic Similarity",
        "Cosine similarity",
        None,
    )

    ax.text(
        similarity,
        0,
        f"  {similarity:.3f}",
        va="center",
        color=TEXT,
    )

    fig.tight_layout()

    detail = (
        f"Cosine similarity: {similarity:.4f}\n"
        "This uses all-MiniLM-L6-v2, a dedicated sentence embedding model. "
        "It is separate from DistilGPT2's token embeddings."
    )

    return fig, detail


# =============================================================================
# MODULE 3 — TRANSFORMER ARCHITECTURE LAB
# =============================================================================

def run_transformer_inspection(
    text: str,
    layer_index: int,
    head_index: int,
):

    tokenizer, model, _ = get_resources()

    inputs = prepare_text(text, tokenizer)

    with torch.inference_mode():

        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    attentions = outputs.attentions

    if attentions is None:
        raise gr.Error(
            "Attention weights were not returned. Ensure eager attention is enabled."
        )

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    layer_index = int(layer_index)
    head_index = int(head_index)

    if not 0 <= layer_index < num_layers:
        raise gr.Error(
            f"Layer must be between 0 and {num_layers - 1}."
        )

    if not 0 <= head_index < num_heads:
        raise gr.Error(
            f"Head must be between 0 and {num_heads - 1}."
        )

    ids = inputs["input_ids"][0].detach().cpu().tolist()

    labels = [
        token_label(tokenizer, token_id)
        for token_id in ids
    ]

    attention = (
        attentions[layer_index][0, head_index]
        .detach()
        .cpu()
        .numpy()
    )

    # Attention head heatmap

    fig_attn, ax = plt.subplots(figsize=(8.5, 6.2))

    im = ax.imshow(
        attention,
        cmap="magma",
        aspect="auto",
        vmin=0,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
        fontsize=8,
    )

    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel(
        "Key token · information being attended to"
    )

    ax.set_ylabel(
        "Query token · token doing the attending"
    )

    ax.set_title(
        f"Attention Map · Layer {layer_index + 1} · Head {head_index + 1}"
    )

    fig_attn.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Attention weight",
    )

    fig_attn.tight_layout()

    # Hidden-state magnitude across the architecture

    layer_norms = []

    for hidden in outputs.hidden_states:

        layer_norms.append(
            float(
                hidden[0]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            )
        )

    fig_layers, ax_layers = plt.subplots(
        figsize=(8.5, 3.5)
    )

    layer_x = np.arange(len(layer_norms))

    ax_layers.plot(
        layer_x,
        layer_norms,
        marker="o",
        linewidth=2.2,
        color=CYAN,
    )

    ax_layers.fill_between(
        layer_x,
        layer_norms,
        alpha=0.12,
        color=CYAN,
    )

    ax_layers.set_xticks(layer_x)

    ax_layers.set_xticklabels(
        ["Input"] + [
            f"Block {i+1}"
            for i in range(num_layers)
        ],
        rotation=25,
    )

    style_axis(
        ax_layers,
        "Representation Magnitude Across Transformer Depth",
        "Model depth",
        "Mean token hidden-state norm",
    )

    fig_layers.tight_layout()

    # Q, K, V projection magnitudes

    qkv_fig, ax_qkv = plt.subplots(
        figsize=(8.5, 3.8)
    )

    qkv_note = (
        "Q/K/V projection extraction is not available for this model wrapper."
    )

    try:

        base = getattr(model, "transformer", model)

        block = base.h[layer_index]

        hidden_size = int(
            getattr(model.config, "n_embd", None)
            or getattr(model.config, "hidden_size", 768)
        )

        layer_input = outputs.hidden_states[layer_index]

        # GPT-2 applies layer normalization before QKV projection.
        normalized_layer_input = block.ln_1(layer_input)

        combined = block.attn.c_attn(
            normalized_layer_input
        )

        q, k, v = combined.split(
            hidden_size,
            dim=2,
        )

        q_norm = (
            q[0].norm(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        k_norm = (
            k[0].norm(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        v_norm = (
            v[0].norm(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        xs = np.arange(len(labels))
        width = 0.25

        ax_qkv.bar(
            xs - width,
            q_norm,
            width,
            label="Query",
            color=ACCENT,
        )

        ax_qkv.bar(
            xs,
            k_norm,
            width,
            label="Key",
            color=CYAN,
        )

        ax_qkv.bar(
            xs + width,
            v_norm,
            width,
            label="Value",
            color=ORANGE,
        )

        ax_qkv.set_xticks(xs)

        ax_qkv.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
            fontsize=7,
        )

        style_axis(
            ax_qkv,
            "Q / K / V Projection Magnitudes by Token",
            "Token",
            "Projection vector norm",
        )

        ax_qkv.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            ncol=3,
        )

        qkv_note = (
            f"Combined QKV projection shape: {tuple(combined.shape)}\n"
            f"Q, K and V each have shape: {tuple(q.shape)}\n"
            f"Attention heads: {num_heads} · "
            f"Head dimension: {hidden_size // num_heads}\n"
            "The chart shows the L2 magnitude of each projection per token."
        )

    except Exception as exc:

        ax_qkv.axis("off")

        ax_qkv.text(
            0.5,
            0.5,
            qkv_note + f"\n{exc}",
            ha="center",
            va="center",
            wrap=True,
        )

    qkv_fig.tight_layout()

    detail = (
        f"Architecture: {MODEL_NAME}\n"
        f"Layers: {num_layers} · Attention heads: {num_heads}\n"
        f"Hidden size: {getattr(model.config, 'n_embd', None) or getattr(model.config, 'hidden_size', 'unknown')}\n"
        f"Selected layer: {layer_index + 1} · Selected head: {head_index + 1}\n"
        f"Attention tensor: {tuple(attentions[layer_index].shape)}\n"
        f"Hidden-state tensors: {len(outputs.hidden_states)} (input + each block)\n\n"
        "Attention rows are query tokens; columns are key tokens."
    )

    return (
        fig_attn,
        fig_layers,
        qkv_fig,
        detail,
        qkv_note,
    )


# =============================================================================
# MODULE 4 — NEXT TOKEN PREDICTION
# =============================================================================

def apply_sampling_filters(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """Apply Top-K then nucleus Top-P filtering to vocabulary logits."""

    filtered = logits.clone()
    vocab_size = filtered.shape[-1]

    if int(top_k) > 0:

        k = min(int(top_k), vocab_size)

        threshold = torch.topk(
            filtered,
            k,
        ).values[-1]

        filtered[filtered < threshold] = -float("inf")

    if float(top_p) < 1.0:

        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
        )

        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        remove = cumulative > float(top_p)

        remove[1:] = remove[:-1].clone()
        remove[0] = False

        filtered[sorted_indices[remove]] = -float("inf")

    return filtered


def run_next_token_prediction(
    text: str,
    temperature: float,
    top_k: int,
    top_p: float,
):

    tokenizer, model, _ = get_resources()

    inputs = prepare_text(text, tokenizer)

    with torch.inference_mode():

        logits = model(
            **inputs,
            return_dict=True,
        ).logits[0, -1, :].float()

    temperature = max(float(temperature), 0.01)

    filtered_logits = apply_sampling_filters(
        logits / temperature,
        int(top_k),
        float(top_p),
    )

    probabilities = torch.softmax(
        filtered_logits,
        dim=-1,
    )

    count = min(
        15,
        probabilities.numel(),
    )

    values, indices = torch.topk(
        probabilities,
        count,
    )

    rows = []
    selected_labels = []
    selected_probs = []

    for rank, (probability, token_id) in enumerate(
        zip(values.tolist(), indices.tolist()),
        start=1,
    ):

        label = token_label(tokenizer, token_id)

        selected_labels.append(label[:22])
        selected_probs.append(float(probability))

        rows.append({
            "Rank": rank,
            "Candidate token": label,
            "Token ID": int(token_id),
            "Probability": float(probability),
            "Probability (%)": round(
                float(probability) * 100,
                3,
            ),
        })

    # Probability bars

    fig_top, ax_top = plt.subplots(
        figsize=(9, 5)
    )

    order = np.arange(count)

    ax_top.barh(
        order[::-1],
        selected_probs[::-1],
        color=ACCENT,
    )

    ax_top.set_yticks(order[::-1])

    ax_top.set_yticklabels(
        selected_labels[::-1]
    )

    style_axis(
        ax_top,
        "Top Next-Token Candidates",
        "Probability",
        "Candidate token",
    )

    fig_top.tight_layout()

    # Cumulative probability curve

    sorted_probs = (
        torch.sort(
            probabilities,
            descending=True,
        )
        .values
        .detach()
        .cpu()
        .numpy()
    )

    cumulative = np.cumsum(sorted_probs)

    show = min(100, len(cumulative))

    fig_cum, ax_cum = plt.subplots(
        figsize=(9, 3.5)
    )

    ax_cum.plot(
        np.arange(1, show + 1),
        cumulative[:show],
        color=CYAN,
        linewidth=2,
    )

    ax_cum.axhline(
        float(top_p),
        color=ORANGE,
        linestyle="--",
        label=f"Top-P threshold = {float(top_p):.2f}",
    )

    style_axis(
        ax_cum,
        "Cumulative Probability Mass",
        "Number of candidate tokens",
        "Cumulative probability",
    )

    ax_cum.set_ylim(0, 1.03)

    ax_cum.legend(
        facecolor=PANEL,
        edgecolor=GRID,
    )

    fig_cum.tight_layout()

    probs_np = probabilities.detach().cpu().numpy()

    nonzero = probs_np[probs_np > 0]

    entropy = float(
        -(nonzero * np.log2(nonzero)).sum()
    )

    info = (
        f"Prompt length: {inputs['input_ids'].shape[1]} tokens\n"
        f"Temperature: {temperature:.2f} · "
        f"Top-K: {int(top_k) or 'Off'} · Top-P: {float(top_p):.2f}\n"
        f"Filtered distribution entropy: {entropy:.3f} bits\n"
        "Temperature rescales logits. Top-K limits candidate count; "
        "Top-P keeps the smallest high-probability set whose cumulative "
        "mass reaches the threshold."
    )

    return (
        pd.DataFrame(rows),
        fig_top,
        fig_cum,
        info,
    )


# =============================================================================
# MODULE 5 — GENERATION LAB
# =============================================================================

def run_generation(
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):

    tokenizer, model, _ = get_resources()

    inputs = prepare_text(prompt, tokenizer)

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    if do_sample:

        if float(temperature) <= 0:
            raise gr.Error(
                "Temperature must be greater than zero when sampling."
            )

        kwargs.update({
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
        })

    with torch.inference_mode():

        result = model.generate(
            **inputs,
            **kwargs,
        )

    sequence = result.sequences[0]

    prompt_len = inputs["input_ids"].shape[1]

    continuation_ids = sequence[prompt_len:]

    continuation = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
    )

    full_text = tokenizer.decode(
        sequence,
        skip_special_tokens=True,
    )

    rows = []
    confidences = []

    generated_ids = continuation_ids.detach().cpu().tolist()

    for step, token_id in enumerate(generated_ids):

        confidence = float("nan")

        if result.scores is not None and step < len(result.scores):

            step_probs = torch.softmax(
                result.scores[step][0].float(),
                dim=-1,
            )

            confidence = float(
                step_probs[int(token_id)].detach().cpu()
            )

        confidences.append(confidence)

        rows.append({
            "Generation step": step + 1,
            "Generated token": token_label(tokenizer, token_id),
            "Token ID": int(token_id),
            "Selected-token probability": confidence,
            "Probability (%)": (
                round(confidence * 100, 2)
                if math.isfinite(confidence)
                else None
            ),
        })

    # Per-step confidence chart

    fig_conf, ax_conf = plt.subplots(
        figsize=(9, 3.8)
    )

    valid = [
        (i + 1, p)
        for i, p in enumerate(confidences)
        if math.isfinite(p)
    ]

    if valid:

        xs, ys = zip(*valid)

        ax_conf.plot(
            xs,
            ys,
            marker="o",
            color=CYAN,
            linewidth=2,
        )

        ax_conf.fill_between(
            xs,
            ys,
            alpha=0.12,
            color=CYAN,
        )

        ax_conf.set_ylim(0, 1.03)

        style_axis(
            ax_conf,
            "Generation Confidence · Probability of the Selected Token",
            "Generation step",
            "Selected-token probability",
        )

    else:

        ax_conf.axis("off")

        ax_conf.text(
            0.5,
            0.5,
            "No generation-step scores were returned.",
            ha="center",
            va="center",
        )

    fig_conf.tight_layout()

    # Generated-token sequence

    if rows:

        timeline_html = (
            "<div class='token-flow generation-flow'>"
            + "".join(
                "<span class='token-chip generated-chip'>"
                f"<small>STEP {row['Generation step']}</small>"
                f"<strong>{html.escape(str(row['Generated token']))}</strong>"
                f"<small>{row['Probability (%)'] if row['Probability (%)'] is not None else 'n/a'}% confidence</small>"
                "</span>"
                for row in rows[:60]
            )
            + "</div>"
        )

    else:

        timeline_html = (
            "<div class='muted-box'>"
            "The model ended generation before producing a continuation."
            "</div>"
        )

    stats = (
        f"Prompt tokens: {prompt_len}\n"
        f"Generated tokens: {len(generated_ids)}\n"
        f"Decoding: {'Sampling' if do_sample else 'Greedy'}\n"
        f"Temperature: {float(temperature):.2f} · "
        f"Top-K: {int(top_k)} · Top-P: {float(top_p):.2f}\n"
        f"Repetition penalty: {float(repetition_penalty):.2f} · "
        f"No-repeat n-gram: {int(no_repeat_ngram_size)}\n"
        "Generation is autoregressive: each selected token is appended "
        "to the context for the next step. KV cache avoids recomputing "
        "earlier keys and values."
    )

    return (
        continuation,
        full_text,
        pd.DataFrame(rows),
        timeline_html,
        fig_conf,
        stats,
    )


# =============================================================================
# USER INTERFACE — POLISHED DARK WORKBENCH
# =============================================================================

CSS = r"""
:root {
  --nexus-bg: #090e1b;
  --nexus-panel: #111a2b;
  --nexus-panel-2: #17243a;
  --nexus-line: rgba(160,176,207,.15);
  --nexus-text: #edf2ff;
  --nexus-muted: #9baac4;
  --nexus-violet: #9183ff;
  --nexus-cyan: #4dd6d0;
}

.gradio-container {
  max-width: 1540px !important;
  margin: 0 auto !important;
  background: var(--nexus-bg) !important;
}

body, .gradio-container {
  color: var(--nexus-text) !important;
}

#nexus-hero {
  position: relative;
  overflow: hidden;
  padding: 27px 32px 24px;
  border: 1px solid var(--nexus-line);
  border-radius: 22px;
  margin: 12px 0 16px;
  background:
    radial-gradient(ellipse at 82% 8%, rgba(145,131,255,.25), transparent 35%),
    radial-gradient(ellipse at 8% 100%, rgba(77,214,208,.10), transparent 38%),
    linear-gradient(120deg, #141e34, #0e1728 62%, #17172f);
  box-shadow: 0 18px 55px rgba(0,0,0,.18);
}

#nexus-hero h1 {
  margin: 0 0 8px;
  font-size: 34px;
  font-weight: 800;
  letter-spacing: .04em;
}

#nexus-hero p {
  margin: 5px 0;
  color: #b9c5dc;
}

#nexus-hero .hero-kicker {
  color: #f1efff;
  font-size: 15px;
  font-weight: 650;
}

#nexus-hero .hero-meta {
  margin-top: 14px;
  color: #9eb0cc;
  font-size: 12px;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.status-good {
  display: inline-block;
  color: #85e3b0;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: .10em;
}

.status-detail {
  color: #aebbd1;
  font-size: 12px;
  margin-top: 4px;
}

.workspace-wrap {
  border: 1px solid rgba(145,131,255,.35);
  background: linear-gradient(135deg, rgba(145,131,255,.10), rgba(77,214,208,.035));
  border-radius: 18px;
  padding: 15px 18px;
  margin-bottom: 16px;
}

.workspace-title {
  font-size: 15px;
  font-weight: 750;
  letter-spacing: .04em;
  color: #e9e7ff;
  margin-bottom: 3px;
}

.workspace-help {
  font-size: 12px;
  color: var(--nexus-muted);
}

.lab-intro {
  border: 1px solid var(--nexus-line);
  border-radius: 15px;
  padding: 15px 18px;
  background: linear-gradient(120deg, rgba(25,38,61,.92), rgba(17,26,43,.94));
  margin: 7px 0 14px;
}

.lab-intro h2 {
  margin: 0 0 5px;
  font-size: 21px;
}

.lab-intro p {
  color: var(--nexus-muted);
  margin: 0;
  font-size: 13px;
}

.section-label {
  color: #b9c6df;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin: 10px 0 6px;
}

.viz-label {
  color: #9eadd0;
  font-size: 11px;
  letter-spacing: .11em;
  font-weight: 800;
  margin: 4px 0 10px;
}

.token-flow {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 14px;
  border: 1px solid var(--nexus-line);
  border-radius: 14px;
  background: rgba(12,19,33,.72);
  min-height: 85px;
}

.token-chip {
  display: inline-flex;
  flex-direction: column;
  gap: 4px;
  min-width: 75px;
  max-width: 190px;
  padding: 9px 11px;
  border: 1px solid rgba(145,131,255,.30);
  border-radius: 10px;
  background: linear-gradient(145deg, rgba(145,131,255,.17), rgba(77,214,208,.07));
}

.token-chip strong {
  color: #f1efff;
  font-size: 13px;
  overflow-wrap: anywhere;
}

.token-chip small {
  color: #9eacc8;
  font-size: 9px;
  letter-spacing: .06em;
}

.generated-chip {
  border-color: rgba(77,214,208,.30);
  background: linear-gradient(145deg, rgba(77,214,208,.14), rgba(145,131,255,.08));
}

.generation-flow {
  max-height: 220px;
  overflow-y: auto;
}

.muted-box {
  border: 1px dashed var(--nexus-line);
  border-radius: 12px;
  padding: 18px;
  color: var(--nexus-muted);
}

.panel-card {
  border: 1px solid var(--nexus-line);
  border-radius: 15px;
  background: var(--nexus-panel);
  padding: 13px;
}

button.primary {
  background: linear-gradient(100deg, #6f63e9, #927fff) !important;
  border: 0 !important;
  box-shadow: 0 5px 18px rgba(120,105,245,.18);
}

button.secondary {
  border: 1px solid var(--nexus-line) !important;
}

.tabs {
  border-bottom: 1px solid var(--nexus-line);
}

.tab-nav button {
  font-weight: 650 !important;
}

footer {
  opacity: .6;
}

@media (max-width: 700px) {
  #nexus-hero {
    padding: 22px 18px;
  }

  #nexus-hero h1 {
    font-size: 27px;
  }
}
"""


def clear_workspace():
    return ""


with gr.Blocks(
    title="NEXUS LLM — Language Intelligence Workbench",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
    css=CSS,
) as demo:

    # Header

    gr.HTML("""
    <div id='nexus-hero'>
      <h1>✦ NEXUS LLM</h1>
      <p class='hero-kicker'>
        Language Intelligence &amp; Transformer Analysis Workbench
      </p>
      <p>
        Explore how large language models tokenize, represent,
        process, predict and generate language.
      </p>
      <div class='hero-meta'>
        INTERACTIVE MODEL WORKBENCH &nbsp; · &nbsp; 5 LABS &nbsp; · &nbsp; DISTILGPT2
      </div>
    </div>
    """)

    # One shared prompt for every module

    with gr.Row(elem_classes=["workspace-wrap"]):

        with gr.Column(scale=5):

            gr.Markdown(
                "<div class='workspace-title'>"
                "◈ SHARED WORKSPACE PROMPT</div>"
                "<div class='workspace-help'>"
                "Enter text once. Every lab below uses this same prompt. "
                "Change it any time and rerun a lab; switching tabs will not clear it."
                "</div>"
            )

            workspace_input = gr.Textbox(
                label="",
                value=(
                    "Large language models learn patterns from text "
                    "and generate language one token at a time."
                ),
                placeholder="Enter or edit your prompt here…",
                lines=3,
                max_lines=6,
                elem_id="workspace-prompt",
            )

        with gr.Column(scale=1, min_width=150):

            gr.Markdown(
                "<div class='section-label'>Model status</div>"
            )

            model_status = gr.HTML(
                "<div class='status-detail'>"
                "Model loads on first analysis."
                "</div>"
            )

            load_button = gr.Button(
                "Load / Check Model",
                variant="secondary",
            )

            clear_button = gr.Button(
                "Clear Prompt",
                variant="secondary",
            )

    load_button.click(
        load_model_status,
        outputs=model_status,
    )

    clear_button.click(
        clear_workspace,
        outputs=workspace_input,
    )

    # Five interactive labs

    with gr.Tabs():

        # ---------------------------------------------------------------------
        # TAB 1 — TOKENIZATION
        # ---------------------------------------------------------------------

        with gr.Tab("01 · Tokenization"):

            gr.HTML("""
            <div class='lab-intro'>
              <h2>01 / Tokenization Studio</h2>
              <p>
                Watch raw text become subword pieces, integer IDs and sequence masks.
              </p>
            </div>
            """)

            token_button = gr.Button(
                "Run Tokenization Pipeline",
                variant="primary",
            )

            token_chips = gr.HTML(
                label="Token sequence"
            )

            with gr.Row():

                token_count_card = gr.Markdown(
                    "**TOKENIZATION FLOW**\n\n"
                    "Text → BPE tokenizer → token IDs → model input"
                )

                decoded_output = gr.Textbox(
                    label="Decoded text (round-trip check)",
                    lines=3,
                    interactive=False,
                )

            with gr.Row():

                token_id_plot = gr.Plot(
                    label="Token ID by sequence position"
                )

                mask_plot = gr.Plot(
                    label="Attention mask visualization"
                )

            token_table = gr.Dataframe(
                label="Token-level inspection",
                interactive=False,
                wrap=True,
            )

            with gr.Row():

                token_info = gr.Textbox(
                    label="Tokenizer metrics",
                    lines=4,
                    interactive=False,
                )

                special_tokens = gr.Textbox(
                    label="Special-token configuration",
                    lines=4,
                    interactive=False,
                )

            token_button.click(
                run_tokenization,
                inputs=workspace_input,
                outputs=[
                    token_chips,
                    token_table,
                    token_id_plot,
                    mask_plot,
                    decoded_output,
                    token_info,
                    special_tokens,
                ],
            )

        # ---------------------------------------------------------------------
        # TAB 2 — EMBEDDINGS
        # ---------------------------------------------------------------------

        with gr.Tab("02 · Embeddings"):

            gr.HTML("""
            <div class='lab-intro'>
              <h2>02 / Embedding Explorer</h2>
              <p>
                See how token IDs become vectors and how Transformer layers
                contextualize those vectors.
              </p>
            </div>
            """)

            embedding_button = gr.Button(
                "Explore Embedding Pipeline",
                variant="primary",
            )

            with gr.Row():

                embedding_heatmap = gr.Plot(
                    label="Input vs contextual embedding heatmaps"
                )

                embedding_pca = gr.Plot(
                    label="Contextual token space · PCA"
                )

            embedding_norms = gr.Plot(
                label="Vector magnitude through the model"
            )

            embedding_table = gr.Dataframe(
                label="Vector inspection · first dimensions and norms",
                interactive=False,
                wrap=True,
            )

            embedding_info = gr.Textbox(
                label="Embedding explanation",
                lines=4,
                interactive=False,
            )

            embedding_button.click(
                run_embeddings,
                inputs=workspace_input,
                outputs=[
                    embedding_table,
                    embedding_heatmap,
                    embedding_pca,
                    embedding_norms,
                    embedding_info,
                ],
            )

            gr.Markdown("""
            ### Sentence meaning comparison

            Uses a separate sentence-embedding model to compare two
            sentences semantically.
            """)

            with gr.Row():

                sentence_a = gr.Textbox(
                    label="Sentence A",
                    value="A dog is playing in the garden.",
                    lines=2,
                )

                sentence_b = gr.Textbox(
                    label="Sentence B",
                    value="An animal is running outdoors.",
                    lines=2,
                )

            similarity_button = gr.Button(
                "Compare Sentence Meaning",
                variant="secondary",
            )

            with gr.Row():

                similarity_plot = gr.Plot(
                    label="Cosine similarity visualization"
                )

                similarity_info = gr.Textbox(
                    label="Similarity result",
                    lines=4,
                    interactive=False,
                )

            similarity_button.click(
                compare_sentence_similarity,
                inputs=[sentence_a, sentence_b],
                outputs=[
                    similarity_plot,
                    similarity_info,
                ],
            )

        # ---------------------------------------------------------------------
        # TAB 3 — TRANSFORMER
        # ---------------------------------------------------------------------

        with gr.Tab("03 · Transformer"):

            gr.HTML("""
            <div class='lab-intro'>
              <h2>03 / Transformer Architecture Lab</h2>
              <p>
                Inspect attention as token-to-token information flow,
                follow representations through layers, and compare Q/K/V projections.
              </p>
            </div>
            """)

            with gr.Row():

                layer_choice = gr.Dropdown(
                    choices=list(range(6)),
                    value=0,
                    label="Transformer layer",
                    scale=1,
                )

                head_choice = gr.Dropdown(
                    choices=list(range(12)),
                    value=0,
                    label="Attention head",
                    scale=1,
                )

                transformer_button = gr.Button(
                    "Inspect Transformer Flow",
                    variant="primary",
                    scale=2,
                )

            attention_plot = gr.Plot(
                label="Selected attention head · token-to-token map"
            )

            with gr.Row():

                layer_plot = gr.Plot(
                    label="Hidden-state magnitude across layers"
                )

                qkv_plot = gr.Plot(
                    label="Query / Key / Value projection magnitudes"
                )

            with gr.Row():

                architecture_info = gr.Textbox(
                    label="Architecture trace",
                    lines=6,
                    interactive=False,
                )

                qkv_info = gr.Textbox(
                    label="Q / K / V explanation",
                    lines=6,
                    interactive=False,
                )

            transformer_button.click(
                run_transformer_inspection,
                inputs=[
                    workspace_input,
                    layer_choice,
                    head_choice,
                ],
                outputs=[
                    attention_plot,
                    layer_plot,
                    qkv_plot,
                    architecture_info,
                    qkv_info,
                ],
            )

        # ---------------------------------------------------------------------
        # TAB 4 — NEXT TOKEN
        # ---------------------------------------------------------------------

        with gr.Tab("04 · Next Token"):

            gr.HTML("""
            <div class='lab-intro'>
              <h2>04 / Next Token Prediction Lab</h2>
              <p>
                Inspect the model's next-token distribution and see how
                temperature, Top-K and Top-P reshape the candidate space.
              </p>
            </div>
            """)

            with gr.Row():

                prediction_temperature = gr.Slider(
                    0.1,
                    2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                )

                prediction_top_k = gr.Slider(
                    0,
                    100,
                    value=50,
                    step=5,
                    label="Top-K · 0 disables",
                )

                prediction_top_p = gr.Slider(
                    0.1,
                    1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P · 1 disables",
                )

            prediction_button = gr.Button(
                "Calculate Next-Token Distribution",
                variant="primary",
            )

            with gr.Row():

                prediction_plot = gr.Plot(
                    label="Top next-token candidates"
                )

                cumulative_plot = gr.Plot(
                    label="Cumulative probability mass · nucleus view"
                )

            prediction_table = gr.Dataframe(
                label="Ranked candidate tokens",
                interactive=False,
                wrap=True,
            )

            prediction_info = gr.Textbox(
                label="Distribution diagnostics",
                lines=4,
                interactive=False,
            )

            prediction_button.click(
                run_next_token_prediction,
                inputs=[
                    workspace_input,
                    prediction_temperature,
                    prediction_top_k,
                    prediction_top_p,
                ],
                outputs=[
                    prediction_table,
                    prediction_plot,
                    cumulative_plot,
                    prediction_info,
                ],
            )

        # ---------------------------------------------------------------------
        # TAB 5 — GENERATION
        # ---------------------------------------------------------------------

        with gr.Tab("05 · Generation"):

            gr.HTML("""
            <div class='lab-intro'>
              <h2>05 / Generation Lab</h2>
              <p>
                Generate a continuation from the shared prompt and inspect
                the confidence assigned to each generated token.
              </p>
            </div>
            """)

            with gr.Row():

                generation_max_tokens = gr.Slider(
                    10,
                    150,
                    value=60,
                    step=10,
                    label="Maximum new tokens",
                )

                generation_temperature = gr.Slider(
                    0.1,
                    1.5,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                )

                generation_repetition = gr.Slider(
                    1.0,
                    2.0,
                    value=1.1,
                    step=0.05,
                    label="Repetition penalty",
                )

            with gr.Row():

                generation_sampling = gr.Checkbox(
                    value=True,
                    label="Enable sampling",
                )

                generation_top_k = gr.Slider(
                    0,
                    100,
                    value=50,
                    step=5,
                    label="Top-K · 0 disables",
                )

                generation_top_p = gr.Slider(
                    0.1,
                    1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P",
                )

                generation_ngram = gr.Slider(
                    0,
                    5,
                    value=0,
                    step=1,
                    label="No-repeat n-gram · 0 disables",
                )

            generation_button = gr.Button(
                "Generate & Trace Tokens",
                variant="primary",
            )

            with gr.Row():

                continuation_output = gr.Textbox(
                    label="Generated continuation",
                    lines=5,
                )

                full_output = gr.Textbox(
                    label="Full prompt + generated text",
                    lines=5,
                )

            gr.Markdown("### Generated token timeline")

            generation_timeline = gr.HTML()

            with gr.Row():

                confidence_plot = gr.Plot(
                    label="Selected-token probability over generation steps"
                )

                generation_table = gr.Dataframe(
                    label="Per-token generation trace",
                    interactive=False,
                    wrap=True,
                )

            generation_info = gr.Textbox(
                label="Generation settings and pipeline",
                lines=5,
                interactive=False,
            )

            generation_button.click(
                run_generation,
                inputs=[
                    workspace_input,
                    generation_max_tokens,
                    generation_sampling,
                    generation_temperature,
                    generation_top_k,
                    generation_top_p,
                    generation_repetition,
                    generation_ngram,
                ],
                outputs=[
                    continuation_output,
                    full_output,
                    generation_table,
                    generation_timeline,
                    confidence_plot,
                    generation_info,
                ],
            )

    gr.Markdown(
        "---\n"
        "**NEXUS LLM** · Interactive LLM exploration workbench · "
        "Python · PyTorch · Hugging Face · Gradio"
    )


if __name__ == "__main__":
    demo.queue().launch()
