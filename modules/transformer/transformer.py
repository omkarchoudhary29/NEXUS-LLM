
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_MODEL_NAME = "distilbert/distilgpt2"

MODEL_NAME = DEFAULT_MODEL_NAME


# ============================================================
# 2. DEVICE
# ============================================================

def get_device():
    """
    Select GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. LOAD MODEL AND TOKENIZER
# ============================================================

def load_transformer(model_name=DEFAULT_MODEL_NAME):
    """
    Load pretrained GPT-2 model and tokenizer.
    """

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

def get_model_config(model):
    """
    Extract important Transformer configuration details.
    """

    config = model.config

    return {
        "hidden_size": config.hidden_size,
        "num_layers": config.n_layer,
        "num_heads": config.n_head,
        "head_dim": config.hidden_size // config.n_head,
        "intermediate_size": config.n_inner or config.hidden_size * 4,
        "vocab_size": config.vocab_size,
        "max_positions": config.n_positions
    }


# ============================================================
# 5. TOKENIZATION
# ============================================================

def tokenize_text(text, tokenizer, device):
    """
    Convert text into token IDs and attention masks.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    return inputs, tokens


# ============================================================
# 6. MODEL OUTPUTS AND HIDDEN STATES
# ============================================================

def get_model_outputs(model, inputs):
    """
    Extract hidden states and attention weights.
    """

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    return outputs


# ============================================================
# 7. ATTENTION MATRIX
# ============================================================

def get_attention_matrix(outputs, layer_index=0, batch_index=0, head_index=0):
    """
    Extract attention weights for a particular layer and head.

    Shape:
    (sequence_length, sequence_length)
    """

    attentions = outputs.attentions

    attention_matrix = attentions[layer_index][
        batch_index,
        head_index
    ]

    return attention_matrix.detach().cpu().numpy()


# ============================================================
# 8. QUERY, KEY AND VALUE EXTRACTION
# ============================================================

def get_qkv(model, inputs, layer_index=0):
    """
    Extract Query, Key and Value projections.

    GPT-2 uses a combined c_attn projection.
    Its output is split into Q, K and V.

    Important:
    GPT2Model exposes wte, wpe and h directly.
    """

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():

        # Get hidden states from the selected block's input.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[layer_index]

        # GPT-2 applies LayerNorm before self-attention.
        block = model.h[layer_index]

        normalized_states = block.ln_1(hidden_states)

        # Combined projection: Query + Key + Value.
        qkv = block.attn.c_attn(normalized_states)

        # Split combined projection into Q, K and V.
        query, key, value = qkv.split(
            model.config.hidden_size,
            dim=2
        )

    return query, key, value


# ============================================================
# 9. SPLIT ATTENTION HEADS
# ============================================================

def split_attention_heads(model, query, key, value):
    """
    Divide Q, K and V into multiple attention heads.

    Example:
    Hidden size = 768
    Number of heads = 12
    Dimension per head = 64

    Output shape:
    (batch_size, num_heads, sequence_length, head_dim)
    """

    batch_size, sequence_length, hidden_size = query.shape

    num_heads = model.config.n_head
    head_dim = hidden_size // num_heads

    def split_heads(tensor):

        tensor = tensor.view(
            batch_size,
            sequence_length,
            num_heads,
            head_dim
        )

        return tensor.transpose(1, 2)

    query_heads = split_heads(query)
    key_heads = split_heads(key)
    value_heads = split_heads(value)

    return query_heads, key_heads, value_heads


# ============================================================
# 10. MANUAL SCALED DOT-PRODUCT ATTENTION
# ============================================================

def calculate_single_head_attention(
    query,
    key,
    value,
    attention_mask=None,
    causal=True
):
    """
    Calculate scaled dot-product attention.

    Formula:

    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

    Supports tensors shaped:
    (batch_size, num_heads, sequence_length, head_dim)
    """

    head_dim = query.shape[-1]

    # Step 1: Calculate similarity scores.
    scores = torch.matmul(
        query,
        key.transpose(-2, -1)
    )

    # Step 2: Scale scores.
    scores = scores / np.sqrt(head_dim)

    sequence_length = query.shape[-2]

    # Step 3: Apply causal mask.
    if causal:

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=query.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(scores.dtype).min
        )

    # Step 4: Apply padding attention mask, if provided.
    if attention_mask is not None:

        # Mask padded key positions.
        key_mask = attention_mask[:, None, None, :] == 0

        scores = scores.masked_fill(
            key_mask,
            torch.finfo(scores.dtype).min
        )

    # Step 5: Convert scores into probabilities.
    attention_weights = F.softmax(scores, dim=-1)

    # Step 6: Weighted sum of values.
    attention_output = torch.matmul(
        attention_weights,
        value
    )

    return attention_output, attention_weights


# ============================================================
# 11. INSPECT TRANSFORMER BLOCK
# ============================================================

def inspect_transformer_block(model, layer_index=0):
    """
    Inspect the internal components of a GPT-2 Transformer block.
    """

    block = model.h[layer_index]

    return {
        "attention": block.attn,
        "mlp": block.mlp,
        "layer_norm_1": block.ln_1,
        "layer_norm_2": block.ln_2,
        "attention_projection": block.attn.c_attn,
        "attention_output_projection": block.attn.c_proj,
        "mlp_input_projection": block.mlp.c_fc,
        "mlp_output_projection": block.mlp.c_proj
    }


# ============================================================
# 12. TRACE TRANSFORMER BLOCK
# ============================================================

def trace_transformer_block(model, inputs, layer_index=0):
    """
    Manually trace the GPT-2 block operations.

    Flow:
    Input
      |
    LayerNorm 1
      |
    Self Attention
      |
    Residual Connection
      |
    LayerNorm 2
      |
    MLP
      |
    Residual Connection
      |
    Output
    """

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():

        # Get input to the selected block.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[layer_index]

        block = model.h[layer_index]

        # First residual connection.
        residual = hidden_states

        # First LayerNorm.
        normalized_states = block.ln_1(hidden_states)

        # Self-attention.
        attention_output = block.attn(
            normalized_states,
            attention_mask=None,
            use_cache=False,
            output_attentions=False
        )[0]

        # GPT-2 attention output projection is included in block.attn.
        attention_result = residual + attention_output

        # Second residual connection.
        residual = attention_result

        # Second LayerNorm.
        normalized_states = block.ln_2(attention_result)

        # Feed-forward MLP.
        mlp_output = block.mlp(normalized_states)

        # Final residual addition.
        final_output = residual + mlp_output

        # Official block output.
        official_output = block(
            hidden_states,
            attention_mask=None,
            use_cache=False,
            output_attentions=False
        )[0]

        difference = torch.max(
            torch.abs(final_output - official_output)
        ).item()

    return {
        "input": hidden_states,
        "attention_output": attention_output,
        "after_attention_residual": attention_result,
        "mlp_output": mlp_output,
        "manual_output": final_output,
        "official_output": official_output,
        "max_difference": difference
    }


# ============================================================
# 13. HIDDEN STATE CHANGES ACROSS LAYERS
# ============================================================

def get_hidden_state_changes(outputs):
    """
    Calculate the L2 norm of changes between consecutive
    hidden-state representations.
    """

    hidden_states = outputs.hidden_states

    changes = []

    for index in range(1, len(hidden_states)):

        previous = hidden_states[index - 1]
        current = hidden_states[index]

        difference = current - previous

        l2_change = torch.norm(
            difference,
            p=2,
            dim=-1
        ).mean().item()

        changes.append(l2_change)

    return changes


# ============================================================
# 14. TOKEN AND POSITION EMBEDDING TABLES
# ============================================================

def get_embedding_tables(model):
    """
    Extract token and position embedding matrices.

    GPT-2 uses learned absolute positional embeddings.
    """

    token_embedding_matrix = model.wte.weight.detach().cpu()

    position_embedding_matrix = model.wpe.weight.detach().cpu()

    return {
        "token_embeddings": token_embedding_matrix,
        "position_embeddings": position_embedding_matrix,
        "token_embedding_shape": tuple(token_embedding_matrix.shape),
        "position_embedding_shape": tuple(position_embedding_matrix.shape)
    }


# ============================================================
# 15. COMPLETE MODEL INSPECTION
# ============================================================

def inspect_model(model):
    """
    Print the important configuration and architecture details.
    """

    config = get_model_config(model)

    print("\nMODEL CONFIGURATION")

    for key, value in config.items():
        print(f"{key}: {value}")

    print("\nNUMBER OF TRANSFORMER BLOCKS:")
    print(len(model.h))

    print("\nFIRST TRANSFORMER BLOCK:")
    print(model.h[0])

    print("\nTOKEN EMBEDDING:")
    print(model.wte)

    print("\nPOSITION EMBEDDING:")
    print(model.wpe)


# ============================================================
# 16. SMOKE TEST
# ============================================================

def main():
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("NEXUS LLM - TRANSFORMER ARCHITECTURE LAB")
    print("=" * 60)

    # Load model.
    tokenizer, model, device = load_transformer()

    print("\nDevice:", device)

    # Configuration.
    config = get_model_config(model)

    print("\nModel configuration:")
    print(config)

    # Sample input.
    text = "Transformers learn relationships between tokens."

    inputs, tokens = tokenize_text(
        text,
        tokenizer,
        device
    )

    print("\nTokens:", tokens)

    # Model forward pass.
    outputs = get_model_outputs(
        model,
        inputs
    )

    print(
        "\nFinal hidden state:",
        tuple(outputs.last_hidden_state.shape)
    )

    print(
        "Attention layers:",
        len(outputs.attentions)
    )

    # Attention matrix.
    attention_matrix = get_attention_matrix(
        outputs,
        layer_index=0,
        batch_index=0,
        head_index=0
    )

    print(
        "\nAttention matrix shape:",
        attention_matrix.shape
    )

    # Query, Key and Value.
    query, key, value = get_qkv(
        model,
        inputs,
        layer_index=0
    )

    print("\nQKV shapes:")
    print("Query:", tuple(query.shape))
    print("Key:", tuple(key.shape))
    print("Value:", tuple(value.shape))

    # Multi-head attention.
    query_heads, key_heads, value_heads = split_attention_heads(
        model,
        query,
        key,
        value
    )

    print("\nMulti-head shapes:")
    print("Query heads:", tuple(query_heads.shape))
    print("Key heads:", tuple(key_heads.shape))
    print("Value heads:", tuple(value_heads.shape))

    # Manual attention.
    manual_output, manual_weights = calculate_single_head_attention(
        query_heads,
        key_heads,
        value_heads,
        attention_mask=inputs.get("attention_mask"),
        causal=True
    )

    print("\nManual attention output:")
    print(tuple(manual_output.shape))

    print(
        "Manual attention weights:",
        tuple(manual_weights.shape)
    )

    # Transformer block inspection.
    block_info = inspect_transformer_block(
        model,
        layer_index=0
    )

    print("\nTransformer block components:")
    for component in block_info:
        print(component)

    # Transformer block tracing.
    trace = trace_transformer_block(
        model,
        inputs,
        layer_index=0
    )

    print(
        "\nManual vs official block difference:",
        trace["max_difference"]
    )

    # Hidden state changes.
    changes = get_hidden_state_changes(outputs)

    print("\nHidden state changes:")
    for index, change in enumerate(changes, start=1):
        print(f"Layer {index}: {change:.4f}")

    # Embedding tables.
    embedding_tables = get_embedding_tables(model)

    print("\nEmbedding table shapes:")
    print(
        "Token embeddings:",
        embedding_tables["token_embedding_shape"]
    )

    print(
        "Position embeddings:",
        embedding_tables["position_embedding_shape"]
    )

    print("\n" + "=" * 60)
    print("TRANSFORMER MODULE EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
