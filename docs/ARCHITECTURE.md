# NEXUS LLM — Architecture Reference

> Deep-dive technical reference for the NEXUS LLM system. For the project overview, see [README.md](../README.md).

---

## System Overview

```
                      ┌─────────────────────────────┐
                      │   Gradio Blocks Interface    │
                      │         (app.py)             │
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │     Shared Model Resources   │
                      │   get_resources() → cached   │
                      │  _TOKENIZER · _MODEL · DEVICE│
                      │  _SENTENCE_MODEL (lazy)      │
                      └──┬──────┬──────┬──────┬──────┬──┘
                         │      │      │      │      │
              ┌──────────▼──┐ ┌─▼──┐ ┌▼───┐ ┌▼───┐ ┌▼────────┐
              │Lab 01       │ │Lab │ │Lab │ │Lab │ │Lab 05   │
              │Tokenization │ │ 02 │ │ 03 │ │ 04 │ │Generation│
              └─────────────┘ └────┘ └────┘ └────┘ └─────────┘
```

---

## Model Initialization

### `get_resources()` — Lazy Loading

```python
# Called on first user action in any lab
_TOKENIZER = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
_MODEL     = AutoModelForCausalLM.from_pretrained(
                 "distilbert/distilgpt2",
                 attn_implementation="eager",  # enables attention weights
             )
_MODEL.to(DEVICE).eval()
```

**Why `attn_implementation="eager"`?**

By default, HuggingFace may use memory-efficient Flash Attention variants that do not return attention weight tensors. Using `eager` ensures `output_attentions=True` works in Lab 03.

**Fallback:** If `attn_implementation` raises `TypeError` (older HuggingFace), the model loads without it.

---

## Module 1 — Tokenization

### `run_tokenization(text)` → 7 outputs

```
text (str)
    ↓
tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ↓
input_ids: [batch=1, seq_len] int64 tensor
attention_mask: [batch=1, seq_len] int64 tensor (all 1s for non-padded input)
    ↓
convert_ids_to_tokens(ids) → raw token strings
token_label(token_id) → readable display string
    ↓
Build token chips (HTML) + rows (DataFrame)
    ↓
plt.subplots: bar chart of IDs · imshow: attention mask
    ↓
tokenizer.decode(ids, skip_special_tokens=False) → round-trip text
```

**Token label logic:**
- Space character `" "` → displayed as `"␠ (space)"`
- Newline `"\n"` → displayed as `"↵"`
- Tab `"\t"` → displayed as `"⇥"`
- Empty decode → `"∅"`

---

## Module 2 — Embedding Explorer

### `run_embeddings(text)` → 5 outputs

```
text (str)
    ↓
model(input_ids, output_hidden_states=True, return_dict=True)
    ↓
input_vectors  = model.get_input_embeddings()(input_ids)[0]  # [seq, 768] — static wte+wpe
contextual_np  = outputs.hidden_states[-1][0]                # [seq, 768] — after all 6 blocks
    ↓
Heatmap: imshow of [:24, :32] for both (first 24 tokens, first 32 dims)
PCA: sklearn.decomposition.PCA(n_components=2).fit_transform(contextual_np)
Norms: np.linalg.norm(arr, axis=1) per token for both vectors
```

**Key detail:** `hidden_states` has 7 elements: index 0 = embeddings before any block, indices 1-6 = output after each of the 6 transformer blocks. `hidden_states[-1]` is after all blocks.

### `compare_sentence_similarity(text_a, text_b)` → 2 outputs

```
_SENTENCE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = encode([text_a, text_b], normalize_embeddings=True)  # [2, 384]
cosine_sim = dot(vectors[0], vectors[1])  # valid because normalized
```

Model is loaded lazily on first call only.

---

## Module 3 — Transformer Architecture Lab

### `run_transformer_inspection(text, layer_index, head_index)` → 5 outputs

```
model(input_ids, output_attentions=True, output_hidden_states=True, return_dict=True)
    ↓
attentions: tuple of 6 tensors, each [1, 12, seq, seq]
hidden_states: tuple of 7 tensors, each [1, seq, 768]
    ↓
selected_attn = attentions[layer_index][0, head_index]  # [seq, seq] — the heatmap
layer_norms = [hidden[0].norm(dim=-1).mean() for hidden in hidden_states]
    ↓
QKV extraction:
    base = getattr(model, "transformer", model)  # safety for GPT2Model API
    block = base.h[layer_index]
    layer_input = hidden_states[layer_index]
    normalized = block.ln_1(layer_input)         # GPT-2 applies LN before attention
    combined = block.attn.c_attn(normalized)     # [1, seq, 768*3]
    q, k, v = combined.split(768, dim=2)         # each [1, seq, 768]
    q_norm, k_norm, v_norm = norm(dim=-1)
```

**Heatmap reading guide:**
- Row (y-axis) = query token (the token doing the attending)
- Column (x-axis) = key token (the token being attended to)
- Cell value = attention weight ∈ [0, 1], rows sum to 1.0

---

## Module 4 — Next Token Prediction

### `run_next_token_prediction(text, temperature, top_k, top_p)` → 4 outputs

```
logits = model(input_ids).logits[0, -1, :].float()  # [50257] — last position only
    ↓
apply_sampling_filters(logits / temperature, top_k, top_p)
    │
    ├── Top-K: topk(k).values[-1] is threshold; logits < threshold → -inf
    └── Top-P:
            sorted_logits → sorted_probs → cumsum
            remove = cumsum > top_p (shifted right by 1 to keep first token)
            sorted_indices[remove] → set to -inf
    ↓
probabilities = softmax(filtered_logits)
values, indices = topk(probabilities, 15)  # top 15 shown
entropy = -sum(p * log2(p)) over nonzero probabilities
```

---

## Module 5 — Generation Lab

### `run_generation(prompt, ..., output_scores=True)` → 6 outputs

```
model.generate(
    **inputs,
    max_new_tokens=N,
    do_sample=bool,
    temperature=T,
    top_k=K, top_p=P,
    repetition_penalty=RP,
    no_repeat_ngram_size=NGS,
    output_scores=True,           # ← captures step logits
    return_dict_in_generate=True, # ← returns GenerateDecoderOnlyOutput
    use_cache=True,               # ← KV cache for efficiency
)
    ↓
result.sequences[0]               # full token ID sequence
result.scores                     # tuple of [vocab] logit tensors, one per step
    ↓
continuation_ids = sequence[prompt_len:]
confidence_per_step = softmax(scores[step])[token_id]
```

**KV Cache:** `use_cache=True` tells the model to store key and value tensors from past steps. At step N, only the new token's Q is computed fresh; K and V from all previous steps are retrieved from cache. This makes generation O(N) per step instead of O(N²).

---

## Visual Design System

All charts use a consistent dark theme:

| Variable | Hex | Role |
|---|---|---|
| `BG` | `#0b1020` | Figure/axes background |
| `PANEL` | `#121b2e` | Panel surfaces |
| `GRID` | `#26334a` | Axis grid lines |
| `TEXT` | `#e8edf7` | All text labels |
| `MUTED` | `#9eacc4` | Secondary labels |
| `ACCENT` | `#9b8cff` | Primary bars / lines |
| `CYAN` | `#4dd6d0` | Contextual embedding color |
| `BLUE` | `#63a6ff` | Input embedding color |
| `ORANGE` | `#ffb86b` | Value projections |
| `GREEN` | `#74d99f` | Generation / output |

---

## CSS Architecture (Gradio Custom Styling)

The interface uses CSS custom properties (CSS variables) for theming:

```css
:root {
  --nexus-bg: #090e1b;
  --nexus-panel: #111a2b;
  --nexus-panel-2: #17243a;
  --nexus-violet: #9183ff;
  --nexus-cyan: #4dd6d0;
}
```

Key components:
- `#nexus-hero` — the top banner with radial gradient glows
- `.token-chip` — individual token display cards
- `.generated-chip` — generation-step token cards (cyan accent)
- `.lab-intro` — per-tab introductory banner
- `.workspace-wrap` — shared prompt container

---

## Performance Notes

- Model loads once and is shared across all five tabs via global `_TOKENIZER`, `_MODEL`
- `torch.inference_mode()` (not `no_grad()`) is used — disables both gradients and version tracking, slightly faster
- Maximum input length is capped at 128 tokens to keep response times acceptable on CPU
- The sentence transformer loads lazily on first use of the similarity button
- Matplotlib figures are returned as objects (Gradio renders them as PNG)

