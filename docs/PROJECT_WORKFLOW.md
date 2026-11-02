# NEXUS LLM — Project Workflow Guide

> Comprehensive operational guide detailing the step-by-step data transformations, tensor flows, intermediate representations, and output visualizations across all five modules in NEXUS LLM.

---

## 📌 Table of Contents

- [1. End-to-End System Flow](#1-end-to-end-system-flow)
- [2. Module 1: Tokenization Studio](#2-module-1-tokenization-studio)
- [3. Module 2: Embedding Explorer](#3-module-2-embedding-explorer)
- [4. Module 3: Transformer Architecture Lab](#4-module-3-transformer-architecture-lab)
- [5. Module 4: Next-Token Prediction Lab](#5-module-4-next-token-prediction-lab)
- [6. Module 5: Autoregressive Generation Lab](#6-module-5-autoregressive-generation-lab)
- [7. Shared Prompt & Execution Lifecycle](#7-shared-prompt--execution-lifecycle)

---

## 1. End-to-End System Flow

```
+---------------------------------------------------------------------------------------------------+
|                                      Shared Workspace Prompt                                      |
|                  "Large language models learn patterns from text and generate..."                 |
+---------------------------------------------------------------------------------------------------+
                                                  |
           +--------------------------------------+---------------------------------------+
           |                                      |                                       |
           v                                      v                                       v
+----------------------+              +-----------------------+              +----------------------+
|  01: Tokenization    |              |    02: Embeddings     |              |   03: Transformer    |
|                      |              |                       |              |                      |
| Raw Text             |              | Token IDs             |              | Layer Norm (ln_1)    |
|   -> Byte-level BPE  |              |   -> wte + wpe Lookup |              |   -> c_attn (Q,K,V)  |
|   -> Subwords        |              |   -> Input Embeddings |              |   -> QK^T / sqrt(d)  |
|   -> Token IDs       |              |   -> Transformer Pass |              |   -> Causal Mask     |
|   -> Attention Mask  |              |   -> Contextual Reps  |              |   -> Softmax Attn    |
|   -> Round-trip text |              |   -> PCA 2D & Norms   |              |   -> Attn x V        |
+----------------------+              +-----------------------+              +----------------------+
                                                  |
           +--------------------------------------+---------------------------------------+
           |                                                                              |
           v                                                                              v
+-------------------------------------+                        +------------------------------------+
|       04: Next-Token Prediction     |                        |        05: Generation Lab          |
|                                     |                        |                                    |
| Final Hidden State [-1]             |                        | Autoregressive Decoding Loop:      |
|   -> LM Head Projection             |                        |   Forward pass -> Logits           |
|   -> Logits [50,257]                |                        |   -> Temperature + Top-K/Top-P     |
|   -> Temperature Scaling (logits/T) |                        |   -> Sample / Greedy pick          |
|   -> Top-K / Top-P Filtering        |                        |   -> Append token to sequence      |
|   -> Softmax -> Probabilities       |                        |   -> KV-Cache update               |
|   -> Shannon Entropy & Ranking      |                        |   -> Repeat until EOS / max_tokens |
+-------------------------------------+                        +------------------------------------+
```

---

## 2. Module 1: Tokenization Studio

### Purpose
Demystify how continuous natural language is mapped into discrete integer symbols digestible by neural networks.

### Execution Steps
1. **Input Validation**: Verifies that the string is non-empty and enforces a maximum input limit (`MAX_INPUT_TOKENS = 128`).
2. **Byte-Pair Encoding (BPE)**: DistilGPT2 uses a byte-level BPE tokenizer with a vocabulary size of $V = 50,257$.
   - Handles arbitrary byte sequences without `<unk>` out-of-vocabulary errors.
   - Prepends `\u0120` (`Ġ`) to indicate preceding whitespace.
3. **ID Mapping & Attention Mask Generation**:
   - `input_ids`: 1D tensor of integer indices $\in [0, 50256]$.
   - `attention_mask`: 1D tensor where `1` indicates actual tokens and `0` indicates padding elements.
4. **Human-Readable Token Formatting**:
   - Special control symbols like `\n` and `\t` are substituted with `↵` and `⇥`.
   - Single space tokens are highlighted as `␠ (space)` to clarify whitespace tokenization.
5. **Round-Trip Reconstruction**:
   - Decodes `input_ids` back to text to confirm lossless representation.

### Visual & Data Outputs
- **Token Flow Chips**: Color-coded badges displaying token position, decoded symbol, and vocabulary ID.
- **Token ID Bar Plot**: Sequence position on the x-axis vs. vocabulary ID on the y-axis.
- **Attention Mask Heatmap**: Binary visual confirmation of unmasked sequence positions.
- **Detailed Token Table**: Tabular breakdown containing Position, Readable Token, Raw Token, Token ID, and Mask Value.

---

## 3. Module 2: Embedding Explorer

### Purpose
Trace how discrete token IDs become continuous 768-dimensional vectors, and observe the transformative effect of self-attention on word representations.

### Mathematical Pipeline
1. **Static Input Embedding**:
   $$\mathbf{e}_i = \mathbf{W}_{\text{te}}[\text{id}_i] + \mathbf{W}_{\text{pe}}[\text{pos}_i]$$
   where $\mathbf{W}_{\text{te}} \in \mathbb{R}^{50257 \times 768}$ is the word token embedding matrix and $\mathbf{W}_{\text{pe}} \in \mathbb{R}^{1024 \times 768}$ is the learned position embedding matrix.
2. **Contextual Representation**:
   $$\mathbf{h}_i^{(L)} = \text{TransformerLayer}_6(\dots \text{TransformerLayer}_1(\mathbf{e}_i))$$
   Obtained from `outputs.hidden_states[-1]`.
3. **Vector Norm Calculation**:
   $$\|\mathbf{v}\|_2 = \sqrt{\sum_{k=1}^{768} v_k^2}$$
   Measures representation magnitude across the token sequence before and after Transformer processing.
4. **PCA Dimensionality Reduction**:
   Computes principal component projection $\mathbb{R}^{N \times 768} \to \mathbb{R}^{N \times 2}$ to visualize the geometric clustering of contextual representations in 2D space.
5. **Sentence Semantic Similarity**:
   Uses the standalone encoder model `sentence-transformers/all-MiniLM-L6-v2`:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### Visual & Data Outputs
- **Side-by-Side Embedding Heatmap**: Directly compares input embeddings vs. final contextual states across the first 32 dimensions.
- **PCA 2D Projection Chart**: Scatter plot with annotated token labels showing semantic proximity.
- **L2 Vector Norm Trajectory**: Line plot comparing magnitude variations across tokens.
- **Sentence Similarity Bar**: Normalized horizontal scale $[-1, +1]$ showing semantic alignment.

---

## 4. Module 3: Transformer Architecture Lab

### Purpose
Inspect the inner mechanics of multi-head causal self-attention, projection matrices ($Q, K, V$), and hidden-state evolution across layers.

### Transformer Block Operations
For selected Layer $l \in [0, 5]$ and Attention Head $h \in [0, 11]$:

1. **Pre-Layer Normalization**:
   $$\mathbf{x}_{\text{norm}} = \text{LayerNorm}(\mathbf{h}^{(l)})$$
2. **Fused Linear Projection**:
   $$\mathbf{W}_{\text{c\_attn}} \mathbf{x}_{\text{norm}} = [\mathbf{Q} \,|\, \mathbf{K} \,|\, \mathbf{V}] \in \mathbb{R}^{N \times 2304}$$
   Split into three matrices each of size $\mathbb{R}^{N \times 768}$, partitioned into 12 heads of dimension $d_k = 64$.
3. **Scaled Dot-Product Attention**:
   $$\mathbf{S} = \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} = \frac{\mathbf{Q}_h \mathbf{K}_h^T}{8}$$
4. **Causal Masking**:
   $$S_{ij} = \begin{cases} S_{ij} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
5. **Attention Weights**:
   $$\mathbf{A}_h = \text{softmax}(\mathbf{S}) \in [0, 1]^{N \times N}$$
6. **Context Aggregation**:
   $$\mathbf{C}_h = \mathbf{A}_h \mathbf{V}_h$$
   Heads are concatenated and projected through $\mathbf{W}_{\text{c\_proj}}$, followed by a residual connection:
   $$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \mathbf{C} \mathbf{W}_{\text{c\_proj}} + \text{MLP}(\text{LayerNorm}(\dots))$$

### Visual & Data Outputs
- **Attention Heatmap**: Interactive matrix where rows represent Query tokens (attending) and columns represent Key tokens (attended to).
- **Depth Representation Norms**: Line plot tracking hidden-state energy growth from input through Block 6.
- **Q / K / V Magnitude Chart**: Triple-bar chart displaying L2 projection norms for each token.
- **Architecture Diagnostics**: Tensor shapes, head counts, and layer parameter statistics.

---

## 5. Module 4: Next-Token Prediction Lab

### Purpose
Examine how the unnormalized output scores (logits) over the 50,257-token vocabulary are transformed into probability distributions through temperature, Top-K, and nucleus (Top-P) sampling.

### Processing Pipeline
1. **Logit Extraction**:
   $$\mathbf{z} = \mathbf{W}_{\text{lm\_head}} \mathbf{h}_N^{(6)} \in \mathbb{R}^{50257}$$
   Extracts scores exclusively from the final sequence position $N$.
2. **Temperature Scaling**:
   $$\tilde{\mathbf{z}} = \frac{\mathbf{z}}{T}$$
   - $T < 1.0$: Sharpens differences; peaks dominate.
   - $T = 1.0$: Unaltered model logits.
   - $T > 1.0$: Flattens distribution toward uniform randomness.
3. **Top-K Truncation**:
   Retains only the $K$ largest logits; all others are set to $-\infty$:
   $$\tilde{z}_i = -\infty \quad \text{if } \text{rank}(\tilde{z}_i) > K$$
4. **Nucleus (Top-P) Filtering**:
   Sorts candidates descendingly, computes cumulative probabilities $C_i = \sum_{j=1}^i p_j$, and removes tokens exceeding cumulative threshold $P$:
   $$\tilde{z}_i = -\infty \quad \text{if } C_{i-1} > P$$
5. **Probability Softmax**:
   $$p_i = \frac{\exp(\tilde{z}_i)}{\sum_j \exp(\tilde{z}_j)}$$
6. **Shannon Entropy**:
   $$H(p) = -\sum_{i: p_i > 0} p_i \log_2(p_i) \quad \text{(bits)}$$
   Quantifies model uncertainty over the next token choice.

### Visual & Data Outputs
- **Top-15 Candidate Bar Plot**: Horizontal bars showing individual token probabilities and readable labels.
- **Cumulative Probability Curve**: Visualizes cumulative distribution mass with Top-P threshold marker.
- **Ranked Candidate Table**: Detailed view with Token Rank, Candidate Token, Vocabulary ID, and Probability (%).
- **Distribution Diagnostics**: Entropy score, active candidate count, and prompt length.

---

## 6. Module 5: Autoregressive Generation Lab

### Purpose
Provide end-to-end visibility into multi-step autoregressive decoding, tracking token generation confidence and KV-cache efficiency.

### Autoregressive Loop Mechanics
At each generation step $t \in [1, \text{max\_new\_tokens}]$:

```
[Prompt Tokens] + [Generated Tokens 1..t-1]
                     |
                     v
           Forward Pass with KV Cache
      (reusing K_{1..t-1}, V_{1..t-1} tensors)
                     |
                     v
               Logits at Step t
                     |
                     v
        Repetition Penalty & N-gram Mask
                     |
                     v
             Sampling / Greedy Choice
                     |
                     v
            Selected Token ID: w_t
                     |
                     v
       Check Termination: w_t == EOS?
            /                \
          Yes                 No
          /                    \
   Halt Generation       Append w_t & Continue
```

1. **Repetition Penalty**:
   $$\tilde{z}_k = \begin{cases} z_k / \theta & \text{if } z_k > 0 \\ z_k \cdot \theta & \text{if } z_k < 0 \end{cases} \quad \text{for tokens } k \text{ previously generated}$$
2. **No-Repeat N-Gram**:
   Checks whether appending a token creates a duplicate sequence of length $N$; sets corresponding logits to $-\infty$.
3. **KV Cache Acceleration (`use_cache=True`)**:
   Instead of recomputing attention over all $t$ tokens, past key and value vectors are stored in memory. Step $t$ only computes projections for the single new token, transforming computational complexity from $\mathcal{O}(t^2)$ to $\mathcal{O}(t)$.
4. **Step Confidence Tracking**:
   Stores softmax probability of each selected token to diagnose output fluency and model confidence.

### Visual & Data Outputs
- **Generated Continuation Box**: Text produced strictly beyond the prompt.
- **Full Text Output**: Combined prompt plus continuation.
- **Token Generation Timeline**: Visual chip flow color-coded with generation step index and confidence percentage.
- **Per-Step Confidence Chart**: Line graph plotting model probability of selected tokens across generation steps.
- **Detailed Step Trace Table**: Tabular record containing Step Number, Generated Token, Token ID, and Probability.

---

## 7. Shared Prompt & Execution Lifecycle

- **Unified Context**: The Shared Workspace Prompt at the top of the interface synchronizes across all five tabs. Modifying the prompt updates the analysis across every module without resetting tab state.
- **Singleton Model Lifecycle**: DistilGPT2 is loaded into host memory once (`get_resources()`). Switching tabs does not trigger model reloading or GPU/CPU memory thrashing.
- **Error Handling**: Input validation checks catch empty strings, out-of-range parameters, and memory constraints gracefully, presenting user-friendly Gradio error alerts.

