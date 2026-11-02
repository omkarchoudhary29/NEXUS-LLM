# Application Screenshots Directory

This directory stores the official application screenshots for the NEXUS LLM workbench documentation.

## Required Screenshots

To complete the README presentation, capture these six screenshots from the running application:

| Filename | Dimensions (Recommended) | Description |
|---|---|---|
| `01_dashboard.png` | ~1400 × 800 px | Overview of the full dashboard including the hero banner, shared prompt, and model status. |
| `02_tokenization.png` | ~1400 × 750 px | Tab 01 after clicking "Run Tokenization Pipeline", showing the token chips, bar plot, and table. |
| `03_embeddings.png` | ~1400 × 800 px | Tab 02 showing the side-by-side embedding heatmaps, PCA scatter plot, and vector norm chart. |
| `04_transformer.png` | ~1400 × 800 px | Tab 03 displaying the selected attention head heatmap, layer depth chart, and Q/K/V projections. |
| `05_next_token.png` | ~1400 × 750 px | Tab 04 displaying the candidate probability bar chart and cumulative distribution curve. |
| `06_generation.png` | ~1400 × 850 px | Tab 05 showing the generated continuation, token timeline chips, and per-step confidence plot. |

## Instructions to Capture

1. Start the app:
   ```bash
   python app.py
   ```
2. Open `http://127.0.0.1:7860` in a full-screen browser window.
3. Use the default prompt:
   ```
   Large language models learn patterns from text and generate language one token at a time.
   ```
4. Run each lab tab sequentially and take a clean screenshot of the tab's results.
5. Save each image with the exact filename listed above into this folder (`assets/screenshots/`).
6. Commit and push the screenshot files.

