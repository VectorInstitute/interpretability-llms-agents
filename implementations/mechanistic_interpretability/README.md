# Mechanistic Interpretability Overview

## Introduction

This folder is the **mechanistic interpretability** module for the Interpretability for LLMs and Agents Bootcamp.
Instead of treating models as black boxes, we use *internal activations* to build causal, testable stories about
how they compute.

This module covers two complementary settings:

- **LLMs**: the central mystery is **superposition** (many concepts packed into the same neuron / direction). We use
  **Sparse Autoencoders (SAEs)** to turn dense activations into a sparse, interpretable feature dictionary.
- **VLMs**: the central mystery is **modality fusion** (where and when does visual information become language?).
  We use the **logit lens** and **activation patching** to localize the causal bottleneck for visual influence.

## Prerequisites

Before starting, you should have:

- **Python**: 3.10+
- **Environment**: ability to run Jupyter notebooks (JupyterLab recommended)
- **Background**: basic familiarity with transformers and PyTorch
- **Hardware**: a CUDA-capable GPU is recommended (CPU may be slow)
- **Accounts/Access**: a HuggingFace token may be needed (the LLM notebook uses `huggingface_hub.login()` and the
  notebooks may download model weights on first run)

## Notebooks

The following Jupyter notebooks are provided in this folder (under `src/`):

1. **[LLM SAE Tutorial](src/Mechanistic_Interpretability_LLM_Tutorial.ipynb)** — *From black boxes to sparse features*
   - **What you’ll do**:
     - Load an LLM in **TransformerLens** and pick a hook point (e.g. `blocks.L.hook_mlp_out`).
     - Load a pretrained **SAE** (Gemma Scope / SAELens ecosystem) and run:
       `activations → SAE encoder → sparse features → SAE decoder → reconstruction`.
     - Find interpretable features via **top-activating examples** (what tokens/contexts trigger a feature).
     - Run a **feature steering** demo (amplify / clamp-to-zero a feature direction and observe behavior changes).
   - **Key outputs**:
     - A short list of candidate features with their strongest triggering contexts.
     - A steering demo showing causal behavioral changes from a single feature direction.
     - Simple “dark matter” proxies: **reconstruction error** + **behavior gap** (e.g., next-token logits / KL gap).

2. **[VLM Tutorial](src/Mechanistic_Interpretability_VLM_Tutorial.ipynb)** — *Where does vision become language?*
   - **What you’ll do**:
     - Inspect a VLM as **vision encoder → connector (projector) → language model**.
     - Apply a **logit lens** layer-by-layer to watch visual tokens “turn into words”.
     - Use **activation patching**: corrupt the image, restore one layer at a time, and measure how much clean
       behavior is recovered to find the causal fusion bottleneck.
     - Compare the VLM story to the LLM SAE story and discuss VLM “dark matter” (information not yet linguistic).
   - **Key outputs**:
     - Layer-wise logit-lens signals (e.g., entropy/decodability trends for visual tokens).
     - A restoration curve from patching that highlights where visual information becomes causally important.

## Package Dependencies

This implementation includes a small local package (installable) defined in
[`pyproject.toml`](pyproject.toml) (project name: `mech-interp`).

Key dependencies include:

- **`transformer-lens`**: TransformerLens model + activation caching/hooks (LLM tutorial)
- **`sae-lens`**: loading/working with pretrained SAEs (LLM tutorial)
- **`transformers`, `accelerate`, `huggingface-hub`**: HuggingFace models + downloads (both tutorials)
- **`torch`, `numpy`, `matplotlib`, `tqdm`, `pillow`, `requests`**: core runtime + plotting/utilities

## Resources

Pointers for the main tools and ideas used here:

- **TransformerLens**: `https://github.com/TransformerLensOrg/TransformerLens`
- **SAELens**: `https://github.com/jbloomAus/SAELens`
- **Gemma Scope (pretrained SAEs + Neuronpedia integration)**: `https://deepmind.google/models/gemma/gemma-scope/`
- **Neuronpedia (interactive feature browser)**: `https://neuronpedia.org/`
- **Superposition framing**: Anthropic, “Toy Models of Superposition” (2022) — `https://transformer-circuits.pub/2022/toy_model/index.html`
- **Logit Lens (original)**: `https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens`
- **Activation patching in VLMs**: Neo et al., 2024 — `https://arxiv.org/abs/2401.15947`
- **Logit lens for VLMs (MMNeuron)**: `https://arxiv.org/abs/2406.11193`
- **VLM interpretability survey (ICLR blog, 2025)**: `https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding/`

## Getting Started

### Option A: Install this module locally (recommended)

From the **repo root**:

```bash
cd implementations/mechanistic_interpretability
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```