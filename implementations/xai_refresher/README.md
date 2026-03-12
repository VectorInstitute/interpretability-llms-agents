# XAI Refresher Overview

## Introduction

Welcome to the **XAI Refresher** implementation of the Interpretability for LLMs and Agents Bootcamp.
This folder covers foundational and advanced techniques in Explainable AI (XAI), with a focus on
post-hoc explanation methods for both traditional neural networks and modern vision-language models
(VLMs). We explore how to make model decisions interpretable through feature attribution,
segmentation-based perturbations, concept decomposition, and gradient-based visualization.

## Prerequisites

Before diving into the materials, ensure you have the following:

- Python 3.10 or higher
- PyTorch 2.x
- Basic familiarity with neural networks and image classification
- Familiarity with Python and Jupyter notebooks
- A CUDA-capable GPU is recommended for the concept grounding notebook and the perturbation notebooks
- Additional libraries for notebooks 5, 6 & 7: `captum`, `transformers`, `datasets` (installed via the same `uv` dependency group)

## Notebooks

The following Jupyter notebooks are provided in this folder:

1. **[LIME](lime.ipynb)** — Covers the LIME (Local Interpretable Model-agnostic Explanations)
   framework for image, tabular, and text models. Includes LORE (rule-based local explanations
   for tabular data) and DSEG-LIME (SAM-powered data-driven segmentation for richer image
   explanations).

2. **[SHAP](shap.ipynb)** — Introduces SHAP (SHapley Additive exPlanations) with KernelExplainer
   applied to a PyTorch MLP trained on the UCI Credit Card Default dataset. Covers SHAP value
   computation, summary plots, and how to interpret additive feature contributions.

3. **[CLIP Interpretability](clip.ipynb)** — Explores concept-based interpretability for
   vision-language models using CLIP. Covers representation-level analysis, Grad-CAM and
   EigenCAM heatmaps, and how embedding-space geometry relates to model decisions.

4. **[Concept Grounding](concept_grounding.ipynb)** — Demonstrates how to extract and decompose
   hidden-state features from LLaVA (7B) using Symmetric Non-negative Matrix Factorization
   (SNMF). Covers concept dictionary learning, multimodal grounding (text + image), and
   local interpretations per sample on COCO.

5. **[Perturbation & Robustness — Vision](perturbation_robustness_captum_image.ipynb)** —
   Covers perturbation-based attribution for image classifiers using the Captum library.
   Implements Occlusion, Feature Ablation, and Noise Tunnel (SmoothGrad) on a ResNet-18
   model. Evaluates explanation quality with the Infidelity and Sensitivity metrics.
   *Do this notebook before the text version.*

6. **[Perturbation & Robustness — Text + Bias](perturbation_robustness_and_bias_text.ipynb)** —
   Mirrors notebook 5 for transformer-based text classifiers (BERT fine-tuned on SST-2).
   Implements token ablation and gradient attribution, then extends to explanation robustness
   under paraphrase, Counterfactual Fairness Distance (CFD) for bias probing, and a Masked
   Language Model pronoun prediction probe to detect occupational gender stereotypes.
   *Do after the vision perturbation notebook.*

7. **[TCAV — Concept-Level Interpretability](tcav_concept_sensitivity.ipynb)** —
   Implements Testing with Concept Activation Vectors (TCAV) for a BERT sentiment classifier.
   Covers CAV training via logistic regression on hidden-layer activations, directional
   derivative computation, and TCAV score analysis across all 13 transformer layers.
   Uses real SST-2 sentences (loaded from HuggingFace) as concept examples for stable CAVs.
   Includes a profession concept probe that reveals BERT-SST2's spurious association between
   professional-activity sentence structure and positive sentiment — independently corroborating
   the bias findings from notebook 6 via a completely different method.
   *Do after notebook 6.*

### Notebooks 5, 6 & 7: Cross-Notebook Connection

These two notebooks are designed as a pair and cover the same core concepts across two modalities:

| Concept                    | Vision notebook (5)              | Text notebook (6)                     |
| -------------------------- | -------------------------------- | ------------------------------------- |
| Perturbation attribution   | Occlusion (patch masking)        | Token ablation (`[MASK]` replacement) |
| Gradient attribution       | Saliency (input gradients)       | Embedding gradient (L2 norm)          |
| Robustness smoothing       | Noise Tunnel / SmoothGrad        | *(extension)*                         |
| Faithfulness metric        | Infidelity + Sensitivity (Captum)| Explanation distance (L2)             |
| Bias analysis              | —                                | CFD + MLM pronoun probe               |

**Key shared insight across both:** perturbation-based methods are more faithful (causal) but
slower; gradient methods are faster but less stable and more sensitive to surface-level token
or pixel changes.

## Package Dependencies

This implementation includes two local packages under `src/`:

- **`xl-vlms`** (`src/xl-vlms/src/`): A library for analyzing vision-language models through
  concept-based explainability. Provides modules for feature extraction from VLM hidden states,
  SNMF-based concept decomposition, multimodal grounding, model steering, and evaluation metrics
  (CLIPScore, BERTScore, VQA accuracy). Supports LLaVA, IDEFICS2, Qwen-VL, and Molmo.

- **`utils`** (`src/utils/`): XAI utility library with implementations of DSEG-LIME
  (data-driven segmentation LIME with SAM integration), Generalized LIME (`GLIME/`), SHAP
  utilities, and evaluation metrics for explanation quality.

## Resources

For further reading on the methods covered in this module:

- **LIME** — Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any
  Classifier", KDD 2016.
- **SHAP** — Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
  NeurIPS 2017.
- **DSEG-LIME** — Narayanan et al., "DSEG-LIME: Improving Image Explanation by Incorporating
  Feature Importance of Superpixels", 2024.
- **Concept Grounding in VLMs** — Toker et al., "Interpretability of Vision-Language Models
  via Concept Bottlenecks", 2024.
- **Grad-CAM** — Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.
- **Occlusion / Perturbation Attribution** — Zeiler & Fergus, "Visualizing and Understanding
  Convolutional Networks", ECCV 2014.
- **SmoothGrad** — Smilkov et al., "SmoothGrad: removing noise by adding noise", arXiv 2017.
- **Infidelity & Sensitivity** — Yeh et al., "On the (In)fidelity and Sensitivity of
  Explanations", NeurIPS 2019.
- **Explanation Robustness (NLP)** — Atmakuri et al., "Robustness of Explanation Methods for
  NLP Models", arXiv 2022.
- **MLM Bias Probing** — Kurita et al., "Measuring Bias in Contextualized Word
  Representations", ACL Workshop on Gender Bias in NLP 2019.

## Getting Started

1. From the **root of the repository**, create a virtual environment and install the
   `ref1-refresher-interpretability` dependency group using `uv`:

   ```bash
   uv sync --group ref1-refresher-interpretability
   ```

   This creates a `.venv` in the repo root and installs all packages needed for this module
   (PyTorch, SHAP, LIME, Grad-CAM, SAM, etc.).

2. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

3. Start with **[lime.ipynb](lime.ipynb)** for a ground-up introduction to post-hoc explanation
   with LIME and its variants.

4. Proceed to **[shap.ipynb](shap.ipynb)** to explore Shapley-value-based attribution on a
   tabular classification task.

5. Move to **[clip.ipynb](clip.ipynb)** to see how gradient-based and representation-level
   explanations apply to vision-language models.

6. Continue with **[perturbation_robustness_captum_image.ipynb](perturbation_robustness_captum_image.ipynb)**
   to explore perturbation-based attribution and explanation faithfulness metrics for vision
   models using Captum.

7. Follow up with **[perturbation_robustness_and_bias_text.ipynb](perturbation_robustness_and_bias_text.ipynb)**
   to apply the same perturbation framework to NLP, and extend it to robustness evaluation
   and bias probing. Note: this notebook downloads `bert-base-uncased` on first run.

8. Finish with **[concept_grounding.ipynb](concept_grounding.ipynb)** for a deep dive into
   concept decomposition and grounding in LLaVA. Note: this notebook requires a GPU and
   will download model weights on first run.
