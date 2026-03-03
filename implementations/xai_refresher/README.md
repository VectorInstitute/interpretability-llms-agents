# XAI Refresher Overview

## Introduction
Welcome to the XAI Refresher of the Interpretability in LLMs and Agents Bootcamp. This folder contains a collection of Jupyter notebooks, a `src/` folder, a `data/` folder, and this README.md file to enhance your understanding of explainable AI (XAI) techniques. Here, we will explore various interpretability methods for machine learning models, including SHAP, LIME, concept grounding, and CLIP-based explainability.

## Prerequisites
Before you dive into the materials, ensure you have the following prerequisites:
- Python 3.12 or higher
- Basic knowledge of machine learning and interpretability techniques
- Familiarity with Jupyter Notebooks and PyTorch
- Installed dependencies listed in the `pyproject.toml` under the `ref1-refresher-interpretability` dependency group

## Notebooks
Here you will find the following Jupyter notebooks:
1. **SHAP** - This notebook covers SHAP (SHapley Additive exPlanations) for explaining neural network predictions using a credit card default dataset.
2. **LIME** - This notebook demonstrates LIME (Local Interpretable Model-agnostic Explanations) for explaining random forest predictions on the Adult Income dataset.
3. **Concept Grounding** - This notebook explores concept grounding and explainability for vision-language models, focusing on token-level hidden-state analysis.
4. **CLIP** - This notebook explains concept-based interpretability for vision-language models using CLIP, including Grad-CAM for spatial attribution.

## Package dependencies
This section describes the packages developed for this specific implementation.
- **src/**: Contains utility scripts for feature extraction, analysis, and model training.
- **data/**: Contains datasets and preprocessed files used in the notebooks. Any other datasets can be added here as needed for the exercises.
- **pyproject.toml**: Dependency management file. Use the `ref1-refresher-interpretability` dependency group for Python 3.12.

## Resources
For further reading and additional studies, consider the following resources:

#### SHAP
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/) - Comprehensive guide to SHAP

#### LIME
- Knab, Patrick, et al. "Which lime should i trust? concepts, challenges, and solutions." World Conference on Explainable Artificial Intelligence. Cham: Springer Nature Switzerland, 2025
- [Which LIME to Trust?](https://patrick-knab.github.io/which-lime-to-trust/) - Interactive website exploring LIME trustworthiness

#### Vision-Language Models & Concept Grounding
- Chen, Hong-You, et al. "Contrastive localized language-image pre-training." arXiv preprint arXiv:2410.02746 (2024) - Paper on contrastive localized language-image pre-training
- Parekh, Jayneel, et al. "A concept-based explainability framework for large multimodal models." Advances in Neural Information Processing Systems 37 (2024): 135783-135818. [arXiv:2406.08074](https://arxiv.org/abs/2406.08074) - Concept-based explainability framework

#### Visual Attribution Methods
- Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE International Conference on Computer Vision. 2017 - Original Grad-CAM paper
- Muhammad, Mohammed Bany, and Mohammed Yeasin. "Eigen-cam: Class activation map using principal components." 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020 - Eigen-CAM for class activation maps

## Getting Started
To get started with the materials in this topic:
1. Install the required dependencies using `uv`:
   ```bash
   uv sync --group ref1-refresher-interpretability
    ```
2. Start with the notebook `shap.ipynb` to understand SHAP-based explanations.
3. Move to `lime.ipynb` to explore LIME for tabular data.
4. Continue with concept_grounding.ipynb to learn about concept-based analysis for vision-language models.
5. Finally, explore `clip.ipynb` to understand CLIP-based interpretability and Grad-CAM visualizations.
6. Use the `src/` folder for additional scripts and the `data/` folder for datasets.