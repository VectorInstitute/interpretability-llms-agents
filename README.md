# Interpretability for LLMs and Agents Bootcamp

This repository contains reference implementations created by the Vector AI Engineering team
for the **Interpretability for LLMs and Agents Bootcamp** — a hands-on program exploring interpretability,
fairness, alignment, and agentic evaluation of large language and vision-language models.

## About This Bootcamp

The bootcamp covers six core topics spanning the modern AI interpretability and evaluation
landscape. Each implementation is a self-contained reference that demonstrates techniques from
recent research, with fully reproducible notebooks and evaluation pipelines.

## Repository Structure

- **docs/**: Additional documentation and setup guides.
- **implementations/**: One directory per topic, each containing notebooks and a README.
- **pyproject.toml**: Centralizes project settings, build requirements, and dependencies.
- **scripts/**: Utility scripts for environment setup and data preparation.

### Implementations

| # | Topic | Description |
|---|-------|-------------|
| 1 | [XAI Refresher](implementations/xai_refresher/) | Foundations of explainable AI — feature attribution, saliency maps, and model-agnostic explanation methods |
| 2 | [Bias & Fairness Analysis](implementations/bias_fairness_analysis/) | Detecting and mitigating bias in ML models across demographic groups |
| 3 | [Preference Alignment](implementations/preference_alignment/) | LLM alignment with human preferences using DPO framework |
| 4 | [Multimedia RAG + VLM](implementations/multimedia_rag/) | Cross-modal retrieval-augmented generation with ImageBind (audio, video, text) |
| 5 | [Agentic ChartQA Evaluation](implementations/agentic_vqa_eval/) | Multi-agent evaluation harness for chart-based VQA using CrewAI and ChartQAPro |

## Getting Started

1. Clone this repository:

   ```bash
   git clone <repo-url>
   cd interpretability_agent_bootcamp
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install dependencies for the topic you want to work with. All dependency groups are defined
   in the root `pyproject.toml` — install only the group(s) you need:

   | Topic | Group name | Install command |
   |-------|-----------|-----------------|
   | XAI Refresher | `ref1-refresher-interpretability` | `uv sync --group ref1-refresher-interpretability` |
   | Bias & Fairness Analysis | `ref2-transparency-xai-toxicity` | `uv sync --group ref2-transparency-xai-toxicity` |
   | Preference Alignment (DPO) | `ref4-llm-alignment-ethics` | `uv sync --group ref4-llm-alignment-ethics` |
   | Multimedia RAG (retrieval) | `ref5-multimedia-rag-vlm` | `uv sync --group ref5-multimedia-rag-vlm` |
   | Multimedia RAG (QA/VLM) | `ref5-multimedia-rag-vlm-qa` | `uv sync --group ref5-multimedia-rag-vlm-qa` |
   | Agentic ChartQA Eval | `ref6-agentic-xai-eval` | `uv sync --group ref6-agentic-xai-eval` |

   > **CUDA note (ref4 — Preference Alignment):** The group uses `torch==2.6.0` from PyPI
   > (which includes CUDA support on Linux). If you specifically need the CUDA 12.4 build, run:

   > ```bash
   > uv sync --group ref4-llm-alignment-ethics \
   >   --index-url https://download.pytorch.org/whl/cu124
   > ```

4. Launch JupyterLab and open the notebooks in the relevant `implementations/<topic>/` directory:

   ```bash
   uv run jupyter lab
   ```

## License

This project is licensed under the terms of the [LICENSE](LICENSE.md) file in the root directory.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

## Contact

For questions or help navigating this repository, contact Aravind Narayanan at
[aravind.narayanan@vectorinstitute.ai](mailto:aravind.narayanan@vectorinstitute.ai)
