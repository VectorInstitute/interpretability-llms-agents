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

2. Set up the environment. Each implementation has its own dependencies — refer to the README
   in the relevant `implementations/<topic>/` directory for setup instructions.

3. Begin with the topic of interest. Each directory contains Jupyter notebooks and a README
   describing prerequisites, data, and how to run the experiments.

## License

This project is licensed under the terms of the [LICENSE](LICENSE.md) file in the root directory.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

## Contact

For questions or help navigating this repository, contact Aravind Narayanan at
aravind.narayanan@vectorinstitute.ai.
