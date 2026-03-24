# Introduction to Preference Alignment with LLM-as-a-Judge (DPO)

This reference implements the **Con-J framework** from:

> - Ye, Z., et al., (2025).
Learning LLM-as-a-Judge for Preference Alignment.
<https://openreview.net/forum?id=HZVIQE1MsJ>

Con-J trains an LLM to act as a **generative judge** using Direct Preference Optimization (DPO), instead of relying on a scalar reward model.

## Why Not a Scalar Reward Model?

Traditional preference alignment (e.g., RLHF-style reward models) trains a model to assign a **scalar score** to each answer.

However, as shown in the following figure:

- Scalar models output only a number → ❌ No explanation
- They are more susceptible to dataset bias
- They may learn superficial patterns (e.g., verbosity)

Con-J instead trains an **LLM-as-a-Judge** that generates:

- A natural language **rationale**
- A binary **preference decision**

This improves:

- Interpretability
- Robustness to bias
- Alignment transparency

![Figure 1: Scalar Reward Model vs Generative Judge](assets/Figure1.png)

## Con-J Training Pipeline

The full framework is illustrated in the following figure.

The process consists of three stages:

### 1️⃣ Judgment Sampling(Repeated and Hint Sampling)

Given a prompt `q` and answers `(a₁, a₂)`,
the pretrained LLM generates multiple **judgments with rationales**.

### 2️⃣ Judgment Filtering

Using ground-truth preference labels, judgments are separated into:

- **Positive judgments** (correct preference)
- **Negative judgments** (incorrect or unclear preference)

These are paired to form **contrastive judgment pairs**.

### 3️⃣ Contrastive Training (DPO)

The LLM is trained using:

- **DPO loss** on contrastive pairs
- A small **SFT loss** on positive judgments

This directly optimizes the model to prefer correct judgments while maintaining generation quality.

![Figure 2: Con-J Pipeline](assets/Figure2.png)

## Contents

- **assets/** – Supporting resources such as figures, prompt templates, and example artifacts.
- **utils/** – Shared helper modules for dataset processing, prompt handling, DPO utilities, and evaluation logic.
- **01_dataset_construction.ipynb** – Constructs and preprocesses the base instruction–response dataset used for preference learning.
- **02_llm_as_judge_inference.ipynb** – Runs LLM-as-a-Judge inference to compare candidate responses and generate preference signals.
- **03_dpo_pair_construction.ipynb** – Converts judge outputs into (prompt, chosen, rejected) pairs compatible with DPO training.
- **04_dpo_training.ipynb** – Fine-tunes the base language model using Direct Preference Optimization (DPO).
- **05_evaluation.ipynb** – Evaluates and compares the base and DPO-aligned models using quantitative and qualitative metrics.

## Dataset Preparation

The filtered `.parquet` files are hosted in a GCP bucket and downloaded separately.

### Download Pre-Filtered Dataset (Recommended)

The filtered dataset used in this implementation is hosted in a GCP bucket.

#### 1) Authenticate with GCP

```bash
gcloud auth login
gcloud auth application-default login
# When prompted, enter the email you used to log into the coder platform.
# A browser window will open for Google sign-in. After signing in, you will receive a code.
# Copy that code back into the terminal to complete authentication.
gcloud config set account YOUR_EMAIL
```

Verify active account:

```bash
gcloud auth list
```

#### 2) Download Dataset

```bash
cd implementations/preference_alignment
gcloud storage cp gs://interp-bootcamp-data/preference-alignment/data.zip .
unzip data.zip
```

The zip extracts a `data/` folder. Move its contents up and remove the wrapper:

```bash
mv data/data_sky . && mv data/data_hh_rlhf . && rm -rf data
```

#### 3) Cleanup temporary files

```bash
rm -rf __MACOSX data.zip .DS_Store
```

After setup, your directory should look like:

```
implementations/preference_alignment/
├── data_sky/
├── data_hh_rlhf/
├── 01_dataset_construction.ipynb
└── ...
```

> **Note:** Use `train_sponsor_filtered.parquet` (for `data_sky`) and `train_singleturn_sponsor_filtered.parquet` (for `data_hh_rlhf`).

Then proceed with `01_dataset_construction.ipynb`.

## Using Your Own Dataset

Participants may use their own preference dataset with this implementation.

Your dataset must satisfy the following structure.

---

### Required Fields

Each example must contain:

- `chosen`   — Preferred response
- `rejected` — Non-preferred response

These represent a pair of candidate responses where `chosen` is preferred over `rejected`.

---

### Supported Formats

#### Format 1 — Structured Chat Format (Sky-style)

```python
{
  "chosen": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
  ],
  "rejected": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
  ]
}
```

In this case, set:

- dataset_format = "sky"

#### Format 2 — Raw Conversation String (HH-style)

```python
{
  "chosen": "Human: ... Assistant: ...",
  "rejected": "Human: ... Assistant: ..."
}
```

In this case, set:

- dataset_format = "hh"

### Adding a New Format

If your dataset follows a different structure:

1. Modify the extract_qa() function in utils/dataset_helpers.py
2. Add a new condition for your custom format
3. Pass the appropriate dataset_format value when calling build_judge_dataset()

### Minimal Requirements Summary

Your dataset must:

Contain preference pairs (```prompt```, ```chosen```, ```rejected```)

Allow extraction of:

1. A single question
2. Two candidate answers

Once formatted properly, the rest of the pipeline (LLM-as-a-Judge → DPO → Evaluation) works unchanged.

## Environment Setup

From the **root of the repository**, install the `preference-alignment` dependency group using `uv`:

```bash
uv sync --active --group preference-alignment
```

The `--active` flag ensures packages are installed into the currently activated virtual environment. After syncing, re-activate the venv to ensure your shell picks up the correct paths:

```bash
deactivate
source .venv/bin/activate
which python3  # should point to .venv/bin/python inside the repo root
```

> **Conflict note:** The `preference-alignment` and `xai-refresher` groups cannot be installed together. Install only one at a time.

<details>
<summary>Installing <code>flash-attn</code> (optional, for faster attention)</summary>

`flash-attn` cannot be built from source on login nodes (no `nvcc`/`CUDA_HOME`). Install a pre-built wheel directly:

```bash
/path/to/.venv/bin/python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

> Use `/path/to/.venv/bin/python -m pip` (explicit venv python) rather than bare `pip` or `python`, to avoid installing into the wrong environment.
>
> The `cu12torch2.8` wheel is compatible with `torch 2.10+cu12x`. Do **not** use `pip install flash-attn==2.7.3 --no-build-isolation` — pip will reject pre-built wheels with local version labels (e.g. `2.7.3+cu12...`) when an exact version like `==2.7.3` is requested.
>
> Skip this step entirely if running on a CPU-only machine.

</details>

## Notes

- Run notebooks sequentially from **01 → 05**.
- Ensure GPU availability before running ```02_inference_runner.ipynb``` and ```04_dpo_training.ipynb```.
- The quality of alignment depends strongly on the judge model and prompt design.
- Our results might have less win rate since we used only 300 samples for training, for better results use larger amount of data.
# Discussion & Conceptual Checkpoints

These questions are intended to help participants reflect on the design choices behind Con-J and DPO.

### 1. Why does Con-J generate a rationale instead of predicting a scalar reward?

**Answer:**
Scalar reward models compress evaluation into a single number, which makes their decisions difficult to interpret and audit. Con-J generates both a rationale and a binary decision, increasing transparency and making it easier to detect bias or flawed reasoning.

### 2. Why is judgment filtering necessary before DPO training?

**Answer:**
LLM-generated judgments can be inconsistent or biased. Filtering ensures that only correct (positive) judgments are contrasted against incorrect (negative) ones. Without filtering, DPO training could reinforce incorrect reasoning patterns.

### 3. What advantage does DPO have over RLHF?

**Answer:**
DPO directly optimizes preference differences without requiring reinforcement learning or PPO-style optimization. It avoids training a separate reward model and is simpler, more stable, and easier to reproduce.

### 4. How do repeated and hint sampling improve the judge model?

**Answer:**
Repeated sampling increases diversity in reasoning patterns, while hint sampling encourages informative rationales. This produces stronger contrastive pairs for DPO training and improves robustness.

### Open Discussion

What risks might arise if the LLM-as-a-Judge itself becomes biased or overconfident in its reasoning?
