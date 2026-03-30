# Multimedia RAG + VLM Reference Implementation

Multimedia Retrieval-Augmented Generation (RAG) pipeline for video understanding using:

* **ImageBind** for cross-modal embedding (audio, video, text)
* **PyTorchVideo** for video feature handling
* **Qwen Omni** stack for multimodal QA
* SONIC-O1 VQA benchmark

The implementation supports:

* 🔎 **Retrieval mode (RAG)** – segment-level audio/video embedding + top-k retrieval
* ❓ **Video QA mode** – multimodal inference over retrieved segments

Two isolated environments are required due to incompatible Torch stacks.

---

## Project Structure

```text
video_rag/
│
├── checkpoints/          # ImageBind weights
├── data/                 # VQA JSON + media files
├── .venv-rag/            # Retrieval environment
├── .venv-qa/             # QA environment
└── pyproject.toml
```

---

## 1. Download Model Checkpoint

```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
cd ..
```

This is required for embedding video/audio segments.

---

## 2. Download Dataset

### Create data directory

```bash
mkdir -p data
```

### VQA JSON Files

These are included in the GCP download below for convenience — same files as the SONIC-O1 HuggingFace dataset. They define the multiple-choice video QA tasks.

If you prefer to download them individually:

- [02_Job_Interviews.json](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/02_Job_Interviews.json)
- [04_Customer_Service_Interactions.json](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/04_Customer_Service_Interactions.json)
- [01_Patient-Doctor_Consultations.json](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/01_Patient-Doctor_Consultations.json)

### Video / Audio / Caption Files

The media dataset is hosted in a GCP bucket.

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
cd implementations/multimedia_rag
gcloud storage cp gs://interp-bootcamp-data/multimedia_rag/data.zip .
unzip data.zip
```

Files are placed correctly after extraction — no manual reorganisation needed.

#### 3) Cleanup temporary files

```bash
rm -f __MACOSX data.zip data/.DS_Store
```

The zip contains everything needed to run the notebooks:

```
data/
├── Customer_Service_Interactions/
│   ├── audio/                   # base audio files
│   ├── video/                   # base video files
│   ├── caption/                 # base caption files
│   ├── process-audio/           # pre-generated, can be regenerated
│   ├── process-video/           # pre-generated, can be regenerated
│   ├── segment-audio_30s/       # pre-generated, can be regenerated
│   ├── segment-video_30s/       # pre-generated, can be regenerated
│   ├── segment-caption_30s/     # pre-generated, can be regenerated
│   ├── audio_embeddings.pt      # pre-generated, can be regenerated
│   ├── video_embeddings.pt      # pre-generated, can be regenerated
│   └── caption_embeddings.pt    # pre-generated, can be regenerated
├── Job_Interviews/              # same structure as above
├── Patient-Doctor_Consultations/  # same structure as above
├── global_embeddings/           # pre-generated, can be regenerated
├── Customer_Service_Interactions.json
├── Customer_Service_Interactions_filtered.json  # pre-generated, can be regenerated
├── Job_Interviews.json
├── Job_Interviews_filtered.json               # pre-generated, can be regenerated
├── Patient-Doctor_Consultations.json
└── Patient-Doctor_Consultations_filtered.json # pre-generated, can be regenerated
```

Pre-generated files (`process-*`, `segment-*`, `*.pt` embeddings, `global_embeddings/`, `*_filtered.json`) are included to save time, but can all be reproduced by running the notebooks from scratch.

---

## 3. Environment Setup

Two independent environments are required due to package conflicts between the RAG (retrieval) and QA (inference) pipelines. Follow the instructions below to set up each environment.

---

> **On the Coder platform**, `ffmpeg` is already available. You can verify with:
>
> ```bash
> ffmpeg -version
> ```

<details>
<summary>Not on Coder? Install ffmpeg manually</summary>

```bash
sudo apt update
sudo apt install ffmpeg
```

</details>

Both the retrieval (RAG) and QA (inference) pipelines use the same dependency group. From the **root of the repository**:

```bash
uv sync --group multimedia-rag
source .venv/bin/activate
```

This installs everything needed for both the Video RAG (ImageBind embedding + retrieval) and Video QA (Qwen Omni inference) notebooks.

---

## 4. Running Notebooks

```bash
jupyter lab
```

---

## Implementation Overview

## Retrieval Pipeline

1. Segment video/audio
2. Encode segments using ImageBind
3. Encode question text
4. Compute cosine similarity
5. Retrieve top-k segments
6. Save retrieval metadata

## QA Pipeline

1. Load retrieval outputs
2. Load associated media segments
3. Pass multimodal inputs to Qwen Omni
4. Generate MCQ answer
5. Evaluate accuracy

## References and Resources

1. **[MAGNET: A Multi-agent Framework for Finding Audio-Visual Needles by Reasoning over Multi-Video Haystacks](https://arxiv.org/abs/2506.07016)**
    A comprehensive framework for temporal, causal, and multi-hop retrieval across long video haystacks.

2. **[SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding](https://arxiv.org/abs/2601.21666)**
    A benchmark for evaluating multimodal LLMs on real-world audio-video understanding tasks.

3. **[VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents](https://arxiv.org/abs/2410.10594)**
    A study on retrieval-augmented generation leveraging vision-based multi-modal documents.

4. **[Videos Dataset for LLMs RAG](https://huggingface.co/datasets/elmoghany/Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text)**
    A dataset designed for retrieval-augmented generation tasks requiring audio, video, and text.

5. **[SONIC-O1 Dataset](https://huggingface.co/datasets/vector-institute/sonic-o1)**
    A dataset for audio-visual question answering tasks across various domains.
