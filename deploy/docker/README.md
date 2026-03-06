# Docker Build & Push Guide

This guide explains how to build Docker images for the LLM Interpretability Bootcamp and push them to Google Cloud Artifact Registry.

## Registry Location

- **Project:** `coderd`
- **Location:** `us-central1`
- **Repository:** `coder`
- **Image:** `llm-interpretability-bootcamp`
- **Full path:** `us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp`

Two tags are maintained:

| Tag | Base image | Used for |
|-----|-----------|----------|
| `latest` | `ubuntu:24.04` | CPU workspaces (`e2-standard-2`) |
| `gpu` | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | GPU workspaces (`g2-standard-8`, `g2-standard-24`) |

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Access to the `coderd` Google Cloud project

## Setup

### 1. Authenticate with gcloud

```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project coderd
```

### 2. Configure Docker for Artifact Registry

```sh
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Build and Push

### CPU image (`latest`)

#### Option A: Using Cloud Build (recommended)

```sh
cd /path/to/interpretability_agent_bootcamp
gcloud builds submit --region=us-central1 \
  --tag us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:latest \
  -f deploy/docker/Dockerfile .
```

#### Option B: Build locally and push

```sh
cd /path/to/interpretability_agent_bootcamp

docker build -t us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:latest \
  -f deploy/docker/Dockerfile .

docker push us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:latest
```

### GPU image (`gpu`)

Uses `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` as the base via the `BASE_IMAGE` build arg.

#### Option A: Using Cloud Build (recommended)

```sh
cd /path/to/interpretability_agent_bootcamp
gcloud builds submit --region=us-central1 \
  --tag us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 \
  -f deploy/docker/Dockerfile .
```

#### Option B: Build locally and push

```sh
cd /path/to/interpretability_agent_bootcamp

docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 \
  -t us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:gpu \
  -f deploy/docker/Dockerfile .

docker push us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:gpu
```

## Managing Images

### List existing images

```sh
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp \
  --include-tags
```

### Delete an image (if needed)

```sh
# CPU
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:latest \
  --quiet

# GPU
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:gpu \
  --quiet
```

### Pull an image

```sh
# CPU
docker pull us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:latest

# GPU
docker pull us-central1-docker.pkg.dev/coderd/coder/llm-interpretability-bootcamp:gpu
```

## References

- [Build and push a Docker image with Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image)
- [Artifact Registry Docker Guide](https://cloud.google.com/artifact-registry/docs/docker)
- [gcloud CLI Documentation](https://cloud.google.com/sdk/gcloud)
