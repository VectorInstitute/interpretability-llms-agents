"""Register ChartQAPro samples as a Langfuse Dataset.

Usage:
    python -m agentic_chartqapro_eval.langfuse_integration.dataset \
        --split test --n 25
"""

import argparse
from typing import Optional

from .client import get_client


def register_dataset(
    samples,
    dataset_name: str = "ChartQAPro",
    split: str = "test",
) -> Optional[str]:
    """Insert PerceivedSamples into a Langfuse Dataset named ``{dataset_name}_{split}``.

    Returns the dataset name, or None if Langfuse is unavailable.
    """
    client = get_client()
    if client is None:
        return None

    name = f"{dataset_name}_{split}"
    try:
        client.create_dataset(name=name)
        for s in samples:
            client.create_dataset_item(
                dataset_name=name,
                input={
                    "source_id": s.sample_id,
                    "question": s.question,
                    "question_type": s.question_type.value,
                    "image_path": s.image_path or "",
                    "choices": s.choices or [],
                },
                expected_output=s.expected_output,
            )
        print(f"[langfuse] Registered {len(samples)} samples → dataset '{name}'")
        return name
    except Exception as exc:
        print(f"[langfuse] Dataset registration failed: {exc}")
        return None


def main() -> None:
    """Register ChartQAPro dataset samples in Langfuse."""
    parser = argparse.ArgumentParser(
        description="Register ChartQAPro samples as Langfuse dataset"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--image_dir", default="data/chartqapro_images")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    from ..datasets.chartqapro_loader import load_chartqapro

    samples = load_chartqapro(
        split=args.split, n=args.n, image_dir=args.image_dir, cache_dir=args.cache_dir
    )
    register_dataset(samples, split=args.split)


if __name__ == "__main__":
    main()
