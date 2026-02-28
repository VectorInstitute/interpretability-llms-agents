import os
import time
import torch

def process_retrieved_files(
    retrieved_files,
    question,
    root_data_dir,
    segment_suffix,
    model,
    bsz,
    default_topic=None,
):
    """
    Process retrieved files and generate answers using the model.

    Args:
    - retrieved_files (list): List of retrieved file identifiers.
    - question (str): The question to be answered.
    - root_data_dir (str): Root directory containing the data.
    - segment_suffix (str): Suffix for segmented media directories (e.g., "30s").
    - model: The multimodal model to use for generating answers.
    - bsz (int): Batch size for processing.
    - default_topic (str, optional): Default topic to use if not specified in the retrieved file name.

    Returns:
    - dict: A dictionary mapping retrieved file identifiers to generated answers.
    """

    agent_answers = {}

    for retrieved_item in retrieved_files:

        if isinstance(retrieved_item, dict):
            retrieved_file = retrieved_item["file"]
        else:
            retrieved_file = retrieved_item

        # ---- Detect global vs local ----
        parts = retrieved_file.split("__")

        if len(parts) >= 3:
            # Global format: Topic__002__000
            topic = parts[0]
            segment_name = "__".join(parts[1:])
        else:
            # Local format: 002__000
            topic = default_topic
            segment_name = retrieved_file

        video_path = os.path.join(
            root_data_dir,
            topic,
            f"segment-video_{segment_suffix}",
            f"{segment_name}.mp4"
        )

        audio_path = os.path.join(
            root_data_dir,
            topic,
            f"segment-audio_{segment_suffix}",
            f"{segment_name}.wav"
        )
            
        if not os.path.exists(video_path):
            print(f"[ERROR] Missing video: {video_path}")
            continue

        inputs = [{
            "text": question,
            "video": video_path,
            "audio": audio_path
        }]

        inputs = model.prepare_input(inputs)

        start_time = time.time()
        text, _ = model.generate(inputs)
        end_time = time.time()

        print(f"[{retrieved_file}] Inference: {end_time - start_time:.2f}s")

        agent_answers[retrieved_file] = text

        torch.cuda.empty_cache()

    return agent_answers


def process_question(source, root_data_dir, segment_suffix, model, bsz, topic):
    """
    Processes a single question using hybrid pipeline.
    """

    question = source["question"]
    retrieved_files = source["retrieved_file"]

    if isinstance(retrieved_files, str):
        retrieved_files = [retrieved_files]

    agent_answers = process_retrieved_files(
        retrieved_files=retrieved_files,
        question=question,
        root_data_dir=root_data_dir,
        segment_suffix=segment_suffix,
        model=model,
        bsz=bsz,
        default_topic=topic,
    )

    source["agent_answers"] = agent_answers
    return source