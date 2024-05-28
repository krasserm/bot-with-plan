import os
import shutil
from pathlib import Path

import jsonargparse
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings


def create_wikipedia_float32_emb_dataset(dataset_output_path: Path, batch_size: int = 64):
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").eval()

    def _encode(batch, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        embedding_model.to(device)

        with torch.no_grad():
            return {
                "_id": batch["_id"],
                "title": batch["title"],
                "text": batch["text"],
                "url": batch["url"],
                "emb_float": embedding_model.encode(
                    batch["text"],
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    device=device,
                ),
            }

    dataset = load_cohere_wikipedia_dataset(subset="en")
    dataset = dataset.map(
        _encode,
        batched=True,
        batch_size=1000,
        with_rank=True,
        num_proc=torch.cuda.device_count(),
    )
    dataset.save_to_disk(dataset_output_path)


def load_cohere_wikipedia_dataset(subset: str):
    return load_dataset(
        "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
        name=subset,
        split="train",
        cache_dir="/mnt/data/datasets/huggingface",
    ).remove_columns(["emb_int8", "emb_ubinary"])


def create_dataset_ranges(dataset, ranges_output_path: Path, batch_size: int = 10000):
    emb_min = None
    emb_max = None

    def _create_ranges(batch):
        nonlocal emb_min
        nonlocal emb_max

        embeddings = batch["emb_float"]
        if emb_min is None:
            emb_min = np.min(embeddings, axis=0)
        else:
            emb_min = np.min(np.vstack((emb_min, embeddings)), axis=0)

        if emb_max is None:
            emb_max = np.max(embeddings, axis=0)
        else:
            emb_max = np.max(np.vstack((emb_max, embeddings)), axis=0)

    dataset.map(
        _create_ranges,
        batched=True,
        batch_size=batch_size,
    )

    ranges = np.vstack((emb_min, emb_max))
    np.save(ranges_output_path, ranges)


def create_wikipedia_quantized_dataset(dataset, ranges: np.ndarray, output_path: Path, batch_size: int = 1000):
    def _add_quantized_columns(batch):
        embeddings = np.array(batch["emb_float"])
        batch["emb_ubinary"] = quantize_embeddings(embeddings, precision="ubinary")
        batch["emb_int8"] = quantize_embeddings(embeddings, precision="int8", ranges=ranges)
        return batch

    dataset = dataset.map(_add_quantized_columns, batched=True, batch_size=batch_size)
    dataset = dataset.remove_columns(["emb_float"])
    dataset.save_to_disk(output_path)


def main(args):
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print("Creating text embeddings...")
    float32_emb_dataset_path = output_dir / "wikipedia-en-float32-emb"
    create_wikipedia_float32_emb_dataset(float32_emb_dataset_path, batch_size=args.encode_batch_size)

    float32_emb_dataset = load_from_disk(str(float32_emb_dataset_path))

    print("Creating dataset ranges...")
    dataset_ranges_path = output_dir / "wikipedia-en-data-ranges.npy"
    create_dataset_ranges(float32_emb_dataset, dataset_ranges_path)

    print("Creating quantized dataset...")
    quantized_dataset_path = output_dir / "wikipedia-2023-11-en-embed-mxbai-int8-binary"
    create_wikipedia_quantized_dataset(float32_emb_dataset, np.load(dataset_ranges_path), quantized_dataset_path)

    shutil.rmtree(float32_emb_dataset_path)
    os.remove(dataset_ranges_path)


if __name__ == "__main__":
    load_dotenv()

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("output", "wikipedia-2023-11-en"))
    parser.add_argument("--encode_batch_size", type=int, default=64)
    main(parser.parse_args())
