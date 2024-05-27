from collections import defaultdict
from pathlib import Path

import faiss
import jsonargparse
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from sqlitedict import SqliteDict
from tqdm import tqdm
from usearch.index import Index


def create_binary_index(dataset, output_path: Path, batch_size: int = 10000):
    index = faiss.IndexBinaryFlat(1024)

    def _add_batch_to_index(batch):
        nonlocal index
        index.add(np.array(batch["emb_ubinary"], dtype=np.uint8))  # type: ignore

    dataset.map(_add_batch_to_index, batched=True, batch_size=batch_size)

    faiss.write_index_binary(index, str(output_path.absolute()))
    del index


def create_uint8_index(dataset, index_output_path: Path, batch_size: int = 10000):
    index = Index(ndim=1024, metric="ip", dtype="i8")

    i = 0

    def _add_batch_to_index(batch):
        nonlocal i
        nonlocal index

        embeddings_int8 = np.array(batch["emb_int8"], dtype=np.int8)
        index.add(np.arange(i, i + len(embeddings_int8)), embeddings_int8)
        i += len(embeddings_int8)

    dataset.map(_add_batch_to_index, batched=True, batch_size=batch_size)

    index.save(index_output_path)
    del index


def create_url_mapping(dataset, output_path: Path):
    url_mapping = defaultdict(list)
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        url_mapping[item["url"]].append(i)

    db = SqliteDict(output_path)

    for url, indices in tqdm(url_mapping.items()):
        db[url] = indices

    db.commit()
    db.close()


def create_wikipedia_text_dataset(dataset, output_path: Path):
    dataset.remove_columns(["emb_ubinary", "emb_int8"]).save_to_disk(output_path)


def main(args):
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    batch_size = args.batch_size

    print("Loading wikipedia dataset...")
    dataset = load_dataset("krasserm/wikipedia-2023-11-en-embed-mxbai-int8-binary")

    print("Creating binary index...")
    binary_index_output_path = output_dir / "faiss-ubinary.index"
    create_binary_index(dataset, binary_index_output_path, batch_size)

    print("Creating int8 index...")
    int8_index_output_path = output_dir / "usearch-int8.index"
    create_uint8_index(dataset, int8_index_output_path, batch_size)

    print("Creating url mapping...")
    url_mapping_output_path = output_dir / "document-url-mappings.sqlite"
    create_url_mapping(dataset, url_mapping_output_path)

    print("Creating text dataset...")
    dataset_output_path = output_dir / "wikipedia-en-text"
    create_wikipedia_text_dataset(dataset, dataset_output_path)


if __name__ == "__main__":
    load_dotenv()

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("output", "wikipedia_search_tool"))
    parser.add_argument("--batch_size", type=int, default=10000)
    main(parser.parse_args())
