import argparse
import json
from pathlib import Path
from typing import List, Tuple

import hnswlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer



class SimpleHNSW:
    """
    Minimal wrapper around hnswlib for cosine similarity search.
    Stores metadata in a separate JSON file.
    """

    def __init__(self, dim: int, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)

    def build(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray,
        m: int = 64,
        ef_construction: int = 200,
    ) -> None:
        self.index.init_index(
            max_elements=len(ids),
            ef_construction=ef_construction,
            M=m,
        )
        # replace_deleted requires allow_replace_deleted=True during init_index; avoid for compatibility.
        self.index.add_items(embeddings, ids)
        self.index.set_ef(ef_construction)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(path))

    def load(self, path: Path) -> None:
        self.index.load_index(str(path))

    def search(self, vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        labels, distances = self.index.knn_query(vectors, k=top_k)
        return labels, distances


def load_embeddings(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "embedding" not in df.columns:
        raise ValueError("Expected an 'embedding' column in the parquet file.")
    return df


def build_hnsw(
    parquet_path: Path,
    index_path: Path,
    metadata_path: Path,
    m: int = 64,
    ef_construction: int = 200,
) -> None:
    df = load_embeddings(parquet_path)
    embeddings = np.vstack(df["title_embedding"].to_numpy()).astype(np.float32)
    ids = np.arange(len(df), dtype=np.int64)

    hnsw = SimpleHNSW(dim=embeddings.shape[1])
    hnsw.build(embeddings, ids, m=m, ef_construction=ef_construction)
    hnsw.save(index_path)

    metadata = {
        "dim": embeddings.shape[1],
        "items": [
            {"id": int(idx), "title": row.get("title", ""), "rules": row.get("rules", "")}
            for idx, row in df.iterrows()
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Built HNSW index with {len(df)} items -> {index_path}")
    print(f"Saved metadata -> {metadata_path}")


def search_hnsw(
    query: str,
    model_path: Path,
    index_path: Path,
    metadata_path: Path,
    top_k: int = 5,
    ef_search: int = 200,
) -> List[dict]:
    metadata = json.loads(metadata_path.read_text())
    dim = metadata.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Metadata missing a valid 'dim' field.")

    hnsw = SimpleHNSW(dim=dim)
    hnsw.load(index_path)
    hnsw.index.set_ef(ef_search)

    model = SentenceTransformer(str(model_path), trust_remote_code=True)
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    labels, distances = hnsw.search(query_vec, top_k=top_k)
    results = []
    for label, dist in zip(labels[0], distances[0]):
        hit = next((item for item in metadata["items"] if item["id"] == int(label)), None)
        if hit:
            results.append(
                {
                    "id": int(label),
                    "score": 1 - float(dist),  # cosine similarity = 1 - distance in hnswlib
                    "title": hit["title"],
                    "rules": hit["rules"],
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a SimpleHNSW index over rule embeddings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build the HNSW index from parquet embeddings.")
    build_p.add_argument("--embeddings", type=Path, default=Path("data/sample_embeddings.parquet"))
    build_p.add_argument("--index-path", type=Path, default=Path("data/rules_hnsw.bin"))
    build_p.add_argument("--metadata-path", type=Path, default=Path("data/rules_metadata.json"))
    build_p.add_argument("--m", type=int, default=64, help="HNSW M parameter.")
    build_p.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="Construction time ef parameter.",
    )

    search_p = subparsers.add_parser("search", help="Query the HNSW index with a text prompt.")
    search_p.add_argument("--query", required=True, help="Query text to embed and search.")
    search_p.add_argument("--model", type=Path, default=Path("dir"), help="Path to local model.")
    search_p.add_argument("--index-path", type=Path, default=Path("data/rules_hnsw.bin"))
    search_p.add_argument("--metadata-path", type=Path, default=Path("data/rules_metadata.json"))
    search_p.add_argument("--top-k", type=int, default=5)
    search_p.add_argument("--ef-search", type=int, default=200)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_hnsw(
            parquet_path=args.embeddings,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            m=args.m,
            ef_construction=args.ef_construction,
        )
    elif args.command == "search":
        results = search_hnsw(
            query=args.query,
            model_path=args.model,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            top_k=args.top_k,
            ef_search=args.ef_search,
        )
        for hit in results:
            print(
                f"[{hit['id']}] score={hit['score']:.4f} title={hit['title'][:80]}",
            )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
