import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

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
        allow_replace_deleted: bool = True,
    ) -> None:
        self.index.init_index(
            max_elements=len(ids),
            ef_construction=ef_construction,
            M=m,
            allow_replace_deleted=allow_replace_deleted,
        )
        # replace_deleted requires allow_replace_deleted=True during init_index.
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
    return pd.read_parquet(parquet_path)


def build_hnsw(
    parquet_path: Path,
    index_path: Path,
    metadata_path: Path,
    m: int = 64,
    ef_construction: int = 200,
    embedding_col: str = "embedding",
    extra_capacity: int = 0,
    normalize: bool = True,
    model_id: Optional[str] = None,
) -> None:
    df = load_embeddings(parquet_path)
    if embedding_col not in df.columns:
        raise ValueError(f"Column '{embedding_col}' not found in {parquet_path}")

    embeddings = np.vstack(df[embedding_col].to_numpy()).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero if blank rows sneak in.
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
    ids = np.arange(len(df), dtype=np.int64)

    hnsw = SimpleHNSW(dim=embeddings.shape[1])
    hnsw.build(embeddings, ids, m=m, ef_construction=ef_construction, allow_replace_deleted=True)
    if extra_capacity > 0:
        hnsw.index.resize_index(len(ids) + extra_capacity)
    hnsw.save(index_path)

    metadata = {
        "dim": embeddings.shape[1],
        "m": m,
        "ef_construction": ef_construction,
        "embedding_col": embedding_col,
        "normalize": normalize,
        "extra_capacity": extra_capacity,
        "model_id": model_id or "",
        "allow_replace_deleted": True,
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

    items = metadata.get("items", [])
    items_by_id = {int(item["id"]): item for item in items if "id" in item}

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
        hit = items_by_id.get(int(label))
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
    build_p.add_argument(
        "--embedding-col",
        type=str,
        default="embedding",
        help="Which embedding column in the parquet to index (e.g., embedding, title_embedding).",
    )
    build_p.add_argument(
        "--extra-capacity",
        type=int,
        default=0,
        help="Pre-reserve additional slots for future inserts.",
    )
    build_p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip re-normalizing embeddings before indexing.",
    )
    build_p.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional model identifier to record in metadata for traceability.",
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
            embedding_col=args.embedding_col,
            extra_capacity=args.extra_capacity,
            normalize=not args.no_normalize,
            model_id=args.model_id or None,
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
