import argparse
from pathlib import Path
from typing import List
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_rules(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV and keep only the columns we need.
    The sample data contains trailing commas, so restrict to explicit columns.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["title", "rules"],
        engine="python",
        dtype=str,
        on_bad_lines="skip",
    )
    df["rules"] = df["rules"].fillna("").str.strip()
    df = df[df["rules"] != ""].reset_index(drop=True)
    return df


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> List[List[float]]:
    """Encode texts with the given model into normalized embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed rules column with local model.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/sample.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("dir"),
        help="Path to the local SentenceTransformer model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample_embeddings.parquet"),
        help="Where to write the embeddings with metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for encoding.",
    )
    args = parser.parse_args()

    model = SentenceTransformer(str(args.model), trust_remote_code=True)
    df = load_rules(args.csv)
    df["embedding"] = embed_texts(model, df["rules"].tolist(), batch_size=args.batch_size)
    df["title_embedding"] = embed_texts(model, df["title"].fillna("").tolist(), batch_size=args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
