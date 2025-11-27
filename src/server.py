import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.index import SimpleHNSW
from src.sim import Sim


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text to search for")
    top_k: int = Field(5, description="Number of nearest neighbors to return")
    ef_search: Optional[int] = Field(None, description="Override ef_search for this query")


class SearchResult(BaseModel):
    id: int
    score: float
    title: str
    rules: str


def _load_resources():
    metadata_path = Path(os.environ.get("METADATA_PATH", "data/rules_metadata.json"))
    index_path = Sim._infer_index_path(metadata_path)
    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Metadata or index missing: {metadata_path} / {index_path}")

    metadata = json.loads(metadata_path.read_text())
    dim = metadata.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Invalid 'dim' in metadata")

    items = metadata.get("items", [])
    items_by_id = {int(item["id"]): item for item in items if "id" in item}

    hnsw = SimpleHNSW(dim=dim)
    hnsw.load(index_path)
    ef_default = metadata.get("ef_construction", 200) or 200
    hnsw.index.set_ef(int(ef_default))

    model_dir = Path(os.environ.get("MODEL_PATH") or metadata.get("model_id") or Sim.DEFAULT_MODEL_DIR)
    model = SentenceTransformer(str(model_dir), trust_remote_code=True)

    return {
        "metadata_path": metadata_path,
        "index_path": index_path,
        "metadata": metadata,
        "items_by_id": items_by_id,
        "hnsw": hnsw,
        "ef_default": ef_default,
        "model": model,
    }


app = FastAPI(title="Simple HNSW Search")
state = _load_resources()


@app.get("/health")
def health():
    return {"status": "ok", "index": str(state["index_path"]), "model": str(state["model"]) }


@app.post("/search", response_model=List[SearchResult])
def search(body: SearchRequest):
    if body.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")

    ef = body.ef_search or state["ef_default"]
    state["hnsw"].index.set_ef(int(ef))

    query_vec = state["model"].encode(
        [body.query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    labels, distances = state["hnsw"].search(query_vec, top_k=body.top_k)
    results: List[SearchResult] = []
    for label, dist in zip(labels[0], distances[0]):
        item = state["items_by_id"].get(int(label))
        if not item:
            continue
        results.append(
            SearchResult(
                id=int(label),
                score=1 - float(dist),
                title=item.get("title", ""),
                rules=item.get("rules", ""),
            )
        )
    return results


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.server:app", host=host, port=port, reload=False)

