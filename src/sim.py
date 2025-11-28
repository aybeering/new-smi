import json
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index import SimpleHNSW


class Sim:
    """Simple similarity search against an HNSW index + metadata JSON."""

    DEFAULT_MODEL_DIR = Path("dir")

    @staticmethod
    def _infer_index_path(metadata_path: Path) -> Path:
        if metadata_path.suffix != ".json":
            raise ValueError("dataset_path must point to a metadata JSON file ending with .json")
        primary = metadata_path.with_suffix(".bin")
        if primary.exists():
            return primary
        # Fallback to common pattern: *_metadata.json -> *_hnsw.bin
        stem = metadata_path.stem
        if "_metadata" in stem:
            candidate = metadata_path.with_name(stem.replace("_metadata", "_hnsw") + ".bin")
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Index file not found. Expected {primary} or *_hnsw.bin next to metadata."
        )

    @staticmethod
    def query(
        query: str,
        dataset_path: str | Path,
        model_path: Optional[str | Path] = None,
    ) -> Tuple[str, Path]:
        """
        Search the HNSW index associated with the given metadata JSON.

        Returns a tuple of (best_match_title, index_path).
        Only the top-1 title is returned.
        """

        started = time.perf_counter()
        metadata_path = Path(dataset_path)
        index_path = Sim._infer_index_path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found next to metadata: {index_path}")

        metadata = json.loads(metadata_path.read_text())
        dim = metadata.get("dim")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Invalid or missing 'dim' in metadata.")

        items = metadata.get("items", [])
        items_by_id = {int(item["id"]): item for item in items if "id" in item}

        hnsw = SimpleHNSW(dim=dim)
        hnsw_load_started = time.perf_counter()
        hnsw.load(index_path)
        hnsw_load_ms = (time.perf_counter() - hnsw_load_started) * 1000
        # If metadata recorded ef_construction, reuse it as a reasonable search ef.
        ef_search = metadata.get("ef_construction", 200) or 200
        hnsw.index.set_ef(int(ef_search))

        model_dir = Path(model_path) if model_path else Path(metadata.get("model_id") or Sim.DEFAULT_MODEL_DIR)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path not found: {model_dir}")

        model_load_started = time.perf_counter()
        model = SentenceTransformer(str(model_dir), trust_remote_code=True)
        model_load_ms = (time.perf_counter() - model_load_started) * 1000

        encode_started = time.perf_counter()
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        encode_ms = (time.perf_counter() - encode_started) * 1000

        labels, distances = hnsw.search(query_vec, top_k=1)
        search_ms = (time.perf_counter() - encode_started) * 1000 - encode_ms
        label = int(labels[0][0])
        hit = items_by_id.get(label)
        if not hit:
            raise ValueError("No matching item found in metadata for the returned label.")

        best_match_text = hit.get("title", "")
        # Latency metrics are kept for potential downstream logging.
        _ = (
            1 - float(distances[0][0]),
            (time.perf_counter() - started) * 1000,
            model_load_ms,
            hnsw_load_ms,
            encode_ms,
            search_ms,
        )
        return best_match_text, index_path

    @staticmethod
    def agent(
        query: str,
        dataset_path: str | Path,
        model_path: Optional[str | Path] = None,
    ) -> dict:
        """
        调用 workflow2，判定原 query 与检索到的 top-1 是否同一事件。

        返回形如 {"same_event": true|false|"unknown", "reason": "...", "title": "...", "analysis": {...}}。
        """
        # 按需引入，避免循环依赖在加载时触发。
        from src.agent.workflow2 import run_once

        try:
            result_text = run_once(
                query,
                str(dataset_path),
                model_path=str(model_path) if model_path else None,
            )
            try:
                return json.loads(result_text)
            except Exception:
                return {"same_event": "unknown", "reason": f"无法解析判定结果：{result_text}"}
        except Exception as e:
            return {"same_event": "unknown", "reason": f"error: {e}"}
