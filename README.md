Project: Rule/Title Embeddings + HNSW Search
=============================================

This project loads a local SentenceTransformer model from `dir/`, embeds the `rules` (and `title`) columns in `data/sample.csv`, builds an HNSW index for fast similarity search, and supports match-or-add logic with a configurable similarity threshold.

Setup
-----
- Python >=3.9
- Install deps:
  ```bash
  python -m pip install -r requirements.txt  # if present
  # or minimally
  python -m pip install sentence-transformers hnswlib pandas pyarrow
  ```
- (Optional, if using the mirror) `export HF_ENDPOINT="https://hf-mirror.com"`

Generate embeddings
-------------------
```bash
python src/embed_rules.py \
  --csv data/sample.csv \
  --model dir \
  --output data/sample_embeddings.parquet
```
Columns written: `title`, `rules`, `embedding` (rules vector), `title_embedding` (title vector).

Build HNSW index
----------------
By default uses `embedding` (rules vectors):
```bash
python src/index.py build \
  --embeddings data/sample_embeddings.parquet \
  --index-path data/rules_hnsw.bin \
  --metadata-path data/rules_metadata.json
```
If you want a title-only index, pass `--embedding-col title_embedding` (and optionally `--extra-capacity` to pre-reserve space for future inserts); write to a different index/metadata file.

Search
------
```bash
python src/index.py search \
  --query "Egg price < $3.00 by September 30, 2025?" \
  --model dir \
  --index-path data/rules_hnsw.bin \
  --metadata-path data/rules_metadata.json \
  --top-k 5 \
  --ef-search 400   # increase for better recall
```
Outputs top-k with cosine-like scores (`1 - hnsw distance`).

Match-or-add (pair detection)
-----------------------------
Given one sentence, find the closest in the dataset. If best score < threshold, treat as new: append its embedding to the parquet.
```bash
python src/index.py match \
  --text "Egg price < $3.00 by September 30, 2025?" \
  --model dir \
  --parquet data/sample_embeddings.parquet \
  --threshold 0.7 \
  --title ""   # optional title for new rows
```
Returns JSON with `is_new`, `score`, and matched/added index. If new data is appended, rebuild the HNSW index to include it.

Notes
-----
- After regenerating embeddings (or appending via `match`), rerun `index.py build` to keep the HNSW index in sync.
- For better quality, you can switch to a stronger model (point `--model` to the new directory), regenerate embeddings, and rebuild the index.
- Programmatic search: `from src.sim import Sim; text, index_path = Sim.test(query="...", dataset_path="data/rules_metadata.json", model_path="dir")`
