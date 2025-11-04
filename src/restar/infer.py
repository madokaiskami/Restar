"""Offline inference helper for JSONL review dumps."""

import argparse
import gzip
import json
import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import load_config
from .utils import append_ids, load_id_set, stable_id


def iter_jsonl(path: str) -> Iterable[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def run_inference(config_path: str, in_jsonl: str, out_parquet: str, append_seen: bool) -> None:
    cfg = load_config(config_path)
    run = getattr(cfg, "run_name", "restar_run")
    model_dir = os.path.join("outputs", run, "model")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    block_ids: set[str] = set()
    if hasattr(cfg, "blocklist"):
        block_ids = load_id_set(getattr(cfg.blocklist, "paths", []))

    rows: List[dict] = []
    ids: List[str] = []
    texts: List[str] = []
    for row in iter_jsonl(in_jsonl):
        rid = stable_id(row.get("text", ""), row.get("asin", ""))
        if rid in block_ids:
            continue
        rows.append(row)
        ids.append(rid)
        texts.append(row.get("text", ""))

    df = pd.DataFrame(rows)
    probs: List[np.ndarray] = []
    batch_size = 64

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if not batch:
                continue
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    stacked = np.vstack(probs) if probs else np.zeros((0, 3), dtype=np.float32)
    if len(df) != len(stacked):
        raise ValueError("Mismatch between collected rows and probability outputs")

    df["p_neg"], df["p_neu"], df["p_pos"] = (
        stacked[:, 0],
        stacked[:, 1],
        stacked[:, 2],
    )

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[ok] wrote {len(df)} rows -> {out_parquet}")

    if append_seen and hasattr(cfg, "blocklist"):
        seen_path = getattr(cfg.blocklist, "write_seen_to", "")
        if seen_path:
            append_ids(seen_path, ids)
            print(f"[ok] appended {len(ids)} ids -> {seen_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--in_jsonl", default="data/pool/pool_unlabeled.jsonl")
    parser.add_argument("--out_parquet", default="outputs/preds.parquet")
    parser.add_argument("--append_seen", action="store_true")
    args = parser.parse_args()

    run_inference(args.config, args.in_jsonl, args.out_parquet, args.append_seen)


if __name__ == "__main__":
    main()
