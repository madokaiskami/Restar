"""Merge human-provided labels into a clean parquet file."""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

from .config import load_config
from .utils import setup_logging

_LABEL_STR_MAP = {"neg": "negative", "pos": "positive", "neu": "neutral"}
_LABEL_VAL_MAP = {"negative": 0, "neutral": 1, "positive": 2, "0": 0, "1": 1, "2": 2}


def _normalize_label(x: Any) -> int:
    s = str(x).strip().lower()
    s = _LABEL_STR_MAP.get(s, s)
    v = _LABEL_VAL_MAP.get(s, s)
    return int(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("merge_labels")

    labels_done: str = cfg.al.labels_done_path
    out_parquet: str = cfg.al.human_gold_path

    df = pd.read_csv(labels_done)
    df = df.copy()
    df["label"] = df["label"].map(_normalize_label)

    keep = ["text", "asin", "label"]
    for col in keep:
        if col not in df.columns:
            df[col] = "" if col != "label" else 0

    out_dir = os.path.dirname(out_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df[keep].to_parquet(out_parquet, index=False)
    logger.info("Wrote merged labels -> %s", os.path.abspath(out_parquet))


if __name__ == "__main__":
    main()
