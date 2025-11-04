"""Utilities to prepare an uncertainty-based batch for annotation."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .config import load_config
from .utils import setup_logging


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(True)

    scored_path = cfg.al.scored_path
    todo_path = cfg.al.todo_path
    batch_size = args.k or cfg.al.batch_size
    strategy = (args.strategy or cfg.al.strategy).lower()

    df = pd.read_parquet(scored_path)
    probs = df[["p_neg", "p_neu", "p_pos"]].to_numpy()
    if strategy == "least_confidence":
        score = 1.0 - np.max(probs, axis=1)
    elif strategy == "margin":
        sorted_probs = -np.sort(-probs, axis=1)
        score = -(sorted_probs[:, 0] - sorted_probs[:, 1])
    elif strategy == "entropy":
        score = entropy(probs)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    idx = np.argsort(-score)[:batch_size]
    out = df.iloc[idx][["id", "text", "title", "asin", "product_category"]].copy()
    out["label"] = ""
    todo_dir = os.path.dirname(todo_path)
    if todo_dir:
        os.makedirs(todo_dir, exist_ok=True)
    out.to_csv(todo_path, index=False)
    logger.info("Prepared %d samples for labeling -> %s", len(out), todo_path)


if __name__ == "__main__":
    main()
