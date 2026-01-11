"""Prepare stage for DVC: materialize train/dev/eval parquet datasets."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any, Optional

import pandas as pd

from .config import load_config
from .data import (
    build_balanced_train_from_hf_stream,
    load_dev,
    load_frozen_eval,
    load_local_parquet,
)
from .utils import maybe_load_blocklist, set_seed, setup_logging


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a ``label`` column; derive from rating if missing."""

    if "label" in df.columns:
        return df

    def _label_from_rating(value: Any) -> int:
        try:
            rating = float(value)
        except Exception:
            rating = 0.0
        if rating <= 2:
            return 0
        if rating >= 4:
            return 2
        return 1

    df = df.copy()
    df["label"] = df.get("rating", []).apply(_label_from_rating)
    return df


def _to_dataframe(dataset) -> pd.DataFrame:
    """Convert a datasets.Dataset (or compatible) to a pandas DataFrame."""

    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    # Fallback: best-effort conversion
    return pd.DataFrame(dataset)


def _dump_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _log_distribution(name: str, df: pd.DataFrame, path: str) -> None:
    labels = df["label"] if "label" in df.columns else []
    dist = Counter(labels)
    print(f"[prepare] {name}: rows={len(df)} label_dist={dict(dist)} -> {path}")


def _sample_dataframe(
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
    *,
    stratify_col: str = "label",
) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    if stratify_col in df.columns and not df.empty:
        counts = df[stratify_col].value_counts()
        total = len(df)
        proportions = counts / total
        base = (proportions * max_rows).apply(lambda x: int(x))
        remainder = max_rows - int(base.sum())
        if remainder > 0:
            fractional = (proportions * max_rows) - base
            for label in fractional.sort_values(ascending=False).index:
                if remainder <= 0:
                    break
                base[label] += 1
                remainder -= 1

        parts = []
        for label, group in df.groupby(stratify_col):
            target = int(base.get(label, 0))
            if target <= 0:
                continue
            if len(group) <= target:
                parts.append(group)
            else:
                parts.append(group.sample(n=target, random_state=seed))
        sampled = pd.concat(parts) if parts else df.sample(n=max_rows, random_state=seed)
        if len(sampled) < max_rows:
            remaining = df.drop(sampled.index, errors="ignore")
            fill = min(max_rows - len(sampled), len(remaining))
            if fill > 0:
                sampled = pd.concat(
                    [sampled, remaining.sample(n=fill, random_state=seed)],
                )
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _load_local_dataset(config_section, name: str) -> Optional[Any]:
    local_path = getattr(config_section, "local_parquet", None) if config_section else None
    if local_path and os.path.exists(str(local_path)):
        print(f"[prepare] loading {name} from local parquet: {local_path}")
        return load_local_parquet(str(local_path), with_label=True)
    return None


def main():
    parser = argparse.ArgumentParser(description="DVC prepare stage")
    parser.add_argument("--config", default="configs/dvc_smoke.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = getattr(cfg, "run_name", "dvc_run")
    out_train = getattr(
        getattr(cfg, "prepare", None),
        "out_train_parquet",
        "data/raw/train.parquet",
    )
    out_dev = getattr(
        getattr(cfg, "prepare", None),
        "out_dev_parquet",
        "data/raw/dev.parquet",
    )
    out_eval = getattr(
        getattr(cfg, "prepare", None),
        "out_eval_parquet",
        "data/raw/eval.parquet",
    )

    setup_logging(True, None)
    set_seed(getattr(cfg, "random_seed", 42))

    block_ids = maybe_load_blocklist(getattr(cfg, "blocklist", None))
    if block_ids:
        print(f"[prepare] loaded blocklist entries: {len(block_ids)}")

    train_stream_cfg = getattr(cfg, "train_stream", None)
    if train_stream_cfg is None:
        raise ValueError("Configuration is missing the 'train_stream' section")

    train_ds = _load_local_dataset(train_stream_cfg, "train")
    if train_ds is None:
        train_ds, class_counts = build_balanced_train_from_hf_stream(
            train_stream_cfg,
            block_ids=block_ids,
        )
        print(f"[prepare] built train dataset via streaming with counts={class_counts}")
    else:
        class_counts = Counter(train_ds["label"]) if "label" in train_ds.column_names else {}

    dev_ds = _load_local_dataset(cfg.dev, "dev")
    if dev_ds is None:
        dev_ds = load_dev(cfg.dev, block_ids=None)

    eval_ds = _load_local_dataset(cfg.eval, "eval")
    if eval_ds is None:
        eval_ds = load_frozen_eval(cfg.eval, block_ids=None)

    train_df = _ensure_label_column(_to_dataframe(train_ds))
    dev_df = _ensure_label_column(_to_dataframe(dev_ds))
    eval_df = _ensure_label_column(_to_dataframe(eval_ds))

    train_max_rows = int(getattr(train_stream_cfg, "max_rows", 0) or 0)
    train_seed = int(getattr(train_stream_cfg, "seed", getattr(cfg, "random_seed", 42)))
    if train_max_rows > 0:
        train_df = _sample_dataframe(train_df, train_max_rows, train_seed)

    dev_max_rows = int(getattr(cfg.dev, "max_rows", 0) or 0)
    dev_seed = int(getattr(cfg.dev, "seed", getattr(cfg, "random_seed", 42)))
    if dev_max_rows > 0:
        dev_df = _sample_dataframe(dev_df, dev_max_rows, dev_seed)

    eval_max_rows = int(getattr(cfg.eval, "max_rows", 0) or 0)
    eval_seed = int(getattr(cfg.eval, "seed", getattr(cfg, "random_seed", 42)))
    if eval_max_rows > 0:
        eval_df = _sample_dataframe(eval_df, eval_max_rows, eval_seed)

    _dump_parquet(train_df, out_train)
    _dump_parquet(dev_df, out_dev)
    _dump_parquet(eval_df, out_eval)

    _log_distribution("train", train_df, out_train)
    _log_distribution("dev", dev_df, out_dev)
    _log_distribution("eval", eval_df, out_eval)

    print(
        "[prepare] completed\n"
        f"  train -> {out_train}\n"
        f"  dev   -> {out_dev}\n"
        f"  eval  -> {out_eval}\n"
        f"  run_name={run_name}"
    )


if __name__ == "__main__":
    main()
