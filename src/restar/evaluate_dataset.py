"""Evaluate a saved model on a local parquet dataset (DVC evaluate stage)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import load_config
from .data import load_frozen_eval, load_local_parquet
from .eval_utils import maybe_plot_confusion
from .train import compute_metrics, create_preprocess_fn
from .utils import set_seed


def _resolve_eval_dataset(cfg, default_path: str):
    eval_local_path: Optional[str] = getattr(
        getattr(cfg, "eval", None),
        "local_parquet",
        None,
    )
    if eval_local_path:
        eval_ds = load_local_parquet(eval_local_path, with_label=True)
        max_rows = int(getattr(cfg.eval, "max_rows", 0) or 0)
        if max_rows > 0:
            eval_seed = int(getattr(cfg.eval, "seed", getattr(cfg, "random_seed", 42)))
            eval_ds = eval_ds.shuffle(seed=eval_seed)
            eval_ds = eval_ds.select(range(min(max_rows, len(eval_ds))))
        return eval_ds
    if default_path and os.path.exists(default_path):
        return load_local_parquet(default_path, with_label=True)
    return load_frozen_eval(cfg.eval, block_ids=None)


logger = logging.getLogger(__name__)


def _resolve_model_source(cfg, trained_dir: str) -> str:
    model_cfg = getattr(cfg, "model", None)
    pretrained_name = getattr(model_cfg, "pretrained_name", "distilbert-base-uncased")
    local_dir = getattr(model_cfg, "pretrained_local_dir", None) if model_cfg else None

    if trained_dir and os.path.isdir(trained_dir):
        logger.info("Loading trained model from %s", os.path.abspath(trained_dir))
        return trained_dir

    if local_dir and os.path.isdir(str(local_dir)):
        logger.info(
            "Model directory %s missing; using local pretrained directory %s",
            trained_dir,
            os.path.abspath(str(local_dir)),
        )
        return str(local_dir)

    logger.warning(
        "Model directory %s missing; falling back to pretrained identifier %s",
        trained_dir,
        pretrained_name,
    )
    return pretrained_name


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on a local parquet dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/dvc_smoke.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config)
    set_seed(getattr(cfg, "random_seed", 42))

    output_root = getattr(cfg, "output_dir", "outputs")
    run_name = getattr(cfg, "run_name", "dvc_run")
    out_dir = os.path.join(output_root, run_name)
    model_dir = os.path.join(out_dir, "model")
    metrics_path = os.path.join(out_dir, "metrics_eval.json")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    os.makedirs(out_dir, exist_ok=True)

    model_source = _resolve_model_source(cfg, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)

    eval_default = "data/raw/eval.parquet"
    eval_ds = _resolve_eval_dataset(cfg, eval_default)

    max_length = int(getattr(getattr(cfg, "model", None), "max_length", 256))
    preprocess = create_preprocess_fn(tokenizer, max_length=max_length)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_eval_batch_size=int(
            getattr(getattr(cfg, "train", None), "batch_size", 32),
        ),
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=getattr(cfg, "random_seed", 42),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        args=training_args,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(eval_ds)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"[evaluate] wrote metrics to {metrics_path}")

    maybe_plot_confusion(metrics, cm_path)
    if os.path.exists(cm_path):
        print(f"[evaluate] wrote confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    main()
