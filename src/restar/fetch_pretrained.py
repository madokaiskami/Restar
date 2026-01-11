"""Materialize a Hugging Face model locally for offline DVC pipelines."""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Tuple

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .config import load_config
from .utils import setup_logging

logger = logging.getLogger(__name__)


def _sanitize_model_id(model_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-")
    return sanitized or "model"


def _resolve_target_dir(model_cfg) -> Tuple[str, str]:
    pretrained_name = getattr(model_cfg, "pretrained_name", None)
    if not pretrained_name:
        raise ValueError("model.pretrained_name is required to materialize a pretrained checkpoint")

    configured_dir = getattr(model_cfg, "pretrained_local_dir", None)
    if configured_dir:
        return pretrained_name, str(configured_dir)

    sanitized = _sanitize_model_id(pretrained_name)
    return pretrained_name, str(Path("artifacts") / "pretrained" / sanitized)


def _log_saved_files(target_dir: str) -> None:
    total_bytes = 0
    file_count = 0
    for path in sorted(Path(target_dir).rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            file_count += 1
            logger.info("Saved: %s (%.2f KB)", path.relative_to(target_dir), size / 1024)
    logger.info(
        "Materialization complete: %d files, %.2f MB",
        file_count,
        total_bytes / 1024 / 1024,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch and store a pretrained HF model locally.")
    parser.add_argument("--config", required=True, help="Path to the DVC config YAML")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not getattr(cfg, "model", None):
        raise ValueError("config must include a 'model' section with pretrained_name")

    pretrained_name, target_dir = _resolve_target_dir(cfg.model)
    os.makedirs(target_dir, exist_ok=True)

    log_path = os.path.join(target_dir, "fetch.log")
    setup_logging(args.verbose, log_path)
    logger.info("Materializing %s into %s", pretrained_name, os.path.abspath(target_dir))

    num_labels = int(getattr(getattr(cfg, "model", None), "num_labels", 3))
    hf_config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        config=hf_config,
    )

    hf_config.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    _log_saved_files(target_dir)
    print(f"[fetch_model] Saved pretrained artifacts to {target_dir}")


if __name__ == "__main__":
    main()
