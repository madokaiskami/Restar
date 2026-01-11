"""Training entry point for the Re*Star sentiment model."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import HfApi, snapshot_download
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_callback import TrainerCallback

from .config import load_config
from .data import (
    build_balanced_train_from_hf_stream,
    ensure_no_leakage_between_train_eval,
    load_dev,
    load_frozen_eval,
    load_local_parquet,
)
from .eval_utils import maybe_plot_confusion
from .utils import maybe_load_blocklist, set_seed, setup_logging

try:
    from transformers.training_args import IntervalStrategy, SaveStrategy
except Exception:  # pragma: no cover - compatibility with older transformers
    from transformers.trainer_utils import IntervalStrategy

    SaveStrategy = IntervalStrategy

try:
    from transformers.utils.import_utils import (
        ACCELERATE_MIN_VERSION,
        is_accelerate_available,
    )
except Exception:  # pragma: no cover - older transformers versions
    ACCELERATE_MIN_VERSION = "0.26.0"

    def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION) -> bool:  # type: ignore
        return False


logger = logging.getLogger(__name__)


class PushCheckpointsToHubCallback(TrainerCallback):
    """Upload the latest checkpoint to the Hugging Face Hub after every save."""

    def __init__(self, repo_id: str):
        self.api = HfApi()
        self.repo_id = repo_id

    def on_save(self, args, state, control, **kwargs):  # pragma: no cover - network side effects
        ckpt_dir = _find_latest_checkpoint_dir(args.output_dir)
        if not ckpt_dir:
            logger.warning("[hub] on_save: no checkpoint directory found")
            return
        rel = os.path.basename(ckpt_dir)
        try:
            logger.info("[hub] uploading checkpoint %s -> %s", rel, self.repo_id)
            self.api.upload_folder(
                repo_id=self.repo_id,
                repo_type="model",
                folder_path=ckpt_dir,
                path_in_repo=rel,
                commit_message=f"Add {rel} (step {state.global_step})",
            )
        except Exception as exc:
            logger.warning("[hub] upload checkpoint failed: %s", exc)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = float((preds == labels).mean())
    f1_macro = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds).tolist()
    return {"accuracy": acc, "f1_macro": f1_macro, "confusion_matrix": cm}


class WeightedTrainer(Trainer):
    """Trainer that supports global class weights and per-sample weights."""

    def __init__(self, class_weights=None, sample_weight_column: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.sample_weight_column = sample_weight_column

    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        sample_weight = None
        if self.sample_weight_column and self.sample_weight_column in inputs:
            sample_weight = inputs.pop(self.sample_weight_column)

        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_vec = F.cross_entropy(logits, labels, weight=weight, reduction="none")

        if sample_weight is not None:
            sw = sample_weight.to(loss_vec.device).float()
            loss = (loss_vec * sw).mean()
        else:
            loss = loss_vec.mean()

        return (loss, outputs) if return_outputs else loss


def _as_mapping(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    mapping: Dict[str, Any] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        if callable(value):
            continue
        mapping[attr] = value
    return mapping


def _find_node(cfg, key):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    wanted = key.strip().lower()
    for name in list(_as_mapping(cfg).keys()):
        norm = str(name).strip().lstrip("\ufeff").lower()
        if norm == wanted:
            return getattr(cfg, name) if hasattr(cfg, name) else _as_mapping(cfg)[name]
    return None


def _get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "y", "yes", "on"}


def _find_latest_checkpoint_dir(root: str) -> Optional[str]:
    if not root or not os.path.isdir(root):
        return None
    candidates: list[Tuple[int, str]] = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and entry.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", entry)
            step = int(match.group(1)) if match else -1
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _flatten_config(cfg) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                walk(item, f"{prefix}.{idx}" if prefix else str(idx))
            return
        mapping = _as_mapping(value)
        if mapping and not isinstance(value, (str, int, float, bool)):
            for key, item in mapping.items():
                walk(item, f"{prefix}.{key}" if prefix else str(key))
            return
        if prefix:
            flattened[prefix] = value

    walk(cfg, "")
    return flattened


def _format_param(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _load_mlflow():
    spec = importlib.util.find_spec("mlflow")
    if spec is None:
        return None
    return importlib.import_module("mlflow")


def _collect_dvc_tags(lock_path: str) -> Dict[str, str]:
    if not os.path.exists(lock_path):
        return {}
    with open(lock_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    tags: Dict[str, str] = {
        "dvc.lock.sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    data = yaml.safe_load(content) or {}
    stages = data.get("stages", {})
    prepare_stage = stages.get("prepare", {})
    for out in prepare_stage.get("outs", []) or []:
        path = out.get("path")
        md5 = out.get("md5")
        if path and md5:
            tags[f"dvc.prepare.outs.{path}.md5"] = str(md5)
    train_stage = stages.get("train", {})
    for out in train_stage.get("outs", []) or []:
        path = out.get("path", "")
        if path.endswith("/model") and out.get("md5"):
            tags["dvc.train.model.md5"] = str(out["md5"])
            break
    return tags


def _log_metrics_from_file(mlflow, path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    if metrics:
        mlflow.log_metrics(metrics)
    skipped = [k for k, v in payload.items() if not isinstance(v, (int, float))]
    if skipped:
        logger.info("Skipping non-scalar MLflow metrics from %s: %s", path, skipped)


def _plot_confusion_from_metrics_file(metrics_path: str, out_path: str) -> bool:
    if not metrics_path or not os.path.exists(metrics_path):
        return False
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception as exc:
        logger.info("Skipping confusion matrix plot from %s: %s", metrics_path, exc)
        return False
    maybe_plot_confusion(metrics, out_path)
    return os.path.exists(out_path)


def _resolve_init_from_hub(cfg_model) -> Tuple[Optional[str], Optional[str]]:
    if not _truthy(_get(cfg_model, "init_from_hub", False)):
        return None, None
    repo_id = _get(cfg_model, "hub_repo_id", None) or _get(cfg_model, "hf_init_repo_id", None)
    if not repo_id:
        logger.warning("init_from_hub=True but model.hub_repo_id is missing; skipping")
        return None, None
    revision = _get(cfg_model, "hub_revision", "main")
    logger.info("[hub-init] snapshot_download %s@%s", repo_id, revision)
    local = snapshot_download(repo_id=repo_id, revision=revision, repo_type="model")
    ckpt = _find_latest_checkpoint_dir(local)
    if ckpt:
        logger.info("[hub-init] found latest checkpoint: %s", ckpt)
        return ckpt, ckpt
    logger.info("[hub-init] no checkpoints found; using repo root: %s", local)
    return local, None


def _resolve_pretrained_local(cfg_model, pretrained_name: str) -> Optional[str]:
    local_dir = _get(cfg_model, "pretrained_local_dir", None)
    if local_dir and os.path.isdir(str(local_dir)):
        resolved = os.path.abspath(str(local_dir))
        logger.info("Using local pretrained directory: %s", resolved)
        return str(local_dir)
    if local_dir:
        logger.info(
            "Configured pretrained_local_dir %s not found; falling back to %s",
            local_dir,
            pretrained_name,
        )
    return None


def _maybe_truncate_dataset(dataset, max_rows: int, name: str):
    try:
        limit = int(max_rows or 0)
    except Exception:
        return dataset
    if limit <= 0:
        return dataset
    if not hasattr(dataset, "select"):
        logger.warning("Dataset %s does not support select(); skipping truncation", name)
        return dataset
    try:
        total = len(dataset)
    except Exception:
        total = None
    if total is not None and total <= limit:
        return dataset
    truncated = dataset.select(range(limit))
    try:
        size = len(truncated)
    except Exception:
        size = limit
    logger.info("Truncated %s dataset to %d rows", name, size)
    return truncated


def create_preprocess_fn(tokenizer, max_length: int = 256):
    def preprocess(batch):
        texts = batch["text"]
        encoded = tokenizer(texts, truncation=True, max_length=max_length)

        if "label" in batch and batch["label"] is not None:
            labels = []
            ratings = batch.get("rating", [0.0] * len(texts))
            for idx, value in enumerate(batch["label"]):
                try:
                    labels.append(int(value))
                except Exception:
                    rating = ratings[idx] if idx < len(ratings) else 0.0
                    labels.append(0 if float(rating) <= 2 else (2 if float(rating) >= 4 else 1))
        else:
            ratings = batch.get("rating", [0.0] * len(texts))
            labels = []
            for rating in ratings:
                try:
                    rating_val = float(rating)
                except Exception:
                    rating_val = 0.0
                labels.append(0 if rating_val <= 2 else (2 if rating_val >= 4 else 1))

        encoded["labels"] = labels
        return encoded

    return preprocess


def resolve_class_weights(train_cfg, class_counts: Dict[str, int]):
    configured = getattr(train_cfg, "class_weights", []) if train_cfg else []
    if configured:
        return configured
    counts = np.array(
        [max(class_counts.get(str(i), 1), 1) for i in (0, 1, 2)],
        dtype=np.float64,
    )
    inverse = 1.0 / counts
    weights = (inverse / inverse.mean()).tolist()
    logger.info("Auto-derived class weights: %s", weights)
    return weights


def _resolve_strategy(value, enum_cls, default):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except Exception:
        normalized = str(value).strip().upper()
        try:
            return enum_cls[normalized]
        except Exception:
            return default


def resolve_training_params(cfg, output_dir: str, push_to_hub: bool) -> Dict[str, Any]:
    train_cfg = getattr(cfg, "train", None)
    train_batch = _get(train_cfg, "batch_size", None)
    if train_batch is None:
        train_batch = _get(train_cfg, "per_device_train_batch_size", 32)
    eval_batch = _get(train_cfg, "batch_size", None)
    if eval_batch is None:
        eval_batch = _get(train_cfg, "per_device_eval_batch_size", train_batch)

    save_strategy_cfg = _resolve_strategy(
        _get(train_cfg, "save_strategy", None),
        SaveStrategy,
        SaveStrategy.EPOCH,
    )
    eval_strategy_cfg = _resolve_strategy(
        _get(train_cfg, "eval_strategy", None),
        IntervalStrategy,
        IntervalStrategy.EPOCH,
    )

    load_best_model = _truthy(_get(train_cfg, "load_best_model_at_end", True))
    if load_best_model:
        if save_strategy_cfg == getattr(SaveStrategy, "NO", None):
            logger.info(
                "Disabling load_best_model_at_end because save_strategy is set to 'no'",
            )
            load_best_model = False
        elif eval_strategy_cfg == getattr(IntervalStrategy, "NO", None):
            logger.info(
                "Disabling load_best_model_at_end because eval_strategy is set to 'no'",
            )
            load_best_model = False
        elif save_strategy_cfg != eval_strategy_cfg:
            logger.info(
                "Disabling load_best_model_at_end: save_strategy (%s) != eval_strategy (%s)",
                save_strategy_cfg,
                eval_strategy_cfg,
            )
            load_best_model = False

    save_total_limit = _get(train_cfg, "save_total_limit", 2)
    if save_total_limit is not None:
        try:
            save_total_limit = int(save_total_limit)
        except Exception:
            save_total_limit = 0
    save_total_limit = int(save_total_limit or 0) or None

    params: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(train_batch),
        "per_device_eval_batch_size": int(eval_batch),
        "learning_rate": _get(train_cfg, "learning_rate", 3e-5),
        "num_train_epochs": _get(train_cfg, "num_train_epochs", 3),
        "weight_decay": _get(train_cfg, "weight_decay", 0.01),
        "warmup_ratio": _get(train_cfg, "warmup_ratio", 0.0),
        "eval_strategy": eval_strategy_cfg,
        "save_strategy": save_strategy_cfg,
        "save_total_limit": save_total_limit,
        "logging_steps": int(_get(train_cfg, "logging_steps", 50)),
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "seed": getattr(cfg, "random_seed", 42),
        "report_to": [],
        "push_to_hub": push_to_hub,
        "hub_model_id": _get(train_cfg, "hub_model_id", None) if push_to_hub else None,
        "hub_strategy": _get(train_cfg, "hub_strategy", "every_save") if push_to_hub else "end",
        "hub_private_repo": (
            bool(_get(train_cfg, "hub_private_repo", False)) if push_to_hub else None
        ),
    }

    return params


def build_training_arguments(
    cfg,
    output_dir: str,
    push_to_hub: bool,
) -> TrainingArguments:
    params = resolve_training_params(cfg, output_dir, push_to_hub)
    return TrainingArguments(**params)


def _trainer_backend_available() -> bool:
    try:
        return is_accelerate_available()
    except Exception:  # pragma: no cover - defensive fallback
        return False


def _evaluate_model(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
    class_weights: Optional[list[float]] = None,
    prefix: str = "eval",
) -> Dict[str, Any]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    weight_tensor = None
    if class_weights:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels")
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            logits_chunks.append(logits.detach().cpu().numpy())
            labels_chunks.append(labels.detach().cpu().numpy())

            loss_vec = F.cross_entropy(
                logits,
                labels,
                weight=weight_tensor,
                reduction="none",
            )
            total_loss += float(loss_vec.sum().item())
            total_examples += int(labels.size(0))

    if logits_chunks:
        stacked_logits = np.concatenate(logits_chunks, axis=0)
        stacked_labels = np.concatenate(labels_chunks, axis=0)
    else:
        stacked_logits = np.zeros((0, model.num_labels), dtype=np.float32)
        stacked_labels = np.zeros((0,), dtype=np.int64)

    metrics = compute_metrics((stacked_logits, stacked_labels))
    metrics_prefixed = {f"{prefix}_{key}": value for key, value in metrics.items()}
    avg_loss = total_loss / total_examples if total_examples else 0.0
    metrics_prefixed[f"{prefix}_loss"] = avg_loss
    model.train()
    return metrics_prefixed


def run_basic_training_loop(
    model: AutoModelForSequenceClassification,
    tokenizer,
    train_ds,
    dev_ds,
    frozen_eval_raw,
    preprocess_fn,
    params: Dict[str, Any],
    class_weights,
    sample_weight_column: str,
    out_dir: str,
    model_dir: str,
) -> None:
    logger.warning(
        "Accelerate >=%s is unavailable; using basic PyTorch training loop.",
        ACCELERATE_MIN_VERSION,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(params["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        dev_ds,
        batch_size=int(params["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    total_steps = max(len(train_loader) * int(params["num_train_epochs"]), 1)
    warmup_steps = int(total_steps * float(params.get("warmup_ratio", 0.0)))
    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        if total_steps > 0
        else None
    )

    weight_tensor = None
    if class_weights:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    sample_weight_column = sample_weight_column or ""

    for epoch in range(int(params["num_train_epochs"])):
        model.train()
        running_loss = 0.0
        running_examples = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels")
            sample_weight = None
            if sample_weight_column and sample_weight_column in batch:
                sample_weight = batch.pop(sample_weight_column)

            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits

            loss_vec = F.cross_entropy(
                logits,
                labels,
                weight=weight_tensor,
                reduction="none",
            )
            if sample_weight is not None:
                sw = sample_weight.to(device).float()
                loss = (loss_vec * sw).mean()
            else:
                loss = loss_vec.mean()

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * labels.size(0)
            running_examples += int(labels.size(0))

            logging_steps = int(params.get("logging_steps", 0))
            if logging_steps > 0 and step % logging_steps == 0:
                avg_loss = running_loss / max(running_examples, 1)
                logger.info(
                    "Epoch %d step %d/%d - running loss %.4f",
                    epoch + 1,
                    step,
                    len(train_loader),
                    avg_loss,
                )

        epoch_loss = running_loss / max(running_examples, 1)
        logger.info("Epoch %d completed - train loss %.4f", epoch + 1, epoch_loss)

    metrics_dev = _evaluate_model(
        model,
        eval_loader,
        device,
        class_weights,
        prefix="eval",
    )

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(os.path.join(out_dir, "metrics_dev.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_dev, handle, ensure_ascii=False, indent=2)

    if frozen_eval_raw is not None:
        frozen_eval = frozen_eval_raw.map(
            preprocess_fn,
            batched=True,
            remove_columns=frozen_eval_raw.column_names,
        )
        frozen_loader = DataLoader(
            frozen_eval,
            batch_size=int(params["per_device_eval_batch_size"]),
            shuffle=False,
            collate_fn=collator,
        )
        metrics_eval = _evaluate_model(
            model,
            frozen_loader,
            device,
            class_weights,
            prefix="eval",
        )
        with open(
            os.path.join(out_dir, "metrics_eval.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metrics_eval, handle, ensure_ascii=False, indent=2)
    else:
        logger.warning("Frozen eval skipped: dataset unavailable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = getattr(cfg, "run_name", "restar_run")
    output_root = getattr(cfg, "output_dir", "outputs")
    out_dir = os.path.join(output_root, run_name)
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    setup_logging(args.verbose, os.path.join(out_dir, "train.log"))
    logger.info("Loading configuration from %s", os.path.abspath(args.config))

    set_seed(getattr(cfg, "random_seed", 42))

    block_ids = maybe_load_blocklist(getattr(cfg, "blocklist", None))
    if block_ids:
        logger.info("Loaded %d blocklisted IDs", len(block_ids))

    train_stream_cfg = _find_node(cfg, "train_stream")
    if train_stream_cfg is None:
        raise ValueError("Configuration is missing the 'train_stream' section")

    train_local_path = _get(train_stream_cfg, "local_parquet", None)
    if train_local_path and os.path.exists(str(train_local_path)):
        train_ds = load_local_parquet(str(train_local_path), with_label=True)
        class_counts = Counter(train_ds["label"]) if "label" in train_ds.column_names else {}
        logger.info(
            "Loaded train dataset from local parquet %s with counts=%s",
            train_local_path,
            class_counts,
        )
    else:
        if train_local_path:
            logger.warning(
                "Train parquet %s not found; falling back to streaming config",
                train_local_path,
            )
        train_ds, class_counts = build_balanced_train_from_hf_stream(
            train_stream_cfg,
            block_ids=block_ids,
        )
        logger.info("Per-class counts from streaming sampler: %s", class_counts)

    train_max_rows = int(_get(train_stream_cfg, "max_rows", 0) or 0)
    if train_max_rows > 0:
        train_ds = _maybe_truncate_dataset(train_ds, train_max_rows, "train")
        if "label" in train_ds.column_names:
            class_counts = Counter(train_ds["label"])

    class_weights = resolve_class_weights(getattr(cfg, "train", None), class_counts)

    dev_local_path = _get(cfg.dev, "local_parquet", None)
    if dev_local_path and os.path.exists(str(dev_local_path)):
        dev_ds = load_local_parquet(str(dev_local_path), with_label=True)
        logger.info("Loaded dev dataset from local parquet %s", dev_local_path)
    else:
        if dev_local_path:
            logger.warning("Dev parquet %s not found; falling back to source loader", dev_local_path)
        dev_ds = load_dev(cfg.dev, block_ids=None)
    dev_max_rows = int(_get(cfg.dev, "max_rows", 0) or 0)
    if dev_max_rows > 0:
        dev_ds = _maybe_truncate_dataset(dev_ds, dev_max_rows, "dev")

    eval_cfg = getattr(cfg, "eval", None)
    frozen_eval_raw = None
    if eval_cfg is not None:
        eval_local_path = _get(eval_cfg, "local_parquet", None)
        if eval_local_path and os.path.exists(str(eval_local_path)):
            try:
                frozen_eval_raw = load_local_parquet(str(eval_local_path), with_label=True)
                logger.info("Loaded eval dataset from local parquet %s", eval_local_path)
            except Exception as exc:  # pragma: no cover - depends on local files
                logger.warning("Local eval parquet load failed: %s", exc)
        else:
            if eval_local_path:
                logger.warning(
                    "Eval parquet %s not found; falling back to source loader",
                    eval_local_path,
                )
            try:
                frozen_eval_raw = load_frozen_eval(eval_cfg, block_ids=None)
            except Exception as exc:  # pragma: no cover - depends on external data
                logger.warning("Frozen eval load failed: %s", exc)

        eval_max_rows = int(_get(eval_cfg, "max_rows", 0) or 0)
        if frozen_eval_raw is not None and eval_max_rows > 0:
            frozen_eval_raw = _maybe_truncate_dataset(
                frozen_eval_raw,
                eval_max_rows,
                "eval",
            )

    if frozen_eval_raw is not None:
        guard_stats = ensure_no_leakage_between_train_eval(train_ds, frozen_eval_raw)
        logger.info(
            "Leakage guard passed: %d unique train IDs vs %d eval rows",
            guard_stats["train_unique"],
            guard_stats["eval_rows"],
        )

    model_cfg = getattr(cfg, "model", None)
    init_dir, resume_ckpt = _resolve_init_from_hub(model_cfg)
    pretrained_name = _get(model_cfg, "pretrained_name", "distilbert-base-uncased")
    local_pretrained_dir = _resolve_pretrained_local(model_cfg, pretrained_name)
    num_labels = int(_get(model_cfg, "num_labels", 3))
    max_length = int(_get(model_cfg, "max_length", 256))
    init_source = init_dir or local_pretrained_dir or pretrained_name

    tokenizer = AutoTokenizer.from_pretrained(init_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        init_source,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    preprocess = create_preprocess_fn(tokenizer, max_length=max_length)
    train_ds = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    dev_ds = dev_ds.map(
        preprocess,
        batched=True,
        remove_columns=dev_ds.column_names,
    )
    logger.info("Train columns: %s", train_ds.column_names)
    logger.info("Dev columns: %s", dev_ds.column_names)

    push_to_hub = bool(_get(cfg.train, "push_to_hub", False))
    hub_model_id = _get(cfg.train, "hub_model_id", None)
    if push_to_hub and not hub_model_id:
        logger.warning(
            "push_to_hub=True but train.hub_model_id is missing; disabling push",
        )
        push_to_hub = False

    trainer_supported = _trainer_backend_available()
    if push_to_hub and not trainer_supported:
        logger.warning(
            "push_to_hub=True but Accelerate >=%s is unavailable; disabling push",
            ACCELERATE_MIN_VERSION,
        )
        push_to_hub = False

    mlflow = _load_mlflow()
    mlflow_context = contextlib.nullcontext()
    if mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "restar")
        mlflow.set_experiment(experiment_name)
        transformers_module = getattr(mlflow, "transformers", None)
        if transformers_module and hasattr(transformers_module, "autolog"):
            transformers_module.autolog()
        mlflow_context = mlflow.start_run(run_name=run_name)

    with mlflow_context:
        if mlflow:
            flat_params = {k: _format_param(v) for k, v in _flatten_config(cfg).items()}
            if flat_params:
                mlflow.log_params(flat_params)
            dvc_tags = _collect_dvc_tags(os.path.abspath("dvc.lock"))
            if dvc_tags:
                mlflow.set_tags(dvc_tags)

        if not trainer_supported:
            params = resolve_training_params(cfg, out_dir, push_to_hub=False)
            run_basic_training_loop(
                model,
                tokenizer,
                train_ds,
                dev_ds,
                frozen_eval_raw,
                preprocess,
                params,
                class_weights,
                _get(cfg.train, "sample_weight_column", ""),
                out_dir,
                model_dir,
            )
        else:
            training_args = build_training_arguments(cfg, out_dir, push_to_hub)

            trainer = WeightedTrainer(
                class_weights=class_weights,
                sample_weight_column=_get(cfg.train, "sample_weight_column", ""),
                model=model,
                args=training_args,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                compute_metrics=compute_metrics,
            )

            if push_to_hub and hub_model_id:
                trainer.add_callback(PushCheckpointsToHubCallback(repo_id=hub_model_id))

            patience = int(_get(cfg.train, "early_stopping_patience", 2) or 0)
            if patience > 0:
                trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

            trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

            trainer.save_model(model_dir)
            tokenizer.save_pretrained(model_dir)

            metrics_dev = trainer.evaluate(dev_ds)
            with open(os.path.join(out_dir, "metrics_dev.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics_dev, handle, ensure_ascii=False, indent=2)

            if frozen_eval_raw is not None:
                frozen_eval = frozen_eval_raw.map(
                    preprocess,
                    batched=True,
                    remove_columns=frozen_eval_raw.column_names,
                )
                metrics_eval = trainer.evaluate(frozen_eval)
                with open(
                    os.path.join(out_dir, "metrics_eval.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(metrics_eval, handle, ensure_ascii=False, indent=2)
            else:
                logger.warning("Frozen eval skipped: dataset unavailable")

            if push_to_hub and hub_model_id:
                try:  # pragma: no cover - network side effects
                    trainer.create_model_card(model_name=hub_model_id, language="en")
                except Exception:
                    pass
                trainer.push_to_hub(commit_message="end of training")

        metrics_source = os.path.join(out_dir, "metrics_eval.json")
        if not os.path.exists(metrics_source):
            metrics_source = os.path.join(out_dir, "metrics_dev.json")
        confusion_path = os.path.join(out_dir, "confusion_matrix.png")
        _plot_confusion_from_metrics_file(metrics_source, confusion_path)

        if mlflow:
            _log_metrics_from_file(mlflow, os.path.join(out_dir, "metrics_dev.json"))
            _log_metrics_from_file(mlflow, os.path.join(out_dir, "metrics_eval.json"))
            if os.path.isdir(model_dir):
                mlflow.log_artifacts(model_dir, artifact_path="model")
            train_log = os.path.join(out_dir, "train.log")
            if os.path.exists(train_log):
                mlflow.log_artifact(train_log)
            if os.path.exists(confusion_path):
                mlflow.log_artifact(confusion_path)
            dvc_lock = os.path.abspath("dvc.lock")
            if os.path.exists(dvc_lock):
                mlflow.log_artifact(dvc_lock)


if __name__ == "__main__":
    main()
