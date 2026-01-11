"""Data access, validation, and streaming utilities."""

from __future__ import annotations

import fnmatch
import logging
import os
import random
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .utils import stable_id

__all__ = [
    "validate_dataframe",
    "ensure_no_leakage_between_train_eval",
    "load_dev",
    "load_frozen_eval",
    "build_balanced_train_from_hf_stream",
    "load_local_parquet",
]

try:  # Optional dependency for Hugging Face datasets
    from datasets import Dataset, Features, Value, load_dataset
except ImportError:  # pragma: no cover - exercised in environments without datasets
    Dataset = None  # type: ignore
    Features = None  # type: ignore
    Value = None  # type: ignore

    def load_dataset(*args, **kwargs):  # type: ignore
        raise ImportError("datasets is required for streaming and HF parquet loading")


logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ["text", "rating", "title", "asin", "product_category"]
_WS = re.compile(r"\s+", flags=re.U)


def _ensure_features() -> Tuple[Any, Any, Any]:
    if Features is None or Value is None or Dataset is None:  # pragma: no cover - import guard
        raise ImportError("datasets is required for Hugging Face dataset conversion")
    return Dataset, Features, Value


def validate_dataframe(
    df: pd.DataFrame,
    *,
    rating_range: Tuple[float, float] = (1.0, 5.0),
    allow_empty_title: bool = True,
) -> Dict[str, Any]:
    """Validate the unified review schema and return a diagnostics report."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    report: Dict[str, Any] = {"rows": int(len(df)), "columns_present": True}

    rating_series = pd.to_numeric(df["rating"], errors="coerce")
    r_min, r_max = rating_range
    invalid_rating = rating_series.isna() | (rating_series < r_min) | (rating_series > r_max)
    if invalid_rating.any():
        bad_index = invalid_rating[invalid_rating].index.tolist()[:5]
        raise ValueError(f"rating out of range for rows {bad_index}")
    report["rating_range_ok"] = True

    text_nonempty = df["text"].astype(str).str.strip()
    if (text_nonempty == "").any():
        empty_idx = text_nonempty[text_nonempty == ""].index.tolist()[:5]
        raise ValueError(f"text column contains empty entries at rows {empty_idx}")
    report["text_nonempty"] = True

    if not allow_empty_title:
        title_nonempty = df["title"].astype(str).str.strip()
        if (title_nonempty == "").any():
            empty_idx = title_nonempty[title_nonempty == ""].index.tolist()[:5]
            raise ValueError(f"title column contains empty entries at rows {empty_idx}")

    # Ensure categorical metadata are strings
    for column in ("asin", "product_category"):
        if not df[column].astype(str).map(lambda x: isinstance(x, str)).all():
            raise ValueError(f"column {column} must contain string-like values")
    report["metadata_types_ok"] = True

    report["categories_present"] = df["product_category"].nunique() > 0
    return report


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, SimpleNamespace):
        return {k: getattr(obj, k) for k in vars(obj)}
    raise TypeError(f"Unsupported row type for synthetic dataset: {type(obj)!r}")


def _to_hf(df: pd.DataFrame, with_label: bool = False) -> "Dataset":
    DatasetCls, FeaturesCls, ValueCls = _ensure_features()

    base = {
        "text": ValueCls("string"),
        "rating": ValueCls("float32"),
        "title": ValueCls("string"),
        "asin": ValueCls("string"),
        "product_category": ValueCls("string"),
    }
    if with_label:
        base["label"] = ValueCls("int64")

    for column in base.keys():
        if column not in df.columns:
            if column == "rating":
                df[column] = 0.0
            elif column == "label":
                df[column] = 0
            else:
                df[column] = ""

    features = FeaturesCls(base)
    return DatasetCls.from_pandas(
        df.reset_index(drop=True),
        features=features,
        preserve_index=False,
    )


def _build_synthetic_rows(
    rows: Sequence[Dict[str, Any]], *, with_label: bool
) -> Tuple[pd.DataFrame, "Dataset"]:
    normalized = [_as_dict(row) for row in rows]
    df = pd.DataFrame(normalized)

    if df.empty:
        schema = {
            column: pd.Series(
                dtype="float64" if column == "rating" else "object"
            )
            for column in _REQUIRED_COLUMNS
        }
        if with_label:
            schema["label"] = pd.Series(dtype="int64")
        df = pd.DataFrame(schema)
    else:
        for column in _REQUIRED_COLUMNS:
            if column not in df.columns:
                if column == "rating":
                    df[column] = [0.0] * len(df)
                elif column == "title":
                    df[column] = ""
                else:
                    df[column] = [""] * len(df)

    if with_label:
        if "label" not in df.columns or len(df["label"]) == 0:
            df["label"] = [_label_from_rating(value) for value in df.get("rating", [])]

    dataset = _to_hf(df, with_label=with_label)
    return df, dataset


def _node_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _download_parquet_files_from_hf(
    repo_id: str,
    glob_pattern: str,
    revision: str = "main",
    repo_type: str = "dataset",
) -> List[str]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub not installed: pip install huggingface_hub") from exc

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type=repo_type)
    matches = [path for path in files if fnmatch.fnmatch(path, glob_pattern)]
    if not matches:
        raise FileNotFoundError(f"no files match '{glob_pattern}' in repo {repo_id}@{revision}")

    local_paths: List[str] = []
    for filename in matches:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=repo_type,
        )
        local_paths.append(local)
    return local_paths


def _load_hf_parquet_dataset(repo_id: str, glob_pattern: str, revision: str = "main") -> "Dataset":
    paths = _download_parquet_files_from_hf(repo_id, glob_pattern, revision=revision)
    ds = load_dataset("parquet", data_files=paths, split="train")
    for column in _REQUIRED_COLUMNS:
        if column not in ds.column_names:
            filler: Sequence[Any]
            if column == "rating":
                filler = [0.0 for _ in range(len(ds))]
            else:
                filler = ["" for _ in range(len(ds))]
            ds = ds.add_column(column, filler)
    return ds


def load_local_parquet(path: str, *, with_label: bool = True) -> "Dataset":
    """Load a local parquet file into a Hugging Face Dataset with required columns."""

    if not path:
        raise ValueError("path must be provided for local parquet loading")
    if not os.path.exists(path):
        raise FileNotFoundError(f"local parquet not found: {path}")

    df = pd.read_parquet(path)
    return _to_hf(df, with_label=with_label)


def load_dev(dev_cfg, block_ids: Optional[Sequence[str]] = None) -> "Dataset":
    source = _node_get(dev_cfg, "source", None)
    block_set = set(block_ids or [])

    if source == "hf_hub_parquet":
        repo = _node_get(dev_cfg, "hf_repo_id", None)
        glob = _node_get(dev_cfg, "hf_glob", None)
        revision = _node_get(dev_cfg, "hf_revision", "main")
        dataset = _load_hf_parquet_dataset(repo, glob, revision=revision)
        max_rows = int(_node_get(dev_cfg, "max_rows", 0) or 0)
        if max_rows > 0 and len(dataset) > max_rows:
            dataset = dataset.select(range(max_rows))
        if block_set:
            dataset = dataset.filter(
                lambda row: stable_id(row.get("text", ""), row.get("asin", "")) not in block_set
            )
        return dataset

    if source == "synthetic":
        rows = _node_get(dev_cfg, "rows", None) or []
        filtered: List[Dict[str, Any]] = []
        for raw in rows:
            entry = _as_dict(raw)
            rid = stable_id(str(entry.get("text", "")), str(entry.get("asin", "")))
            if rid in block_set:
                continue
            filtered.append(entry)
        df, dataset = _build_synthetic_rows(filtered, with_label=True)
        max_rows = int(_node_get(dev_cfg, "max_rows", 0) or 0)
        if max_rows > 0 and len(df) > max_rows:
            dataset = _to_hf(df.iloc[:max_rows], with_label=True)
        return dataset

    empty_df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    return _to_hf(empty_df)


def load_frozen_eval(eval_cfg, block_ids: Optional[Sequence[str]] = None) -> "Dataset":
    del block_ids  # frozen evaluation never filters by blocklist

    source = _node_get(eval_cfg, "source", None)
    if source == "hf_hub_parquet":
        repo = _node_get(eval_cfg, "hf_repo_id", None)
        glob = _node_get(eval_cfg, "hf_glob", None)
        revision = _node_get(eval_cfg, "hf_revision", "main")
        dataset = _load_hf_parquet_dataset(repo, glob, revision=revision)
        return dataset

    if source == "synthetic":
        rows = _node_get(eval_cfg, "rows", None) or []
        _df, dataset = _build_synthetic_rows(rows, with_label=True)
        return dataset

    empty_df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    return _to_hf(empty_df)


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(_WS.sub(" ", text).strip().split())


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


def _iter_hf_amazon_stream(categories, seed, buffer_size) -> Iterable[Dict[str, Any]]:
    """Stream ``McAuley-Lab/Amazon-Reviews-2023`` splits with approximate shuffling."""

    for category in categories:
        cfg = f"raw_review_{category}"
        stream = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            cfg,
            streaming=True,
            trust_remote_code=True,
        )["full"].shuffle(seed=seed, buffer_size=buffer_size)
        for example in stream:
            yield {
                "text": example.get("text", "") or "",
                "rating": example.get("rating", 0.0),
                "title": example.get("title", "") or "",
                "asin": example.get("asin", "") or "",
                "product_category": category,
            }


def build_balanced_train_from_hf_stream(
    cfg, block_ids: Optional[Sequence[str]]
) -> Tuple["Dataset", Dict[str, int]]:
    source = str(_node_get(cfg, "source", "hf_stream") or "hf_stream").lower()
    block_set = set(block_ids or [])

    if source == "synthetic":
        rows = _node_get(cfg, "rows", None) or []
        dedup = bool(_node_get(cfg, "dedup", True))
        filtered: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw in rows:
            entry = _as_dict(raw)
            rid = stable_id(str(entry.get("text", "")), str(entry.get("asin", "")))
            if rid in block_set or (dedup and rid in seen_ids):
                continue
            if "label" not in entry:
                entry["label"] = _label_from_rating(entry.get("rating"))
            filtered.append(entry)
            if dedup:
                seen_ids.add(rid)

        df, dataset = _build_synthetic_rows(filtered, with_label=True)
        counts = Counter(df.get("label", []))
        class_counts = {str(cls): int(counts.get(cls, 0)) for cls in (0, 1, 2)}
        logger.info("Constructed train dataset from synthetic rows: %s", class_counts)
        return dataset, class_counts

    categories = list(getattr(cfg, "categories", []) or [])
    if not categories:
        raise ValueError("train_stream.categories must not be empty")

    per_class_target = int(getattr(cfg, "per_class_target", 100000))
    seed = int(getattr(cfg, "seed", 42))
    shuffle_buffer = int(getattr(cfg, "hf_shuffle_buffer", 10000))
    dedup = bool(getattr(cfg, "dedup", True))
    min_words = int(getattr(cfg, "sample_text_min_words", 0))
    stop_when_full = bool(getattr(cfg, "stop_when_full", True))

    rng = random.Random(seed)
    buckets: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}
    seen_counts = {0: 0, 1: 0, 2: 0}
    seen_ids: set[str] = set()

    def _reservoir_update(
        buffer: List[Dict[str, Any]],
        target: int,
        item: Dict[str, Any],
        seen_n: int,
    ) -> None:
        if len(buffer) < target:
            buffer.append(item)
            return
        idx = rng.randint(0, seen_n - 1)
        if idx < target:
            buffer[idx] = item

    for row in _iter_hf_amazon_stream(categories, seed, shuffle_buffer):
        text = row["text"]
        if min_words and _word_count(text) < min_words:
            continue

        rid = stable_id(text, row["asin"])
        if rid in block_set or (dedup and rid in seen_ids):
            continue

        label = _label_from_rating(row["rating"])
        seen_counts[label] += 1
        example = dict(row)
        example["label"] = label
        _reservoir_update(buckets[label], per_class_target, example, seen_counts[label])

        if dedup:
            seen_ids.add(rid)

        if stop_when_full and all(len(buckets[idx]) >= per_class_target for idx in (0, 1, 2)):
            break

    collected = buckets[0] + buckets[1] + buckets[2]
    rng.shuffle(collected)

    df = pd.DataFrame(collected)
    dataset = _to_hf(df, with_label=True)
    class_counts = {str(cls): len(rows) for cls, rows in buckets.items()}
    logger.info("Constructed train dataset via reservoir sampling: %s", class_counts)
    return dataset, class_counts


def ensure_no_leakage_between_train_eval(
    train_dataset: "Dataset",
    eval_dataset: "Dataset",
) -> Dict[str, int]:
    """Guard against train/eval overlap by comparing IDs or stable hashes."""

    def _ids(ds: "Dataset") -> set[str]:
        columns = set(ds.column_names)
        if "id" in columns:
            return {str(value) for value in ds["id"]}

        texts = ds["text"] if "text" in columns else ["" for _ in range(len(ds))]
        asins = ds["asin"] if "asin" in columns else ["" for _ in range(len(ds))]
        return {stable_id(str(text or ""), str(asin or "")) for text, asin in zip(texts, asins)}

    train_ids = _ids(train_dataset)
    eval_ids = _ids(eval_dataset)
    shared = train_ids.intersection(eval_ids)
    if shared:
        sample = list(shared)[:5]
        raise ValueError(f"data leakage detected: {len(shared)} shared IDs (e.g. {sample})")

    return {"train_unique": len(train_ids), "eval_rows": len(eval_ids)}
