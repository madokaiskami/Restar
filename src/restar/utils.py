"""Utility helpers shared across the project."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
from typing import Iterable, Optional, Sequence, Set

import numpy as np

_WS = re.compile(r"\s+", flags=re.U)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return the project logger."""

    logger = logging.getLogger("restar")
    level_stream = logging.DEBUG if verbose else logging.INFO

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level_stream)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level_stream)

    return logger


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_id(text: str, asin: str) -> str:
    """Return a deterministic identifier derived from text and ASIN."""

    t = _WS.sub(" ", (text or "").strip()).lower()
    a = (asin or "").strip().lower()
    raw = f"{t}\x01{a}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_id_set(paths: Sequence[str]) -> Set[str]:
    """Load newline-delimited IDs from ``paths`` into a set."""

    found: Set[str] = set()
    for path in paths or []:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                token = line.strip()
                if token:
                    found.add(token)
    return found


def append_ids(path: Optional[str], ids_iterable: Iterable[str]) -> None:
    """Append IDs to ``path`` creating directories as needed."""

    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for token in ids_iterable:
            handle.write(token + "\n")


def load_blocklist_from_hf(
    repo_id: str,
    filename: str,
    revision: str = "main",
    token: Optional[str] = None,
    repo_type: str = "dataset",
) -> Set[str]:
    """Best-effort blocklist download that tolerates old ``huggingface_hub`` versions."""

    logger = logging.getLogger(__name__)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("huggingface_hub import failed; skip HF blocklist: %s", exc)
        return set()

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            repo_type=repo_type,
        )
        return load_id_set([local_path])
    except Exception as exc:  # pragma: no cover - network dependent
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type=repo_type)
            hint = [
                candidate
                for candidate in files
                if candidate.lower().endswith(".txt")
                or filename.split("/")[-1].lower() in candidate.lower()
            ]
            logger.warning(
                "HF blocklist fetch failed: %s@%s:%s. Candidates: %s. Err: %s",
                repo_id,
                revision,
                filename,
                hint[:5],
                exc,
            )
        except Exception as exc_list:  # pragma: no cover - network dependent
            logger.warning(
                "HF blocklist fetch failed and listing failed: %s; %s",
                exc,
                exc_list,
            )
        return set()


def maybe_load_blocklist(cfg_blocklist) -> Set[str]:
    """Load blocklist IDs from local paths and/or the Hugging Face Hub."""

    block_ids: Set[str] = set()
    if not cfg_blocklist:
        return block_ids

    if hasattr(cfg_blocklist, "paths"):
        paths = getattr(cfg_blocklist, "paths") or []
    elif isinstance(cfg_blocklist, dict):
        paths = cfg_blocklist.get("paths", [])
    else:
        paths = []
    block_ids |= load_id_set(paths)

    if hasattr(cfg_blocklist, "hf_repo_id"):
        repo = getattr(cfg_blocklist, "hf_repo_id")
        filename = getattr(cfg_blocklist, "hf_filename", None)
        revision = getattr(cfg_blocklist, "hf_revision", "main")
        token_env = getattr(cfg_blocklist, "hf_token_env", None)
    elif isinstance(cfg_blocklist, dict):
        repo = cfg_blocklist.get("hf_repo_id")
        filename = cfg_blocklist.get("hf_filename")
        revision = cfg_blocklist.get("hf_revision", "main")
        token_env = cfg_blocklist.get("hf_token_env")
    else:
        repo = filename = token_env = None
        revision = "main"

    token = os.environ.get(token_env) if token_env else None
    if repo and filename:
        block_ids |= load_blocklist_from_hf(repo, filename, revision=revision, token=token)

    return block_ids
