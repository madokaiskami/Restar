"""Configuration loader for YAML files with namespace access."""

from types import SimpleNamespace

import yaml


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path: str):
    """Load a UTF-8 (or UTF-8 with BOM) YAML file into nested namespaces."""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    return _to_namespace(data if data is not None else {})
