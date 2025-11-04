import pytest
from datasets import Dataset

from restar.data import ensure_no_leakage_between_train_eval


def _make_dataset(rows):
    keys = set()
    for row in rows:
        keys.update(row.keys())

    columns = {key: [] for key in keys}
    for row in rows:
        for key in keys:
            columns[key].append(row.get(key, ""))

    for required in ("text", "asin", "rating", "title", "product_category"):
        if required not in columns:
            columns[required] = ["" for _ in rows]

    return Dataset.from_dict(columns)


def test_leakage_guard_passes():
    train_rows = [
        {
            "text": "Great phone",
            "asin": "A1",
            "rating": 5,
            "title": "",
            "product_category": "Electronics",
        },
        {
            "text": "Bad charger",
            "asin": "A2",
            "rating": 1,
            "title": "",
            "product_category": "Electronics",
        },
    ]
    eval_rows = [
        {
            "text": "Average case",
            "asin": "B1",
            "rating": 3,
            "title": "",
            "product_category": "Electronics",
        }
    ]
    stats = ensure_no_leakage_between_train_eval(
        _make_dataset(train_rows), _make_dataset(eval_rows)
    )
    assert stats["train_unique"] == 2
    assert stats["eval_rows"] == 1


def test_leakage_guard_detects_overlap():
    train_rows = [
        {
            "text": "Same text",
            "asin": "C1",
            "rating": 5,
            "title": "",
            "product_category": "Books",
        }
    ]
    eval_rows = [
        {
            "text": "Same text",
            "asin": "C1",
            "rating": 5,
            "title": "",
            "product_category": "Books",
        }
    ]
    with pytest.raises(ValueError):
        ensure_no_leakage_between_train_eval(
            _make_dataset(train_rows), _make_dataset(eval_rows)
        )
