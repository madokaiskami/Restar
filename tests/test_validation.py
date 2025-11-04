import pandas as pd
import pytest

from restar.data import validate_dataframe


def test_validate_dataframe_ok():
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "rating": [1, 5],
            "title": ["", ""],
            "asin": ["X", "Y"],
            "product_category": ["C", "C"],
        }
    )
    report = validate_dataframe(df)
    assert report["rows"] == 2
    assert report["rating_range_ok"] is True


def test_validate_dataframe_bad_rating():
    df = pd.DataFrame(
        {
            "text": ["a"],
            "rating": [6],
            "title": [""],
            "asin": ["X"],
            "product_category": ["C"],
        }
    )
    with pytest.raises(ValueError):
        validate_dataframe(df)
