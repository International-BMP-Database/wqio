import numpy
import pandas
import polars
from polars.testing import assert_frame_equal

from wqio.utils import from_pandas, to_pandas


def test_from_pandas_moves_index_levels_into_columns():
    index = pandas.MultiIndex.from_product([["A", "B"], [1, 2]], names=["param", "bmp"])
    pdf = pandas.DataFrame({"res": [1.0, 2.0, numpy.nan, 4.0]}, index=index)

    result = from_pandas(pdf)
    assert result.columns == ["param", "bmp", "res"]
    assert result["res"].null_count() == 1
    assert result["res"].is_nan().sum() == 0


def test_from_pandas_without_index():
    pdf = pandas.DataFrame({"res": [1.0, 2.0]}, index=["x", "y"])
    assert from_pandas(pdf, include_index=False).columns == ["res"]


def test_to_pandas_round_trip():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None], "c": [1.5, None, 2.5]})
    pdf = to_pandas(df)
    assert isinstance(pdf, pandas.DataFrame)
    assert pdf["a"].tolist() == [1, 2, 3]
    assert pdf["c"].isna().tolist() == [False, True, False]
    assert_frame_equal(from_pandas(pdf, include_index=False), df, check_dtypes=False)


def test_to_pandas_categories_are_ordered():
    df = polars.DataFrame({"size": ["big", "small", "big", "medium"]})
    pdf = to_pandas(df, categories={"size": ["small", "medium", "big"]})
    assert pdf["size"].cat.ordered
    assert pdf["size"].cat.categories.tolist() == ["small", "medium", "big"]


def test_to_pandas_passes_other_objects_through():
    pdf = pandas.DataFrame({"a": [1]})
    assert to_pandas(pdf) is pdf
    assert to_pandas(None) is None
