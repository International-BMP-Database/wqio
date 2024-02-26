from datetime import datetime

import numpy
import pandas
import pytest
from matplotlib import pyplot

from wqio import validate
from wqio.tests import helpers


@pytest.mark.parametrize(
    ("fname", "expected", "error_to_raise"),
    [
        ("BMPdata", "bmpdata.zip", None),
        ("bmpdata.csv", "bmpdata.zip", None),
        ("bmpdata.zip", "bmpdata.zip", None),
        ("NSQD", "nsqd.zip", None),
        ("NSQD.csv", "nsqd.zip", None),
        ("NsqD.zip", "nsqd.zip", None),
        ("CvC", "cvc.zip", None),
        ("CVC.csv", "cvc.zip", None),
        ("cVc.zip", "cvc.zip", None),
        ("junk", None, ValueError),
    ],
)
def test_dataset(fname, expected, error_to_raise):
    with helpers.raises(error_to_raise):
        assert expected == validate.dataset(fname)


@pytest.mark.parametrize(
    ("value", "expected", "error_to_raise"),
    [
        (datetime(2015, 1, 5), pandas.Timestamp("2015-01-05"), None),
        ("2015-01-05", pandas.Timestamp("2015-01-05"), None),
        (pandas.Timestamp("2015-01-05"), pandas.Timestamp("2015-01-05"), None),
        ("junk;", None, ValueError),
    ],
)
def test_timestamp(value, expected, error_to_raise):
    with helpers.raises(error_to_raise):
        assert validate.timestamp(value) == expected


def test_axes_invalid():
    with helpers.raises(ValueError):
        validate.axes("junk")


def test_axes_with_ax():
    fig, ax = pyplot.subplots()
    fig1, ax1 = validate.axes(ax)
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)
    assert ax1 is ax
    assert fig1 is fig


def test_axes_with_None():
    fig1, ax1 = validate.axes(None)
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)


def test_axes_fallback_current():
    fig, ax = pyplot.subplots()
    fig1, ax1 = validate.axes(None, fallback="current")
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)
    assert ax1 is ax
    assert fig1 is fig


@pytest.fixture
def multiindex_df():
    dates = range(5)
    params = list("ABCDE")
    locations = ["Inflow", "Outflow"]
    index = pandas.MultiIndex.from_product(
        [dates, params, locations], names=["date", "param", "loc"]
    )
    data = pandas.DataFrame(numpy.random.normal(size=len(index)), index=index)
    return data


@pytest.mark.parametrize(
    ("level", "expected", "error_to_raise"),
    [("param", "A", None), ("date", None, ValueError)],
)
@helpers.seed
def test_getUniqueDataframeIndexVal(multiindex_df, level, expected, error_to_raise):
    data = multiindex_df.xs("A", level="param", drop_level=False)
    with helpers.raises(error_to_raise):
        result = validate.single_value_in_index(data, level)
        assert result == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        (1, [1]),
        ("abc", ["abc"]),
        (numpy.array([1, 2, 3]), [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        (0, [0]),
    ],
)
def test_at_least_empty_list(value, expected):
    result = validate.at_least_empty_list(value)
    assert result == expected


@pytest.mark.parametrize(
    ("value", "options", "expected", "error_to_raise"),
    [
        (None, None, {}, None),
        ("", None, {}, None),
        ({"a": 1, "b": "bee"}, None, {"a": 1, "b": "bee"}, None),
        (None, {"a": 2}, {"a": 2}, None),
        ("", {"a": 2}, {"a": 2}, None),
        ({"a": 1, "b": "bee"}, {"a": 2}, {"a": 2, "b": "bee"}, None),
        ("a", {"b": 1}, None, ValueError),
    ],
)
def test_at_least_empty_dict(value, options, expected, error_to_raise):
    with helpers.raises(error_to_raise):
        if options is None:
            result = validate.at_least_empty_dict(value)
        else:
            result = validate.at_least_empty_dict(value, **options)

        assert result == expected


@pytest.mark.parametrize(
    ("value", "error_to_raise"),
    [
        ("x", None),
        ("y", None),
        ("both", None),
        (None, None),
        ("abc", ValueError),
        (1, ValueError),
    ],
)
def test_fit_arguments(value, error_to_raise):
    with helpers.raises(error_to_raise):
        result = validate.fit_arguments(value, "example arg name")
        assert result == value
