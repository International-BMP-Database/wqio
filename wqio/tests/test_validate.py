from datetime import datetime

from matplotlib import pyplot
import pandas

import pytest

from wqio import validate


@pytest.mark.parametrize(('value', 'expected'), [
    (datetime(2015, 1, 5), pandas.Timestamp('2015-01-05')),
    ('2015-01-05', pandas.Timestamp('2015-01-05')),
    (pandas.Timestamp('2015-01-05'), pandas.Timestamp('2015-01-05')),
    ('junk;', None),
])
def test_timestamp(value, expected):
    if expected is None:
        with pytest.raises(ValueError):
            validate.timestamp(value)
    else:
        assert validate.timestamp(value) == expected


def test_axes_invalid():
    with pytest.raises(ValueError):
        validate.axes('junk')


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
