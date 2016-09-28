from datetime import datetime

import numpy
from matplotlib import pyplot
import pandas

import pytest

from wqio import validate
from wqio.tests import helpers


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


@pytest.fixture
def multiindex_df():
    dates = range(5)
    params = list('ABCDE')
    locations = ['Inflow', 'Outflow']
    index = pandas.MultiIndex.from_product(
        [dates, params, locations],
        names=['date', 'param', 'loc']
    )
    data = pandas.DataFrame(numpy.random.normal(size=len(index)), index=index)
    return data


@pytest.mark.parametrize(('level', 'expected'), [(0, 0), (None, None)])
@helpers.seed
def test_getUniqueDataframeIndexVal(multiindex_df, level, expected):
    if expected is None:
        data = multiindex_df.copy()
        with pytest.raises(ValueError):
            validate.single_value_in_index(data, 'date')
    else:
        data = multiindex_df.select(lambda row: row[level] in [expected])
        test = validate.single_value_in_index(data, 'date')
        assert test == expected


@pytest.mark.parametrize(('value', 'expected'), [
    (None, []),
    ('', []),
    (1, [1]),
    ('abc', ['abc']),
    (numpy.array([1, 2, 3]), [1, 2, 3]),
    ([1, 2, 3], [1, 2, 3])
])
def test_at_least_empty_list(value, expected):
    result = validate.at_least_empty_list(value)
    assert result == expected


@pytest.mark.parametrize(('value', 'options', 'expected'), [
    (None, None, {}),
    ('', None, {}),
    ({'a': 1, 'b': 'bee'}, None, {'a': 1, 'b': 'bee'}),
    (None, {'a': 2}, {'a': 2}),
    ('', {'a': 2}, {'a': 2}),
    ({'a': 1, 'b': 'bee'}, {'a': 2}, {'a': 2, 'b': 'bee'}),
])
def test_at_least_empty_dict(value, options, expected):
    if expected is None:
        with pytest.raises(ValueError):
            validate.at_least_empty_dict(value)

    if options is None:
        result = validate.at_least_empty_dict(value)
    else:
        result = validate.at_least_empty_dict(value, **options)

    assert result == expected


@pytest.mark.parametrize(('value', 'should_error'), [
    ('x', False),
    ('y', False),
    ('both', False),
    (None, False),
    ('abc', True),
    (1, True),
])
def test_fit_arguments(value, should_error):
    if should_error:
        with pytest.raises(ValueError):
            validate.fit_arguments(value, 'example arg name')
    else:
        result = validate.fit_arguments(value, 'example arg name')
        assert result == value
