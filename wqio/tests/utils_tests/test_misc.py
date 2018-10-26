from functools import partial
from textwrap import dedent
from io import StringIO


import pytest
import pandas.util.testing as pdtest

import numpy
import pandas

from wqio.utils import misc
from wqio.tests import helpers


@pytest.fixture
def basic_data():
    testcsv = """\
    Date,A,B,C,D
    X,1,2,3,4
    Y,5,6,7,8
    Z,9,0,1,2
    """
    return pandas.read_csv(StringIO(dedent(testcsv)), index_col=['Date'])


@pytest.fixture
def multiindex_df():
    index = pandas.MultiIndex.from_product([['A', 'B', 'C'], ['mg/L']], names=['loc', 'units'])
    return pandas.DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=['a', 'b'])


class mockDataset(object):
    def __init__(self, inflow, outflow):
        self.inflow = mockLocation(inflow)
        self.outflow = mockLocation(outflow)


class mockLocation(object):
    def __init__(self, data):
        self.data = data
        self.stats = mockSummary(data)


class mockSummary(object):
    def __init__(self, data):
        self.N = len(data)
        self.max = max(data)
        self.min = min(data)
        self.nonething = None


def test_add_column_level(basic_data):
    known_cols = pandas.MultiIndex.from_tuples([(u'test', u'A'), (u'test', u'B'),
                                                (u'test', u'C'), (u'test', u'D')])
    newdata = misc.add_column_level(basic_data, 'test', 'testlevel')
    assert known_cols.tolist() == newdata.columns.tolist()

    # can only add levels to non-MultiIndex columns
    with helpers.raises(ValueError):
        misc.add_column_level(newdata, 'test2', 'testlevel2')


@pytest.mark.parametrize('L1', [0, 'loc'])
@pytest.mark.parametrize('L2', [2, 'units'])
def test_swap_column_levels(multiindex_df, L1, L2):
    columns = pandas.MultiIndex.from_product(
        [['A', 'B', 'C'], ['res', 'cen'], ['mg/L']],
        names=['loc', 'value', 'units']
    )
    data = numpy.arange(len(columns) * 10).reshape((10, len(columns)))
    df = pandas.DataFrame(data, columns=columns).pipe(misc.swap_column_levels, L1, L2)

    expected_columns = pandas.MultiIndex.from_product(
        [['mg/L'], ['cen', 'res'], ['A', 'B', 'C']],
        names=['units', 'value', 'loc']
    )

    pdtest.assert_index_equal(df.columns, expected_columns)


def test_flatten_columns(multiindex_df, basic_data):
    expected = ['A_mg/L', 'B_mg/L', 'C_mg/L']
    flat = misc.flatten_columns(multiindex_df.T)
    assert flat.columns.tolist() == expected
    assert misc.flatten_columns(basic_data).columns.tolist() == basic_data.columns.tolist()


def test_expand_columns():
    x = numpy.arange(12).reshape(3, 4)
    df = pandas.DataFrame(x, columns=('A_a', 'A_b', 'B_a', 'B_c'))

    res_cols = pandas.MultiIndex(
        levels=[['A', 'B'], ['a', 'b', 'c']],
        labels=[[0, 0, 1, 1], [0, 1, 0, 2]],
        names=['top', 'bottom']
    )
    expected = pandas.DataFrame(x, columns=res_cols)
    result = misc.expand_columns(df, ['top', 'bottom'])
    pdtest.assert_frame_equal(result, expected)


@pytest.mark.parametrize('criteria', [None, lambda row: row[0] in ['A', 'B']])
@pytest.mark.parametrize('dropold', [True, False])
def test_redefine_index_level(multiindex_df, criteria, dropold):
    expected_cols = ['a', 'b']
    if dropold:
        expected_value = [[1, 2], [3, 4], [5, 6]]
        if criteria:
            expected_index = [('A', 'ug/L'), ('B', 'ug/L'), ('C', 'mg/L')]
        else:
            expected_index = [('A', 'ug/L'), ('B', 'ug/L'), ('C', 'ug/L')]

    else:
        if criteria:
            expected_value = [[1, 2], [1, 2], [3, 4], [3, 4], [5, 6]]
            expected_index = [
                ('A', 'mg/L'), ('A', 'ug/L'),
                ('B', 'mg/L'), ('B', 'ug/L'),
                ('C', 'mg/L')
            ]
        else:
            expected_value = [[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]]
            expected_index = [
                ('A', 'mg/L'), ('A', 'ug/L'),
                ('B', 'mg/L'), ('B', 'ug/L'),
                ('C', 'mg/L'), ('C', 'ug/L')
            ]

    result = misc.redefine_index_level(multiindex_df, 'units', 'ug/L',
                                       criteria=criteria, dropold=dropold)
    expected = pandas.DataFrame(
        data=expected_value,
        index=pandas.MultiIndex.from_tuples(expected_index, names=['loc', 'units']),
        columns=expected_cols
    )
    pdtest.assert_frame_equal(result, expected)


@pytest.fixture
def basic_dataset():
    known_float = 123.4567
    inflow = [1, 3, 4, 12.57]
    outflow = [2, 5, 7, 15.17]
    return mockDataset(inflow, outflow)


def test_nested_getattr(basic_dataset):
    result = misc.nested_getattr(basic_dataset, 'inflow.stats.max')
    expected = basic_dataset.inflow.stats.max
    assert result == expected


@pytest.mark.parametrize(('strformat', 'expected', 'attribute'), [
    ('%d', '4', 'inflow.stats.N'),
    ('%0.2f', '15.17', 'outflow.stats.max'),
    (None, '--', 'inflow.stats.nonething')
])
def test_stringify(basic_dataset, strformat, expected, attribute):
    result = misc.stringify(basic_dataset, strformat, attribute=attribute)
    assert result == expected


def test_categorize_columns():
    csvdata = StringIO(dedent("""\
        parameter,units,season,lower,NSQD Median,upper
        Cadmium (Cd),ug/L,autumn,0.117,0.361,0.52
        Cadmium (Cd),ug/L,spring,0.172,0.352,0.53
        Cadmium (Cd),ug/L,summer,0.304,0.411,0.476
        Cadmium (Cd),ug/L,winter,0.355,0.559,1.125
        Dissolved Chloride (Cl),mg/L,autumn,0.342,2.3,5.8
        Dissolved Chloride (Cl),mg/L,spring,2.5,2.5,2.5
        Dissolved Chloride (Cl),mg/L,summer,0.308,0.762,1.24
        Escherichia coli,MPN/100 mL,autumn,1200.0,15500.0,24000.0
        Escherichia coli,MPN/100 mL,spring,10.0,630.0,810.0
        Escherichia coli,MPN/100 mL,summer,21000.0,27000.0,35000.0
        Escherichia coli,MPN/100 mL,winter,20.0,200.0,800.0
    """))
    df = pandas.read_csv(csvdata)
    df2 = misc.categorize_columns(df, 'parameter', 'units', 'season')

    # check pandas API that the first df has object columns
    assert object in df.dtypes.values

    # confirm that all of those objects are gone
    assert object not in df2.dtypes.values

    with helpers.raises(ValueError):
        misc.categorize_columns(df, 'parameter', 'upper')


@pytest.mark.parametrize(("value", "expected"), [
    (3, '<5'),
    (17, '15 - 20'),
    (25, '20 - 25'),
    (46, '>35'),
])
@pytest.mark.parametrize("units", [None, 'mm'])
def test_classifier(value, units, expected):
    bins = numpy.arange(5, 36, 5)
    if units is not None:
        expected = '{} {}'.format(expected, units)

    result = misc.classifier(value, bins, units=units)
    assert result == expected
    assert numpy.isnan(misc.classifier(numpy.nan, bins, units=units))


def test_unique_categories():
    bins = [5, 10, 15]
    classifier = partial(misc.classifier, bins=bins, units='mm')

    known_categories = ['<5 mm', '5 - 10 mm', '10 - 15 mm', '>15 mm']
    result_categories = misc.unique_categories(classifier, bins)

    assert result_categories == known_categories


def test_pop_many():
    some_dict = dict(zip(list('ABCDE'), range(5)))
    expected = {'C': 2, 'D': 3}
    assert misc.pop_many(some_dict, 'A', 'B', 'E') == expected


def test_selector():
    x = numpy.arange(10)
    expected = numpy.array(list('AAABBBCCZZ'))
    result = misc.selector('Z', (x <= 2, 'A'), (x < 6, 'B'), (x <= 7, 'C'))
    assert all(result == expected)


@pytest.mark.parametrize('input_values', ['A', 4, ('this', '5')])
def test_non_filter(input_values):
    assert misc.non_filter(input_values)


@pytest.mark.parametrize('input_values', ['A', 4, ('this', '5')])
def test_no_op(input_values):
    assert misc.no_op(input_values) == input_values


@pytest.mark.parametrize('value', [4, lambda x: 4])
def test_assign_multilevel_column(value):
    df = pandas.DataFrame(
        data=1,
        index=pandas.MultiIndex.from_product([list('ABCD'), [1, 2, 3, 4]]),
        columns=pandas.MultiIndex.from_product([list('abc'), [1, 2, 3]]),
    )

    result = misc.assign_multilevel_column(df, value, 'd', 1)
    expected = pandas.Series(4, index=df.index, name=('d', 1))
    pdtest.assert_series_equal(result[('d', 1)], expected)
