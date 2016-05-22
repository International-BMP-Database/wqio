from functools import partial
from textwrap import dedent
from io import StringIO

import pytest
import numpy.testing as nptest
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


def test_addSecondColumnLevel(basic_data):
    known_cols = pandas.MultiIndex.from_tuples([(u'test', u'A'), (u'test', u'B'),
                                                (u'test', u'C'), (u'test', u'D')])
    newdata = misc.addSecondColumnLevel('test', 'testlevel', basic_data)
    assert known_cols.tolist() == newdata.columns.tolist()

    # can only add levels to non-MultiIndex columns
    with pytest.raises(ValueError):
        misc.addSecondColumnLevel('test2', 'testlevel2', newdata)


def test_addColumnLevel(basic_data):
    known_cols = pandas.MultiIndex.from_tuples([(u'test', u'A'), (u'test', u'B'),
                                                (u'test', u'C'), (u'test', u'D')])
    newdata = misc.addColumnLevel(basic_data, 'test', 'testlevel')
    assert known_cols.tolist() == newdata.columns.tolist()

    # can only add levels to non-MultiIndex columns
    with pytest.raises(ValueError):
        misc.addColumnLevel(newdata, 'test2', 'testlevel2')


@helpers.seed
def test_getUniqueDataframeIndexVal():
    dates = range(5)
    params = list('ABCDE')
    locations = ['Inflow', 'Outflow']
    index = pandas.MultiIndex.from_product(
        [dates, params, locations],
        names=['date', 'param', 'loc']
    )
    data = pandas.DataFrame(numpy.random.normal(size=len(index)), index=index)
    test = misc.getUniqueDataframeIndexVal(data.select(lambda x: x[0] == 0), 'date')
    assert test == 0

    with pytest.raises(ValueError):
        misc.getUniqueDataframeIndexVal(data, 'date')



@pytest.fixture
def index_df():
    index = pandas.MultiIndex.from_product([['A', 'B', 'C'], ['mg/L']], names=['loc', 'units'])
    return pandas.DataFrame([[1,2], [3, 4], [5,6]], index=index, columns=['a', 'b'])


@pytest.mark.parametrize('criteria', [None, lambda row: row[0] in ['A', 'B']])
@pytest.mark.parametrize('dropold', [True, False])
def test_redefineIndexLevel(index_df, criteria, dropold):
    expected_cols = ['a', 'b']
    if dropold:
        expected_value = [[1,2], [3, 4], [5, 6]]
        if criteria:
            expected_index = [('A', 'ug/L'), ('B', 'ug/L'), ('C', 'mg/L')]
        else:
            expected_index = [('A', 'ug/L'), ('B', 'ug/L'), ('C', 'ug/L')]

    else:
        if criteria:
            expected_value = [[1,2], [1,2], [3, 4], [3, 4], [5, 6]]
            expected_index = [
                ('A', 'mg/L'), ('A', 'ug/L'),
                ('B', 'mg/L'), ('B', 'ug/L'),
                ('C', 'mg/L')
            ]
        else:
            expected_value = [[1,2], [1,2], [3, 4], [3, 4], [5, 6], [5, 6]]
            expected_index = [
                ('A', 'mg/L'), ('A', 'ug/L'),
                ('B', 'mg/L'), ('B', 'ug/L'),
                ('C', 'mg/L'), ('C', 'ug/L')
            ]

    result = misc.redefineIndexLevel(index_df, 'units', 'ug/L', criteria=criteria, dropold=dropold)
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
    outflow = [2, 5 ,7, 15.17]
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
    assert object in df.dtypes.values
    assert object not in df2.dtypes.values

    with pytest.raises(ValueError):
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

    result = misc._classifier(value, bins, units=units)
    assert result == expected
    assert numpy.isnan(misc._classifier(numpy.nan, bins, units=units))


def test__unique_categories():
    bins = [5, 10, 15]
    classifier = partial(misc._classifier, bins=bins, units='mm')

    known_categories = ['<5 mm', '5 - 10 mm', '10 - 15 mm', '>15 mm']
    result_categories = misc._unique_categories(classifier, bins)

    assert result_categories == known_categories

