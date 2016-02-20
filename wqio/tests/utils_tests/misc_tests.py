from functools import partial
from textwrap import dedent
import random
import glob

from six import StringIO
import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import pandas.util.testing as pdtest

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from scipy import stats
import pandas
import statsmodels.api as sm

from wqio.utils import misc
from wqio.utils import numutils
from wqio import testing


testcsv = """\
Date,A,B,C,D
X,1,2,3,4
Y,5,6,7,8
Z,9,0,1,2
"""


@nt.nottest
class Dataset(object):
    def __init__(self, inflow, outflow):
        self.inflow = Location(inflow)
        self.outflow = Location(outflow)


@nt.nottest
class Location(object):
    def __init__(self, data):
        self.data = data
        self.stats = Summary(data)


@nt.nottest
class Summary(object):
    def __init__(self, data):
        self.N = len(data)
        self.max = max(data)
        self.min = min(data)
        self.nonething = None


class test_addSecondColumnLevel(object):
    def setup(self):
        self.testcsv = StringIO(testcsv)
        self.data = pandas.read_csv(self.testcsv, index_col=['Date'])
        self.known = pandas.MultiIndex.from_tuples([(u'test', u'A'), (u'test', u'B'),
                                                    (u'test', u'C'), (u'test', u'D')])

    def test_normal(self):
        newdata = misc.addSecondColumnLevel('test', 'testlevel', self.data)
        nt.assert_list_equal(self.known.tolist(), newdata.columns.tolist())

    @nptest.raises(ValueError)
    def test_error(self):
        newdata1 = misc.addSecondColumnLevel('test1', 'testlevel1', self.data)
        newdata2 = misc.addSecondColumnLevel('test2', 'testlevel2', newdata1)


class test_addColumnLevel(test_addSecondColumnLevel):
    def test_normal(self):
        newdata = misc.addColumnLevel(self.data, 'test', 'testlevel')
        nt.assert_list_equal(self.known.tolist(), newdata.columns.tolist())

    @nptest.raises(ValueError)
    def test_error(self):
        newdata1 = misc.addColumnLevel(self.data, 'test1', 'testlevel1')
        newdata2 = misc.addColumnLevel(newdata1, 'test2', 'testlevel2')


class test_uniqueIndex(object):
    def setup(self):
        dates = range(5)
        params = list('ABCDE')
        locations = ['Inflow', 'Outflow']
        dual = []
        for d in dates:
            for p in params:
                for loc in locations:
                    dual.append([d,p,loc])
        dual_array = np.array(dual)
        index = pandas.MultiIndex.from_arrays([dual_array[:,0],
                                               dual_array[:,1],
                                               dual_array[:,2]])
        index.names = ['date', 'param', 'loc']

        self.data = pandas.DataFrame(np.random.normal(size=len(index)), index=index)

    def test_getUniqueDataframeIndexVal(self):
        test = misc.getUniqueDataframeIndexVal(self.data.select(lambda x: x[0]=='0'), 'date')
        known = '0'
        nt.assert_equal(test, known)

    @nptest.raises(ValueError)
    def test_getUniqueDataframeIndexVal_error(self):
        misc.getUniqueDataframeIndexVal(self.data, 'date')


class test_redefineIndexLevel(object):
    def setup(self):
        index = pandas.MultiIndex.from_product(
            [['A', 'B', 'C'], ['mg/L']],
            names=['loc', 'units']
        )
        self.df = pandas.DataFrame(
            [[1,2], [3, 4], [5,6]],
            index=index,
            columns=['a', 'b']
        )

    def test_criteria_dropoldTrue(self):
        crit = lambda row: row[0] in ['A', 'B']
        newdf = misc.redefineIndexLevel(
            self.df, 'units', 'ug/L',
            criteria=crit, dropold=True
        )

        knowndf = pandas.DataFrame(
            [[1,2], [3, 4], [5,6]],
            index=pandas.MultiIndex.from_tuples([
                ('A', 'ug/L'), ('B', 'ug/L'), ('C', 'mg/L')
            ]),
            columns=['a', 'b']
        )
        nt.assert_true(newdf.equals(knowndf))

    def test_nocriteria_dropoldTrue(self):
        crit = None
        newdf = misc.redefineIndexLevel(
            self.df, 'units', 'ug/L',
            criteria=crit, dropold=True
        )

        knowndf = pandas.DataFrame(
            [[1,2], [3, 4], [5,6]],
            index=pandas.MultiIndex.from_tuples([
                ('A', 'ug/L'), ('B', 'ug/L'), ('C', 'ug/L')
            ]),
            columns=['a', 'b']
        )
        nt.assert_true(newdf.equals(knowndf))

    def test_criteria_dropoldFalse(self):
        crit = lambda row: row[0] in ['A', 'B']
        newdf = misc.redefineIndexLevel(
            self.df, 'units', 'ug/L',
            criteria=crit, dropold=False
        )

        knowndf = pandas.DataFrame(
            [[1,2], [1,2], [3, 4], [3, 4], [5,6]],
            index=pandas.MultiIndex.from_tuples([
                ('A', 'mg/L'), ('A', 'ug/L'),
                ('B', 'mg/L'), ('B', 'ug/L'),
                ('C', 'mg/L')
            ]),
            columns=['a', 'b']
        )
        nt.assert_true(newdf.equals(knowndf))

    def test_nocriteria_dropoldFalse(self):
        crit = None
        newdf = misc.redefineIndexLevel(
            self.df, 'units', 'ug/L',
            criteria=crit, dropold=False
        )

        knowndf = pandas.DataFrame(
            [[1,2], [1,2], [3, 4], [3, 4], [5,6], [5,6]],
            index=pandas.MultiIndex.from_tuples([
                ('A', 'mg/L'), ('A', 'ug/L'),
                ('B', 'mg/L'), ('B', 'ug/L'),
                ('C', 'mg/L'), ('C', 'ug/L')
            ]),
            columns=['a', 'b']
        )
        nt.assert_true(newdf.equals(knowndf))


class tests_with_objects(object):
    def setup(self):
        self.known_N = 50
        self.known_float = 123.4567
        self.inflow = [random.random() for n in range(self.known_N)]
        self.outflow = [random.random() for n in range(self.known_N)]
        self.dataset = Dataset(self.inflow, self.outflow)

    def test_nested_getattr(self):
        nt.assert_almost_equal(misc.nested_getattr(self.dataset, 'inflow.stats.max'),
                            self.dataset.inflow.stats.max)

    def test_stringify_int(self):
        test_val = misc.stringify(self.dataset, '%d', attribute='inflow.stats.N')
        known_val = '%d' % self.known_N
        nt.assert_equal(test_val, known_val)

    def test_stringify_float(self):
        test_val = misc.stringify(self.known_float, '%0.2f', attribute=None)
        known_val = '123.46'
        nt.assert_equal(test_val, known_val)

    def test_stringify_None(self):
        test_val = misc.stringify(self.dataset, '%d', attribute='inflow.stats.nonething')
        known_val = '--'
        nt.assert_equal(test_val, known_val)


class test_categorize_columns(object):
    def setup(self):
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
        self.df = pandas.read_csv(csvdata)

    def test_basic(self):
        df2 = misc.categorize_columns(self.df, 'parameter', 'units', 'season')
        nt.assert_true(object in self.df.dtypes.values)
        nt.assert_false(object in df2.dtypes.values)

    @nt.raises(ValueError)
    def test_float_col(self):
        misc.categorize_columns(self.df, 'parameter', 'upper')


class Check__classifier_Mixin(object):
    bins = np.arange(5, 36, 5)

    def test_small_value(self):
        result = misc._classifier(3, self.bins, units=self.units)
        nt.assert_equal(result, self.known_small_value)

    def test_between_value(self):
        result = misc._classifier(17, self.bins, units=self.units)
        nt.assert_equal(result, self.known_between_value)

    def test_edge_value(self):
        result = misc._classifier(25, self.bins, units=self.units)
        nt.assert_equal(result, self.known_edge_value)

    def test_big_value(self):
        result = misc._classifier(40, self.bins, units=self.units)
        nt.assert_equal(result, self.known_big_value)

    def test_nan(self):
        nt.assert_true(np.isnan(misc._classifier(np.nan, self.bins, units=self.units)))


class Test__classifier_without_units(Check__classifier_Mixin):
    def setup(self):
        self.units = None
        self.known_small_value = '<5'
        self.known_between_value = '15 - 20'
        self.known_edge_value = '20 - 25'
        self.known_big_value = '>35'


class Test__classifier_with_units(Check__classifier_Mixin):
    def setup(self):
        self.units = 'mm'
        self.known_small_value = '<5 mm'
        self.known_between_value = '15 - 20 mm'
        self.known_edge_value = '20 - 25 mm'
        self.known_big_value = '>35 mm'


def test__unique_categories():
    bins = [5, 10, 15]
    classifier = partial(misc._classifier, bins=bins, units='mm')

    known_categories = ['<5 mm', '5 - 10 mm', '10 - 15 mm', '>15 mm']
    result_categories = misc._unique_categories(classifier, bins)

    nt.assert_list_equal(result_categories, known_categories)

