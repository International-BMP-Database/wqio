import warnings
import random
import sys
import os
from six import StringIO

from pkg_resources import resource_filename
from textwrap import dedent
import glob

import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import pandas.util.testing as pdtest

from wqio import testing
usetex = False #testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex
import matplotlib.pyplot as plt

from scipy import stats
import pandas
import statsmodels.api as sm

from wqio.utils import misc
from wqio.utils import numutils
from wqio import testing


testcsv = StringIO("""\
Date,A,B,C,D
X,1,2,3,4
Y,5,6,7,8
Z,9,0,1,2
""")


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


def test__sig_figs_helper():
    x = 1.2
    nt.assert_equal(misc._sig_figs(x), numutils.sigFigs(x, 3, tex=True))


class test_addSecondColumnLevel(object):
    def setup(self):
        self.testcsv = StringIO("""\
Date,A,B,C,D
X,1,2,3,4
Y,5,6,7,8
Z,9,0,1,2
        """)
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


def test__boxplot_legend():
    fig, ax = plt.subplots()
    misc._boxplot_legend(ax, notch=True)


def test_processFilename():
    startname = 'This-is a, very+ &dumb$_{test/name}'
    endname = 'This-isaverydumbtestname'
    nt.assert_equal(endname, misc.processFilename(startname))


def test_makeTexTable_normal():
    known = '\n    \\begin{table}[h!]\n        \\rowcolors{1}{CVCWhite}{CVCLightGrey}\n    ' \
            '    \\caption{test caption}\n        \\centering\n        \\input{fake.tex}\n    ' \
            '\\end{table}\n    \n    \n    '
    test = misc.makeTexTable('fake.tex', 'test caption')
    nt.assert_equal(known, test)


def test_makeTexTable_allOptions():
    known = '\n    \\begin{sidewaystable}[bt]\n        \\rowcolors{1}{CVCWhite}{CVCLightGrey}' \
            '\n        \\caption{test caption}\n        \\centering\n        \\input{fake.tex}\n    ' \
            '\\end{sidewaystable}\n    test footnote\n    \\clearpage\n    '
    test = misc.makeTexTable('fake.tex', 'test caption', sideways=True,
           footnotetext='test footnote', clearpage=True, pos='bt')
    nt.assert_equal(known, test)


class test_makeLongLandscapeTexTable(object):
    def setup(self):
        self.maxDiff = None
        dfdict = {
            'W': {
                'a': 0.84386963791251501,
                'b': -0.22109837444207142,
            },
            'X': {
                'a': 0.70049867715201963,
                'b': 1.4764939161054218,
            },
            'Y': {
                'a': -1.3477794473987552,
                'b': -1.1939220296611821,
            },
        }
        self.df = pandas.DataFrame.from_dict(dfdict)
        self.known_nofootnote =   '\n    \\begin{landscape}\n        \\centering\n        \\rowcolors{1}{CVCWhite}{CVCLightGrey}\n        \\begin{longtable}{lcc}\n            \\caption{test caption} \\label{label} \\\\\n            \\toprule\n                \\multicolumn{1}{l}{W} &\n\t\t\\multicolumn{1}{p{16mm}}{X} &\n\t\t\\multicolumn{1}{p{16mm}}{Y} \\\\\n            \\toprule\n            \\endfirsthead\n\n            \\multicolumn{3}{c}\n            {{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n            \\toprule\n                \\multicolumn{1}{l}{W} &\n\t\t\\multicolumn{1}{p{16mm}}{X} &\n\t\t\\multicolumn{1}{p{16mm}}{Y} \\\\\n            \\toprule\n            \\endhead\n\n            \\toprule\n                \\rowcolor{CVCWhite}\n                \\multicolumn{3}{r}{{Continued on next page...}} \\\\\n            \\bottomrule\n            \\endfoot\n\n            \\bottomrule\n            \\endlastfoot\n\n 0.844 & 0.700 & -1.35 \\\\\n-0.221 &  1.48 & -1.19 \\\\\n\n        \\end{longtable}\n    \\end{landscape}\n    \n    \\clearpage\n    '
        self.known_withfootnote = '\n    \\begin{landscape}\n        \\centering\n        \\rowcolors{1}{CVCWhite}{CVCLightGrey}\n        \\begin{longtable}{lcc}\n            \\caption{test caption} \\label{label} \\\\\n            \\toprule\n                \\multicolumn{1}{l}{W} &\n\t\t\\multicolumn{1}{p{16mm}}{X} &\n\t\t\\multicolumn{1}{p{16mm}}{Y} \\\\\n            \\toprule\n            \\endfirsthead\n\n            \\multicolumn{3}{c}\n            {{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n            \\toprule\n                \\multicolumn{1}{l}{W} &\n\t\t\\multicolumn{1}{p{16mm}}{X} &\n\t\t\\multicolumn{1}{p{16mm}}{Y} \\\\\n            \\toprule\n            \\endhead\n\n            \\toprule\n                \\rowcolor{CVCWhite}\n                \\multicolumn{3}{r}{{Continued on next page...}} \\\\\n            \\bottomrule\n            \\endfoot\n\n            \\bottomrule\n            \\endlastfoot\n\n 0.844 & 0.700 & -1.35 \\\\\n-0.221 &  1.48 & -1.19 \\\\\n\n        \\end{longtable}\n    \\end{landscape}\n    test note\n    \\clearpage\n    '

    def test_makeLongLandscapeTexTable_noFootnote(self):
        test = misc.makeLongLandscapeTexTable(self.df, 'test caption', 'label')
        nt.assert_equal(self.known_nofootnote, test)

    def test_makeLongLandscapeTexTable_Footnote(self):
        test = misc.makeLongLandscapeTexTable(self.df, 'test caption', 'label', 'test note')
        nt.assert_equal(self.known_withfootnote, test)


def test_makeTexFigure():
    known =  '\n    \\begin{figure}[hb]   % FIGURE\n        \\centering\n        ' \
             '\\includegraphics[scale=1.00]{fake.pdf}\n        \\caption{test caption}\n    ' \
             '\\end{figure}         % FIGURE\n    \\clearpage\n    '
    test = misc.makeTexFigure('fake.pdf', 'test caption')
    nt.assert_equal(known, test)


class tests_with_paths(object):
    @nt.nottest
    def makePath(self, filename):
        path = resource_filename("wqio.data", filename)
        return path
        #return os.path.join(sys.prefix, 'wqio_data', 'testing', filename)

    def setup(self):
        if os.path.split(os.getcwd())[-1] == 'src':
            self.prefix = os.path.join('.', 'utils', 'tests')
        else:
            self.prefix = os.path.join('..', 'utils', 'tests')

        self.tablestring = "Date,A,B,C,D\nX,1,2,3,4\nY,5,6,7,8\nZ,9,0,1,2"
        self.testcsvpath = self.makePath('testtable.csv')
        with open(self.testcsvpath, 'w') as testfile:
            testfile.write(self.tablestring)

    @nptest.dec.skipif(not usetex)
    def test_makeBoxplotLegend(self):
        misc.makeBoxplotLegend(self.makePath('bplegendtest'))

    def test_constructPath(self):
        testpath = misc.constructPath('test1 2', 'pdf', 'cvc', 'output')
        expectedpath = os.path.join('cvc', 'output', 'img', 'test12.pdf')
        nt.assert_equal(testpath, expectedpath)

    def test_makeTablesFromCSVStrings(self):
        misc.makeTablesFromCSVStrings(self.tablestring,
                    texpath=self.makePath('testtable.tex'),
                    csvpath=self.testcsvpath)

    def test_addStatsToOutputSummary(self):
        inputcols = pandas.read_csv(self.testcsvpath).columns.tolist()
        summary = misc.addStatsToOutputSummary(self.testcsvpath)
        known_index = ['X', 'Y', 'Z', 'count', 'mean', 'std',
                       'min', '25%', '50%', '75%', 'max']
        nt.assert_list_equal(summary.index.tolist(), known_index)
        nt.assert_list_equal(inputcols[1:], summary.columns.tolist())

    def test_csvToTex(self):
        testfile = 'testtable_toTex.tex'
        misc.csvToTex(self.testcsvpath, self.makePath(testfile))

        if sys.platform == 'win32':
            knownfile = 'testtable_toTex_KnownWin.tex'
        else:
            knownfile = 'testtable_toTex_KnownUnix.tex'

        with open(self.makePath(testfile), 'r') as test, \
             open(self.makePath(knownfile), 'r') as known:
             nt.assert_equal(test.read(), known.read())

    def test_csvToXlsx(self):
        misc.csvToXlsx(self.testcsvpath,
                        self.makePath('testtable_toXL.xlsx'))

    def test_addExternalValueToOutputSummary(self):
        compardict = {'test1': 10, 'test2': 11}
        comparecol = 'A'
        misc.addExternalValueToOutputSummary(self.testcsvpath, compardict,
                                              comparecol, index_cols=['Date'])

        known = 'Date,A,B,C,D\nX,1,2.0,3.0,4.0\nY,5,6.0,7.0,8.0\n' \
                'Z,9,0.0,1.0,2.0\ntest1,10,--,--,--\ntest2,11,--,--,--\n'

        with open(self.testcsvpath, 'r') as outfile:
            test = outfile.read()

        nt.assert_equal(test, known)


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


class test_uniqueIndex(object):
    def setup():
        dates = range(5)
        params = list('ABCDE')
        locations = ['Inflow', 'Outflow']
        for d in dates:
            for p in params:
                for loc in locations:
                    dual.append([d,p,loc])

        index = pandas.MultiIndex.from_arrays([dual_array[:,0],
                                                    dual_array[:,1],
                                                    dual_array[:,2]])
        index.names = ['date', 'param', 'loc']

        self.data = pandas.DataFrame(np.range.normal(size=len(index)), index=index)

        def test_getUniqueDataframeIndexVal():
            test = misc.getUniqueDataframeIndexVal(self.data.select(lambda x: x[0]=='0'), 'date')
            known = '0'
            nt.assert_equal(test, known)

        @nptest.raises(ValueError)
        def test_getUniqueDataframeIndexVal_error():
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


def test_sanitizeTex():
    inputstring = r""" \
    \$x\_\{4\}\textasciicircum\{2\} \textbackslashtimes \% ug/L$ \textbackslash \\
    """
    desiredstring = r""" \
    $x_{4}^{2} \times \% \si[per-mode=symbol]{\micro\gram\per\liter}$  \tabularnewline
    """

    nt.assert_equal(misc.sanitizeTex(inputstring), desiredstring)


class base_whiskers_and_fliersMixin(object):
    def teardown(self):
        pass

    def setup(self):
        self.base_setup()
        self.decimal = 3
        self.data = [
            2.20e-01, 2.70e-01, 3.08e-01, 3.20e-01, 4.10e-01, 4.44e-01,
            4.82e-01, 5.46e-01, 6.05e-01, 6.61e-01, 7.16e-01, 7.70e-01,
            8.24e-01, 1.00e-03, 4.90e-02, 5.60e-02, 1.40e-01, 1.69e-01,
            1.83e-01, 2.06e-01, 2.10e-01, 2.13e-01, 2.86e-01, 3.16e-01,
            3.40e-01, 3.57e-01, 3.71e-01, 3.72e-01, 3.78e-01, 3.81e-01,
            3.86e-01, 3.89e-01, 3.90e-01, 3.93e-01, 4.00e-01, 4.03e-01,
            4.10e-01, 4.10e-01, 4.29e-01, 4.40e-01, 4.40e-01, 4.40e-01,
            4.46e-01, 4.46e-01, 4.50e-01, 4.51e-01, 4.52e-01, 4.56e-01,
            4.60e-01, 4.66e-01, 4.72e-01, 4.78e-01, 4.81e-01, 4.83e-01,
            4.86e-01, 4.89e-01, 4.98e-01, 5.00e-01, 5.00e-01, 5.03e-01,
            5.18e-01, 5.32e-01, 5.37e-01, 5.38e-01, 5.68e-01, 5.69e-01,
            5.78e-01, 5.88e-01, 5.94e-01, 5.96e-01, 6.02e-01, 6.10e-01,
            6.10e-01, 6.10e-01, 6.19e-01, 6.20e-01, 6.20e-01, 6.28e-01,
            6.38e-01, 6.39e-01, 6.42e-01, 6.61e-01, 6.71e-01, 6.75e-01,
            6.80e-01, 6.96e-01, 7.00e-01, 7.01e-01, 7.09e-01, 7.16e-01,
            7.17e-01, 7.30e-01, 7.62e-01, 7.64e-01, 7.69e-01, 7.70e-01,
            7.77e-01, 7.80e-01, 8.06e-01, 8.10e-01, 8.23e-01, 8.30e-01,
            8.50e-01, 8.50e-01, 8.56e-01, 8.56e-01, 8.80e-01, 8.80e-01,
            8.93e-01, 8.96e-01, 8.97e-01, 8.99e-01, 9.22e-01, 9.28e-01,
            9.30e-01, 9.64e-01, 9.65e-01, 9.76e-01, 9.79e-01, 9.90e-01,
            9.99e-01, 1.00e+00, 1.00e+00, 1.01e+00, 1.02e+00, 1.03e+00,
            1.03e+00, 1.03e+00, 1.04e+00, 1.05e+00, 1.05e+00, 1.05e+00,
            1.06e+00, 1.07e+00, 1.08e+00, 1.08e+00, 1.10e+00, 1.10e+00,
            1.11e+00, 1.12e+00, 1.12e+00, 1.13e+00, 1.14e+00, 1.14e+00,
            1.14e+00, 1.15e+00, 1.16e+00, 1.17e+00, 1.17e+00, 1.17e+00,
            1.19e+00, 1.19e+00, 1.20e+00, 1.20e+00, 1.21e+00, 1.22e+00,
            1.22e+00, 1.23e+00, 1.23e+00, 1.23e+00, 1.25e+00, 1.25e+00,
            1.26e+00, 1.26e+00, 1.27e+00, 1.27e+00, 1.28e+00, 1.29e+00,
            1.29e+00, 1.30e+00, 1.30e+00, 1.30e+00, 1.31e+00, 1.31e+00,
            1.31e+00, 1.32e+00, 1.33e+00, 1.34e+00, 1.35e+00, 1.35e+00,
            1.35e+00, 1.36e+00, 1.36e+00, 1.36e+00, 1.36e+00, 1.37e+00,
            1.38e+00, 1.39e+00, 1.39e+00, 1.40e+00, 1.41e+00, 1.43e+00,
            1.44e+00, 1.44e+00, 1.47e+00, 1.47e+00, 1.48e+00, 1.51e+00,
            1.51e+00, 1.53e+00, 1.55e+00, 1.55e+00, 1.55e+00, 1.57e+00,
            1.57e+00, 1.57e+00, 1.59e+00, 1.59e+00, 1.60e+00, 1.60e+00,
            1.61e+00, 1.62e+00, 1.62e+00, 1.62e+00, 1.62e+00, 1.63e+00,
            1.63e+00, 1.63e+00, 1.64e+00, 1.66e+00, 1.68e+00, 1.68e+00,
            1.68e+00, 1.68e+00, 1.70e+00, 1.70e+00, 1.71e+00, 1.71e+00,
            1.71e+00, 1.74e+00, 1.75e+00, 1.75e+00, 1.75e+00, 1.76e+00,
            1.76e+00, 1.77e+00, 1.77e+00, 1.77e+00, 1.78e+00, 1.78e+00,
            1.79e+00, 1.79e+00, 1.80e+00, 1.81e+00, 1.81e+00, 1.82e+00,
            1.82e+00, 1.82e+00, 1.83e+00, 1.85e+00, 1.85e+00, 1.85e+00,
            1.85e+00, 1.86e+00, 1.86e+00, 1.86e+00, 1.86e+00, 1.87e+00,
            1.87e+00, 1.89e+00, 1.90e+00, 1.91e+00, 1.92e+00, 1.92e+00,
            1.92e+00, 1.94e+00, 1.95e+00, 1.95e+00, 1.95e+00, 1.96e+00,
            1.96e+00, 1.97e+00, 1.97e+00, 1.97e+00, 1.97e+00, 1.98e+00,
            1.99e+00, 1.99e+00, 1.99e+00, 2.00e+00, 2.00e+00, 2.00e+00,
            2.01e+00, 2.01e+00, 2.01e+00, 2.02e+00, 2.04e+00, 2.05e+00,
            2.06e+00, 2.06e+00, 2.06e+00, 2.07e+00, 2.08e+00, 2.09e+00,
            2.09e+00, 2.10e+00, 2.10e+00, 2.11e+00, 2.11e+00, 2.12e+00,
            2.12e+00, 2.12e+00, 2.13e+00, 2.13e+00, 2.13e+00, 2.14e+00,
            2.14e+00, 2.14e+00, 2.14e+00, 2.14e+00, 2.15e+00, 2.16e+00,
            2.17e+00, 2.18e+00, 2.18e+00, 2.18e+00, 2.19e+00, 2.19e+00,
            2.19e+00, 2.19e+00, 2.19e+00, 2.21e+00, 2.23e+00, 2.23e+00,
            2.23e+00, 2.25e+00, 2.25e+00, 2.25e+00, 2.25e+00, 2.26e+00,
            2.26e+00, 2.26e+00, 2.26e+00, 2.26e+00, 2.27e+00, 2.27e+00,
            2.28e+00, 2.28e+00, 2.28e+00, 2.29e+00, 2.29e+00, 2.29e+00,
            2.30e+00, 2.31e+00, 2.32e+00, 2.33e+00, 2.33e+00, 2.33e+00,
            2.33e+00, 2.34e+00, 2.36e+00, 2.38e+00, 2.38e+00, 2.39e+00,
            2.39e+00, 2.39e+00, 2.41e+00, 2.42e+00, 2.43e+00, 2.45e+00,
            2.45e+00, 2.47e+00, 2.48e+00, 2.49e+00, 2.49e+00, 2.49e+00,
            2.50e+00, 2.51e+00, 2.51e+00, 2.52e+00, 2.53e+00, 2.53e+00,
            2.54e+00, 2.54e+00, 2.56e+00, 2.58e+00, 2.59e+00, 2.59e+00,
            2.60e+00, 2.61e+00, 2.61e+00, 2.61e+00, 2.62e+00, 2.62e+00,
            2.63e+00, 2.65e+00, 2.65e+00, 2.66e+00, 2.66e+00, 2.68e+00,
            2.69e+00, 2.69e+00, 2.70e+00, 2.72e+00, 2.72e+00, 2.73e+00,
            2.75e+00, 2.77e+00, 2.78e+00, 2.79e+00, 2.81e+00, 2.81e+00,
            2.82e+00, 2.84e+00, 2.84e+00, 2.85e+00, 2.85e+00, 2.86e+00,
            2.86e+00, 2.88e+00, 2.92e+00, 2.93e+00, 2.93e+00, 2.95e+00,
            2.96e+00, 2.96e+00, 2.99e+00, 3.00e+00, 3.01e+00, 3.02e+00,
            3.03e+00, 3.03e+00, 3.14e+00, 3.15e+00, 3.16e+00, 3.17e+00,
            3.17e+00, 3.18e+00, 3.18e+00, 3.19e+00, 3.20e+00, 3.22e+00,
            3.24e+00, 3.25e+00, 3.29e+00, 3.31e+00, 3.32e+00, 3.32e+00,
            3.34e+00, 3.35e+00, 3.36e+00, 3.38e+00, 3.44e+00, 3.45e+00,
            3.46e+00, 3.48e+00, 3.49e+00, 3.53e+00, 3.59e+00, 3.63e+00,
            3.70e+00, 3.70e+00, 3.76e+00, 3.80e+00, 3.80e+00, 3.80e+00,
            3.83e+00, 3.84e+00, 3.88e+00, 3.90e+00, 3.91e+00, 3.96e+00,
            3.97e+00, 3.97e+00, 4.02e+00, 4.03e+00, 4.06e+00, 4.12e+00,
            4.19e+00, 4.21e+00, 4.53e+00, 4.56e+00, 4.61e+00, 4.62e+00,
            4.73e+00, 5.13e+00, 5.21e+00, 5.40e+00, 5.98e+00, 6.12e+00,
            6.94e+00, 7.38e+00, 7.56e+00, 8.06e+00, 1.38e+01, 1.51e+01,
            1.82e+01
        ]
        self.q1 = np.percentile(self.data, 25)
        self.q3 = np.percentile(self.data, 75)


        self.bs = misc.whiskers_and_fliers(
            self.transformin(self.data),
            self.transformin(self.q1),
            self.transformin(self.q3),
            transformout=self.transformout
        )

    def test_hi_whiskers(self):
        nptest.assert_almost_equal(
            self.known_bs['whishi'],
            self.bs['whishi'],
            decimal=self.decimal
        )

    def test_hi_whiskers(self):
        nptest.assert_almost_equal(
            self.known_bs['whislo'],
            self.bs['whislo'],
            decimal=self.decimal
        )

    def test_fliers(self):
        nptest.assert_array_almost_equal(
            self.known_bs['fliers'],
            self.bs['fliers'],
            decimal=self.decimal
        )


class test_whiskers_and_fliers_ari(base_whiskers_and_fliersMixin):
    @nt.nottest
    def base_setup(self):
        self.known_bs = {
            'whishi': 4.7300000190734863,
            'fliers': np.array([
                4.730,   5.130,   5.210,   5.400,
                5.980,   6.120,   6.940,   7.380,
                7.560,   8.060,  13.800,  15.100,
               18.200
            ]),
            'whislo': 0.0011100000143051147
        }
        self.transformin = lambda x: x
        self.transformout = lambda x: x


class test_whiskers_and_fliers_natlog(base_whiskers_and_fliersMixin):
    @nt.nottest
    def base_setup(self):
        self.known_bs = {
            'whishi': 8.0606803894042987,
            'fliers': np.array([
                2.200e-01, 1.000e-03, 4.900e-02, 5.600e-02, 1.400e-01,
                1.690e-01, 1.830e-01, 2.060e-01, 2.100e-01, 2.130e-01,
                1.380e+01, 1.510e+01, 1.820e+01
            ]),
            'whislo': 0.27031713639148325
        }
        self.transformin = lambda x: np.log(x)
        self.transformout = lambda x: np.exp(x)


class test_whiskers_and_fliers_log10(base_whiskers_and_fliersMixin):
    @nt.nottest
    def base_setup(self):
        self.known_bs = {
            'whishi': 0.3741651859057793,
            'fliers': np.array([
                2.200e-01, 1.000e-03, 4.900e-02, 5.600e-02, 1.400e-01,
                1.690e-01, 1.830e-01, 2.060e-01, 2.100e-01, 2.130e-01,
                1.380e+01, 1.510e+01, 1.820e+01
            ]),
            'whislo':  0.27000000000000002
        }
        self.transformin = lambda x: np.log10(x)
        self.transformout = lambda x: 10**x


class test_LaTeXDirectory(object):
    def setup(self):
        self.origdir = os.getcwd()
        self.deepdir = os.path.join(os.getcwd(), 'test1', 'test2', 'test3')
        self.deepfile = os.path.join(self.deepdir, 'f.tex')
        tex_content = dedent(r"""
            \documentclass[12pt]{article}
            \usepackage{lingmacros}
            \usepackage{tree-dvips}
            \begin{document}

            \section*{Notes for My Paper}

            Don't forget to include examples of topicalization.
            They look like this:

            {\small
            \enumsentence{Topicalization from sentential subject:\\
            \shortex{7}{a John$_i$ [a & kltukl & [el &
              {\bf l-}oltoir & er & ngii$_i$ & a Mary]]}
            { & {\bf R-}clear & {\sc comp} &
              {\bf IR}.{\sc 3s}-love   & P & him & }
            {John, (it's) clear that Mary loves (him).}}
            }

            \subsection*{How to handle topicalization}

            I'll just assume a tree structure like (\ex{1}).

            {\small
            \enumsentence{Structure of A$'$ Projections:\\ [2ex]
            \begin{tabular}[t]{cccc}
                & \node{i}{CP}\\ [2ex]
                \node{ii}{Spec} &   &\node{iii}{C$'$}\\ [2ex]
                    &\node{iv}{C} & & \node{v}{SAgrP}
            \end{tabular}
            \nodeconnect{i}{ii}
            \nodeconnect{i}{iii}
            \nodeconnect{iii}{iv}
            \nodeconnect{iii}{v}
            }
            }

            \subsection*{Mood}

            Mood changes when there is a topic, as well as when
            there is WH-movement.  \emph{Irrealis} is the mood when
            there is a non-subject topic or WH-phrase in Comp.
            \emph{Realis} is the mood when there is a subject topic
            or WH-phrase.

            \end{document}
        """)

        if os.path.exists(self.deepdir):
            self.teardown()

        os.makedirs(self.deepdir)
        with open(self.deepfile, 'w') as dfile:
            dfile.write(tex_content)

    def teardown(self):
        allfiles = glob.glob(os.path.join(self.deepdir, "f.*"))
        for af in allfiles:
            os.remove(af)
        os.removedirs(self.deepdir)

    def test_dir(self):
        nt.assert_equal(os.getcwd(), self.origdir)
        with misc.LaTeXDirectory(self.deepdir):
            nt.assert_equal(os.getcwd(), self.deepdir)

        nt.assert_equal(os.getcwd(), self.origdir)

    def test_file(self):
        nt.assert_equal(os.getcwd(), self.origdir)
        with misc.LaTeXDirectory(self.deepfile):
            nt.assert_equal(os.getcwd(), self.deepdir)

        nt.assert_equal(os.getcwd(), self.origdir)

    @nptest.dec.skipif(testing.checkdep_tex() is None)
    def test_compile_smoke(self):
        with misc.LaTeXDirectory(self.deepfile) as latex:
            latex.compile(self.deepfile)
