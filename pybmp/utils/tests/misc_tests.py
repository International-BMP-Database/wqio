import warnings
import random
import sys
import os
if sys.version_info.major == 3:
    from io import StringIO
else:
    from StringIO import StringIO

from nose.tools import *
import numpy as np
import numpy.testing as nptest

from pybmp import testing
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex
import matplotlib.pyplot as plt

import pandas

from .. import misc


testcsv = StringIO("""\
Date,A,B,C,D
X,1,2,3,4
Y,5,6,7,8
Z,9,0,1,2
""")


@nottest
class Dataset(object):
    def __init__(self, inflow, outflow):
        self.inflow = Location(inflow)
        self.outflow = Location(outflow)


@nottest
class Location(object):
    def __init__(self, data):
        self.data = data
        self.stats = Summary(data)


@nottest
class Summary(object):
    def __init__(self, data):
        self.N = len(data)
        self.max = max(data)
        self.min = min(data)
        self.nonething = None


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
        assert_list_equal(self.known.tolist(), newdata.columns.tolist())

    @nptest.raises(ValueError)
    def test_error(self):
        newdata1 = misc.addSecondColumnLevel('test1', 'testlevel1', self.data)
        newdata2 = misc.addSecondColumnLevel('test2', 'testlevel2', newdata1)


def test_sigFigs_baseline_gt1():
    x1 = 1234.56
    assert_equal(misc.sigFigs(x1, 3), '1,230')
    assert_equal(misc.sigFigs(x1, 4), '1,235')

def test_sigFigs_baseline_lt1():
    x2 = 0.12345
    assert_equal(misc.sigFigs(x2, 3), '0.123')
    assert_equal(misc.sigFigs(x2, 4), '0.1235')

def test_sigFigs_bad_n():
    with assert_raises(ValueError):
        misc.sigFigs(1.234, 0)

def test_sigFigs_trailing_zeros_gt1():
    x1 = 1234.56
    assert_equal(misc.sigFigs(x1, 8), '1,234.5600')

def test_sigFigs_trailing_zeros_lt1():
    x2 = 0.12345
    assert_equal(misc.sigFigs(x2, 8), '0.12345000')

def test_sigFigs_exponents_notex_gt1():
    x1 = 123456789.123456789
    assert_equal(misc.sigFigs(x1, 3, tex=False), '1.23e+08')

def test_sigFigs_exponents_notex_lt1():
    x2 = 0.000000123
    assert_equal(misc.sigFigs(x2, 3, tex=False), '1.23e-07')

def test_sigFigs_exponents_tex_gt1():
    x1 = 123456789.123456789
    assert_equal(misc.sigFigs(x1, 3, tex=True), r'$1.23 \times 10 ^ {8}$')

def test_sigFigs_exponents_tex_lt1():
    x2 = 0.000000123
    assert_equal(misc.sigFigs(x2, 3, tex=True), r'$1.23 \times 10 ^ {-7}$')

def test_sigFigs_pvals_noop():
    p = 0.001
    assert_equal(misc.sigFigs(p, 1, pval=True), '0.001')

def test_sigFigs_pvals_op():
    p = 0.0005
    assert_equal(misc.sigFigs(p, 3, pval=True), '<0.001')

def test_sigFigs_pvals_op_tex():
    p = 0.0005
    assert_equal(misc.sigFigs(p, 3, tex=True, pval=True), '$<0.001$')

@raises
def test_sigFigs_exception():
    misc.sigFigs(199, -1)

def test__sig_figs():
    x1 = 1234.56
    x2 = 0.12345
    assert_equal(misc.sigFigs(x1, 3), misc._sig_figs(x1))
    assert_equal(misc.sigFigs(x2, 3), misc._sig_figs(x2))


def test__boxplot_legend():
    fig, ax = plt.subplots()
    misc._boxplot_legend(ax, notch=True)


def test_processFilename():
    startname = 'This-is a, very+ &dumb$_{test/name}'
    endname = 'This-isaverydumbtestname'
    assert_equal(endname, misc.processFilename(startname))


def test_makeTexTable_normal():
    known = '\n    \\begin{table}[h!]\n        \\rowcolors{1}{CVCWhite}{CVCLightGrey}\n    ' \
            '    \\caption{test caption}\n        \\centering\n        \\input{fake.tex}\n    ' \
            '\\end{table}\n    \n    \n    '
    test = misc.makeTexTable('fake.tex', 'test caption')
    assert_equal(known, test)


def test_makeTexTable_allOptions():
    known = '\n    \\begin{sidewaystable}[bt]\n        \\rowcolors{1}{CVCWhite}{CVCLightGrey}' \
            '\n        \\caption{test caption}\n        \\centering\n        \\input{fake.tex}\n    ' \
            '\\end{sidewaystable}\n    test footnote\n    \\clearpage\n    '
    test = misc.makeTexTable('fake.tex', 'test caption', sideways=True,
           footnotetext='test footnote', clearpage=True, pos='bt')
    assert_equal(known, test)


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
        assert_equal(self.known_nofootnote, test)

    def test_makeLongLandscapeTexTable_Footnote(self):
        test = misc.makeLongLandscapeTexTable(self.df, 'test caption', 'label', 'test note')
        assert_equal(self.known_withfootnote, test)


def test_makeTexFigure():
    known =  '\n    \\begin{figure}[hb]   % FIGURE\n        \\centering\n        ' \
             '\\includegraphics[scale=1.00]{fake.pdf}\n        \\caption{test caption}\n    ' \
             '\\end{figure}         % FIGURE\n    \\clearpage\n    '
    test = misc.makeTexFigure('fake.pdf', 'test caption')
    assert_equal(known, test)


class tests_with_paths(object):
    @nottest
    def makePath(self, filename):
        return os.path.join(sys.prefix, 'pybmp_data', 'testing', filename)

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
        assert_equal(testpath, expectedpath)

    def test_makeTablesFromCSVStrings(self):
        misc.makeTablesFromCSVStrings(self.tablestring,
                    texpath=self.makePath('testtable.tex'),
                    csvpath=self.testcsvpath)

    def test_addStatsToOutputSummary(self):
        inputcols = pandas.read_csv(self.testcsvpath).columns.tolist()
        summary = misc.addStatsToOutputSummary(self.testcsvpath)
        known_index = ['X', 'Y', 'Z', 'count', 'mean', 'std',
                       'min', '25%', '50%', '75%', 'max']
        assert_list_equal(summary.index.tolist(), known_index)
        assert_list_equal(inputcols[1:], summary.columns.tolist())

    def test_csvToTex(self):
        testfile = 'testtable_toTex.tex'
        misc.csvToTex(self.testcsvpath, self.makePath(testfile))

        if sys.platform == 'win32':
            knownfile = 'testtable_toTex_KnownWin.tex'
        else:
            knownfile = 'testtable_toTex_KnownUnix.tex'

        with open(self.makePath(testfile), 'r') as test, \
             open(self.makePath(knownfile), 'r') as known:
             assert_equal(test.read(), known.read())

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

        assert_equal(test, known)


class test_with_objects(object):
    def setup(self):
        self.known_N = 50
        self.known_float = 123.4567
        self.inflow = [random.random() for n in range(self.known_N)]
        self.outflow = [random.random() for n in range(self.known_N)]
        self.dataset = Dataset(self.inflow, self.outflow)

    def test_nested_getattr(self):
        assert_almost_equal(misc.nested_getattr(self.dataset, 'inflow.stats.max'),
                            self.dataset.inflow.stats.max)

    def test_stringify_int(self):
        test_val = misc.stringify(self.dataset, '%d', attribute='inflow.stats.N')
        known_val = '%d' % self.known_N
        assert_equal(test_val, known_val)

    def test_stringify_float(self):
        test_val = misc.stringify(self.known_float, '%0.2f', attribute=None)
        known_val = '123.46'
        assert_equal(test_val, known_val)

    def test_stringify_None(self):
        test_val = misc.stringify(self.dataset, '%d', attribute='inflow.stats.nonething')
        known_val = '--'
        assert_equal(test_val, known_val)


class test_normalize_units(object):
    def setup(self):
        data_csv = StringIO("""\
        storm,param,station,units,conc
        1,"Lead, Total",inflow,ug/L,10
        2,"Lead, Total",inflow,mg/L,0.02
        3,"Lead, Total",inflow,g/L,0.00003
        4,"Lead, Total",inflow,ug/L,40
        1,"Lead, Total",outflow,mg/L,0.05
        2,"Lead, Total",outflow,ug/L,60
        3,"Lead, Total",outflow,ug/L,70
        4,"Lead, Total",outflow,g/L,0.00008""")
        self.raw_data = pandas.read_csv(data_csv)

        self.units_map = {
            'ug/L': 1e-6,
            'mg/L': 1e-3,
            'g/L' : 1e+0,
        }

        self.params = {
            "Lead, Total": 1e-6
        }
        self.known_conc = np.array([ 10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.])

    def test_normalize_units(self):
        data = misc.normalize_units(self.raw_data, self.units_map, 'ug/L',
                                     paramcol='param', rescol='conc',
                                     unitcol='units')
        nptest.assert_array_equal(self.known_conc, data['conc'].values)


    @nptest.raises(ValueError)
    def test_normalize_units(self):
        data = misc.normalize_units(self.raw_data, self.units_map, 'ng/L',
                                     paramcol='param', rescol='conc',
                                     unitcol='units')
        nptest.assert_array_equal(self.known_conc, data['conc'].values)


    def test_normalize_units2_nodlcol(self):
        normalize = self.units_map.get
        convert = self.params.get
        unit = lambda x: 'ug/L'
        data = misc.normalize_units2(self.raw_data, normalize, convert,
                                     unit, paramcol='param', rescol='conc',
                                     unitcol='units')
        nptest.assert_array_almost_equal(self.known_conc, data['conc'].values)


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
            assert_equal(test, known)

        @nptest.raises(misc.DataError)
        def test_getUniqueDataframeIndexVal_error():
            misc.getUniqueDataframeIndexVal(self.data, 'date')



def test_pH2concentration_normal():
    test_val = misc.pH2concentration(4)
    known_val = 0.10072764682551091
    assert_almost_equal(test_val, known_val)


@nptest.raises(ValueError)
def test_pH2concentration_raises_high():
    misc.pH2concentration(14.1)


@nptest.raises(ValueError)
def test_pH2concentration_raises_low():
    misc.pH2concentration(-0.1)


class test_estimateLineFromParams(object):
    def setup(self):
        self.x = np.arange(1, 11, 0.5)
        self.slope = 2
        self.intercept = 3.5

        self.known_ylinlin = np.array([
             5.5,   6.5,   7.5,   8.5,   9.5,  10.5,  11.5,  12.5,  13.5,
            14.5,  15.5,  16.5,  17.5,  18.5,  19.5,  20.5,  21.5,  22.5,
            23.5,  24.5
        ])


        self.known_yloglin = np.array([
            3.5       ,  4.31093022,  4.88629436,  5.33258146,  5.69722458,
            6.00552594,  6.27258872,  6.50815479,  6.71887582,  6.90949618,
            7.08351894,  7.24360435,  7.3918203 ,  7.52980604,  7.65888308,
            7.78013233,  7.89444915,  8.0025836 ,  8.10517019,  8.20275051
        ])

        self.known_yloglog = np.array([
              33.11545196,    74.50976691,   132.46180783,   206.97157474,
             298.03906763,   405.66428649,   529.84723134,   670.58790216,
             827.88629897,  1001.74242175,  1192.15627051,  1399.12784525,
            1622.65714598,  1862.74417268,  2119.38892536,  2392.59140402,
            2682.35160865,  2988.66953927,  3311.54519587,  3650.97857845
        ])

        self.known_ylinlog = np.array([
             2.44691932e+02,   6.65141633e+02,   1.80804241e+03,
             4.91476884e+03,   1.33597268e+04,   3.63155027e+04,
             9.87157710e+04,   2.68337287e+05,   7.29416370e+05,
             1.98275926e+06,   5.38969848e+06,   1.46507194e+07,
             3.98247844e+07,   1.08254988e+08,   2.94267566e+08,
             7.99902177e+08,   2.17435955e+09,   5.91052206e+09,
             1.60664647e+10,   4.36731791e+10
         ])

    def test_linlin(self):
        ylinlin = misc.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=False, ylog=False)
        nptest.assert_array_almost_equal(ylinlin, self.known_ylinlin)

    def test_loglin(self):
        yloglin = misc.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=True, ylog=False)
        nptest.assert_array_almost_equal(yloglin, self.known_yloglin)

    def test_loglog(self):
        yloglog = misc.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=True, ylog=True)
        nptest.assert_array_almost_equal(yloglog, self.known_yloglog)

    def test_linlog(self):
        ylinlog = misc.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=False, ylog=True)
        percent_diff = np.abs(ylinlog - self.known_ylinlog) / self.known_ylinlog
        nptest.assert_array_almost_equal(
            percent_diff,
            np.zeros(self.x.shape[0]),
            decimal=5
        )


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
        assert_true(newdf.equals(knowndf))

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
        assert_true(newdf.equals(knowndf))

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
        assert_true(newdf.equals(knowndf))

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
        assert_true(newdf.equals(knowndf))


def test_checkIntervalOverlap_oneway():
    assert_true(not misc.checkIntervalOverlap([1, 2], [3, 4], oneway=True))
    assert_true(not misc.checkIntervalOverlap([1, 4], [2, 3], oneway=True))
    assert_true(misc.checkIntervalOverlap([1, 3], [2, 4], oneway=True))

def test_checkIntervalOverlap_twoway():
    assert_true(not misc.checkIntervalOverlap([1, 2], [3, 4], oneway=False))
    assert_true(misc.checkIntervalOverlap([1, 4], [2, 3], oneway=False))
    assert_true(misc.checkIntervalOverlap([1, 3], [2, 4], oneway=False))

def test_sanitizeTex():
    inputstring = r""" \
    \$x\_\{4\}\textasciicircum\{2\} \textbackslashtimes \% ug/L$ \textbackslash \\
    """
    desiredstring = r""" \
    $x_{4}^{2} \times \% \si[per-mode=symbol]{\micro\gram\per\liter}$  \tabularnewline
    """

    assert_equal(misc.sanitizeTex(inputstring), desiredstring)


class test_makeTimestamp(object):
    def setup(self):
        warnings.resetwarnings()
        warnings.simplefilter("always")
        self.known_tstamp = pandas.Timestamp('2012-05-25 16:54')
        self.known_tstamp_fbdate = pandas.Timestamp('1901-01-01 16:54')
        self.known_tstamp_fbtime = pandas.Timestamp('2012-05-25 00:00')
        self.known_tstamp_fbboth = pandas.Timestamp('1901-01-01 00:00')

    def teardown(self):
        warnings.resetwarnings()

    def test_default_cols(self):
        row = {'sampledate': '2012-05-25', 'sampletime': '16:54'}
        tstamp = misc.makeTimestamp(row)
        assert_equal(self.known_tstamp, tstamp)

    def test_custom_cols(self):
        row = {'mydate': '2012-05-25', 'mytime': '16:54'}
        tstamp = misc.makeTimestamp(row, datecol='mydate', timecol='mytime')
        assert_equal(self.known_tstamp, tstamp)

    def test_fallback_date(self):
        row = {'sampledate': None, 'sampletime': '16:54'}
        tstamp = misc.makeTimestamp(row)
        assert_equal(self.known_tstamp_fbdate, tstamp)

    def test_fallback_time(self):
        row = {'sampledate': '2012-05-25', 'sampletime': None}
        tstamp = misc.makeTimestamp(row)
        assert_equal(self.known_tstamp_fbtime, tstamp)

    def test_fallback_both(self):
        row = {'sampledate': None, 'sampletime': None}
        tstamp = misc.makeTimestamp(row)
        assert_equal(self.known_tstamp_fbboth, tstamp)
