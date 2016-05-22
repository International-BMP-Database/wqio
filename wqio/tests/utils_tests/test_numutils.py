from collections import namedtuple

from six import StringIO
import numpy
from scipy import stats
import pandas
import statsmodels.api as sm

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

from wqio.utils import numutils


class base_sigfigsMixin(object):
    def teardown(self):
        pass

    def test_baseline(self):
        assert numutils.sigFigs(self.x, 3) == self.known_3
        assert numutils.sigFigs(self.x, 4) == self.known_4

    def test_trailing_zeros(self):
        assert numutils.sigFigs(self.x, 8) == self.known_8

    def test_sigFigs_zero_n(self):
        with pytest.raises(ValueError):
            numutils.sigFigs(self.x, 0)

    def test_sigFigs_negative_n(self):
        with pytest.raises(ValueError):
            numutils.sigFigs(self.x, -1)

    def test_exp_notex(self):
        assert numutils.sigFigs(self.x * self.factor, 3, tex=False) == self.known_exp3

    def test_exp_tex(self):
        assert numutils.sigFigs(self.x * self.factor, 3, tex=True) == self.known_exp3_tex

    def test_forceint(self):
        assert numutils.sigFigs(self.x, 3, forceint=True) == self.known_int


class Test_sigfig_gt1(base_sigfigsMixin):
    def setup(self):
        self.x = 1234.56
        self.known_3 = '1,230'
        self.known_4 = '1,235'
        self.known_8 = '1,234.5600'
        self.known_exp3 = '1.23e+08'
        self.known_exp3_tex = r'$1.23 \times 10 ^ {8}$'
        self.known_int = '1,235'
        self.factor = 10**5


class Test_sigfig_lt1(base_sigfigsMixin):
    def setup(self):
        self.x = 0.123456
        self.known_3 = '0.123'
        self.known_4 = '0.1235'
        self.known_8 = '0.12345600'
        self.known_exp3 = '1.23e-07'
        self.known_exp3_tex = r'$1.23 \times 10 ^ {-7}$'
        self.known_int = '0'
        self.factor = 10**-6

    def test_sigFigs_pvals_noop(self):
        p = 0.001
        assert numutils.sigFigs(p, 1, pval=True) == '0.001'

    def test_sigFigs_pvals_op(self):
        p = 0.0005
        assert numutils.sigFigs(p, 3, pval=True) == '<0.001'

    def test_sigFigs_pvals_op_tex(self):
        p = 0.0005
        assert numutils.sigFigs(p, 3, tex=True, pval=True) == '$<0.001$'


class Test_formatResult(object):
    def setup(self):
        self.big_num = 12498.124
        self.med_num = 12.540
        self.small_num = 0.00257

    def test_big_3(self):
        assert numutils.formatResult(self.big_num, '<', 3) == '<12,500'

    def test_med_6(self):
        assert numutils.formatResult(self.med_num, '>', 6) == '>12.5400'

    def test_small_2(self):
        assert numutils.formatResult(self.small_num, '=', 2) == '=0.0026'

    def test_med_no_qual_3(self):
        assert numutils.formatResult(self.med_num, '', 3) == '12.5'


def test_process_p_value():
    assert numutils.process_p_vals(None) == 'NA'
    assert numutils.process_p_vals(0.0009) == '<0.001'
    assert numutils.process_p_vals(0.001) == '0.001'
    assert numutils.process_p_vals(0.0012) == '0.001'
    assert numutils.process_p_vals(0.01) == '0.010'
    with pytest.raises(ValueError):
        numutils.process_p_vals(1.001)


class Test_processAndersonDarlingResults(object):
    def setup(self):
        fieldnames = ['statistic', 'critical_values', 'significance_level']
        AndersonResult = namedtuple('AndersonResult', fieldnames)
        self.good = AndersonResult(
            statistic=0.30194681312357829,
            critical_values=numpy.array([0.529, 0.602, 0.722, 0.842, 1.002]),
            significance_level=numpy.array([15., 10., 5., 2.5, 1.])
        )

        self.bad = AndersonResult(
            statistic=numpy.inf,
            critical_values=numpy.array([ 0.907,  1.061,  1.32 ,  1.58 ,  1.926]),
            significance_level=numpy.array([ 15. ,  10. ,   5. ,   2.5,   1. ])
        )

        self.known_good = '99.0%'
        self.known_bad = '<85.0%'

    def test_good(self):
        res = numutils.processAndersonDarlingResults(self.good)
        assert res == self.known_good

    def test_bad(self):
        res = numutils.processAndersonDarlingResults(self.bad)
        assert res == self.known_bad


class Test_normalize_units(object):
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
        self.known_conc = numpy.array([ 10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.])

    def test_normalize_units(self):
        data = numutils.normalize_units(self.raw_data, self.units_map, 'ug/L',
                                     paramcol='param', rescol='conc',
                                     unitcol='units')
        nptest.assert_array_equal(self.known_conc, data['conc'].values)


    def test_normalize_units(self):
        with pytest.raises(ValueError):
            data = numutils.normalize_units(self.raw_data, self.units_map, 'ng/L',
                                         paramcol='param', rescol='conc',
                                         unitcol='units')
            nptest.assert_array_equal(self.known_conc, data['conc'].values)


    def test_normalize_units2_nodlcol(self):
        normalize = self.units_map.get
        convert = self.params.get
        unit = lambda x: 'ug/L'
        data = numutils.normalize_units2(self.raw_data, normalize, convert,
                                     unit, paramcol='param', rescol='conc',
                                     unitcol='units')
        nptest.assert_array_almost_equal(self.known_conc, data['conc'].values)


class Test_test_pH2concentration(object):
    def setup(self):
        self.pH = 4
        self.known_conc = 0.10072764682551091

    def test_pH2concentration_normal(self):
        assert abs(numutils.pH2concentration(self.pH) - self.known_conc) < 0.0001

    def test_pH2concentration_raises_high(self):
        with pytest.raises(ValueError):
            numutils.pH2concentration(14.1)

    def test_pH2concentration_raises_low(self):
        with pytest.raises(ValueError):
            numutils.pH2concentration(-0.1)


class Test_fit_line(object):
    def setup(self):
        self.data = numpy.array([
            2.00,   4.0 ,   4.62,   5.00,   5.00,   5.50,   5.57,   5.66,
            5.75,   5.86,   6.65,   6.78,   6.79,   7.50,   7.50,   7.50,
            8.63,   8.71,   8.99,   9.50,   9.50,   9.85,  10.82,  11.00,
           11.25,  11.25,  12.20,  14.92,  16.77,  17.81,  19.16,  19.19,
           19.64,  20.18,  22.97
        ])

        self.zscores = numpy.array([
            -2.06188401, -1.66883254, -1.4335397 , -1.25837339, -1.11509471,
            -0.99166098, -0.8817426 , -0.78156696, -0.68868392, -0.60139747,
            -0.51847288, -0.4389725 , -0.36215721, -0.28742406, -0.21426459,
            -0.14223572, -0.07093824,  0.00000000,  0.07093824,  0.14223572,
             0.21426459,  0.28742406,  0.36215721,  0.43897250,  0.51847288,
             0.60139747,  0.68868392,  0.78156696,  0.88174260,  0.99166098,
             1.11509471,  1.25837339,  1.43353970,  1.66883254,  2.06188401
        ])

        self.probs = stats.norm.cdf(self.zscores) * 100.

        self.y = numpy.array([
            0.07323274,  0.12319301,  0.16771455,  0.1779695 ,  0.21840761,
            0.25757016,  0.2740265 ,  0.40868106,  0.44872637,  0.5367353 ,
            0.55169933,  0.56211726,  0.62375442,  0.66631353,  0.68454978,
            0.72137134,  0.87602096,  0.94651962,  1.01927875,  1.06040448,
            1.07966792,  1.17969506,  1.21132273,  1.30751428,  1.45371899,
            1.76381932,  1.98832275,  2.09275652,  2.66552831,  2.86453334,
            3.23039631,  4.23953492,  4.25892247,  4.5834766 ,  6.53100725
        ])

        self.known_y_linlin = numpy.array([ -0.89650596,  21.12622025])
        self.known_y_linlog = numpy.array([  2.80190754,  27.64958934])
        self.known_y_linprob = numpy.array([  8.48666156,  98.51899616])
        self.known_y_loglin = numpy.array([-2.57620461,  1.66767934])
        self.known_y_loglog = numpy.array([ 0.0468154 ,  5.73261406])
        self.known_y_logprob = numpy.array([  0.49945757,  95.23103009])
        self.known_y_problin = numpy.array([ -0.89650596,  21.12622025])
        self.known_y_problog = numpy.array([  2.80190754,  27.64958934])
        self.known_y_probprob = numpy.array([  1.96093902,  98.03906098])

        self.custom_xhat = [-2, -1, 0, 1, 2]
        self.known_custom_yhat = numpy.array([-0.56601826, 4.77441944, 10.11485714,
                                           15.45529485, 20.79573255])

    def test_xlinear_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linlin)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xlinear_ylog(self):
        scales = {'fitlogs': 'y', 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linlog)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xlinear_yprob(self):
        scales = {'fitlogs': None, 'fitprobs': 'y'}
        x, y = self.data, self.probs
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linprob)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xlog_ylinear(self):
        scales = {'fitlogs': 'x', 'fitprobs': None}
        x, y = self.data, self.zscores
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_loglin)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xlog_ylog(self):
        scales = {'fitlogs': 'both', 'fitprobs': None}
        x, y = self.data, self.y
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_loglog)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xlog_yprob(self):
        scales = {'fitlogs': 'x', 'fitprobs': 'y'}
        x, y = self.data, self.probs
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_logprob)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xprob_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': 'x'}
        x, y = self.probs, self.data
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_problin)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xprob_ylog(self):
        scales = {'fitlogs': 'y', 'fitprobs': 'x'}
        x, y = self.probs, self.data
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_problog)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xprob_yprob(self):
        z2, _y = stats.probplot(self.y, fit=False)
        p2 = stats.norm.cdf(z2) * 100

        scales = {'fitlogs': None, 'fitprobs': 'both'}
        x, y = self.probs, p2,
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_probprob)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_bad_fitlogs(self):
        with pytest.raises(ValueError):
            x, y = self.zscores, self.data
            x_, y_, res = numutils.fit_line(x, y, fitlogs='junk')

    def test_bad_fitprobs(self):
        with pytest.raises(ValueError):
            x, y = self.zscores, self.data
            x_, y_, res = numutils.fit_line(x, y, fitprobs='junk')

    def test_custom_xhat(self):
        x, y = self.zscores, self.data
        x_, y_, res = numutils.fit_line(x, y, xhat=self.custom_xhat)
        nptest.assert_array_almost_equal(y_, self.known_custom_yhat)


class Test_estimateLineFromParams(object):
    def setup(self):
        self.x = numpy.arange(1, 11, 0.5)
        self.slope = 2
        self.intercept = 3.5

        self.known_ylinlin = numpy.array([
             5.5,   6.5,   7.5,   8.5,   9.5,  10.5,  11.5,  12.5,  13.5,
            14.5,  15.5,  16.5,  17.5,  18.5,  19.5,  20.5,  21.5,  22.5,
            23.5,  24.5
        ])


        self.known_yloglin = numpy.array([
            3.5       ,  4.31093022,  4.88629436,  5.33258146,  5.69722458,
            6.00552594,  6.27258872,  6.50815479,  6.71887582,  6.90949618,
            7.08351894,  7.24360435,  7.3918203 ,  7.52980604,  7.65888308,
            7.78013233,  7.89444915,  8.0025836 ,  8.10517019,  8.20275051
        ])

        self.known_yloglog = numpy.array([
              33.11545196,    74.50976691,   132.46180783,   206.97157474,
             298.03906763,   405.66428649,   529.84723134,   670.58790216,
             827.88629897,  1001.74242175,  1192.15627051,  1399.12784525,
            1622.65714598,  1862.74417268,  2119.38892536,  2392.59140402,
            2682.35160865,  2988.66953927,  3311.54519587,  3650.97857845
        ])

        self.known_ylinlog = numpy.array([
             2.44691932e+02,   6.65141633e+02,   1.80804241e+03,
             4.91476884e+03,   1.33597268e+04,   3.63155027e+04,
             9.87157710e+04,   2.68337287e+05,   7.29416370e+05,
             1.98275926e+06,   5.38969848e+06,   1.46507194e+07,
             3.98247844e+07,   1.08254988e+08,   2.94267566e+08,
             7.99902177e+08,   2.17435955e+09,   5.91052206e+09,
             1.60664647e+10,   4.36731791e+10
         ])

    def test_linlin(self):
        ylinlin = numutils.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=False, ylog=False)
        nptest.assert_array_almost_equal(ylinlin, self.known_ylinlin)

    def test_loglin(self):
        yloglin = numutils.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=True, ylog=False)
        nptest.assert_array_almost_equal(yloglin, self.known_yloglin)

    def test_loglog(self):
        yloglog = numutils.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=True, ylog=True)
        nptest.assert_array_almost_equal(yloglog, self.known_yloglog)

    def test_linlog(self):
        ylinlog = numutils.estimateFromLineParams(self.x, self.slope, self.intercept,
                                              xlog=False, ylog=True)
        percent_diff = numpy.abs(ylinlog - self.known_ylinlog) / self.known_ylinlog
        nptest.assert_array_almost_equal(
            percent_diff,
            numpy.zeros(self.x.shape[0]),
            decimal=5
        )


def test_checkIntervalOverlap_oneway():
    assert not numutils.checkIntervalOverlap([1, 2], [3, 4], oneway=True)
    assert not numutils.checkIntervalOverlap([1, 4], [2, 3], oneway=True)
    assert numutils.checkIntervalOverlap([1, 3], [2, 4], oneway=True)


def test_checkIntervalOverlap_twoway():
    assert not numutils.checkIntervalOverlap([1, 2], [3, 4], oneway=False)
    assert numutils.checkIntervalOverlap([1, 4], [2, 3], oneway=False)
    assert numutils.checkIntervalOverlap([1, 3], [2, 4], oneway=False)


class Test_winsorize_dataframe(object):
    def setup(self):
        self.x = numpy.array([
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        ])

        self.w_05 = numpy.array([
            1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 19
        ])

        self.w_10 = numpy.array([
            2,  2,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 18, 18
        ])

        self.w_20 = numpy.array([
            4,  4,  4,  4,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 16, 16, 16, 16
        ])

        self.w_05_20 = numpy.array([
            1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 16, 16, 16, 16
        ])

        self.df = pandas.DataFrame({'A': self.x, 'B': self.x, 'C': self.x})

    def test_no_op(self):
        w = numutils.winsorize_dataframe(self.df)
        expected = self.df.copy()
        pdtest.assert_frame_equal(w, expected)

    def test_one_col(self):
        w = numutils.winsorize_dataframe(self.df, A=0.05)
        expected = pandas.DataFrame({'A': self.w_05, 'B': self.x, 'C': self.x})
        pdtest.assert_frame_equal(w, expected)

    def test_two_col(self):
        w = numutils.winsorize_dataframe(self.df, A=0.05, C=0.10)
        expected = pandas.DataFrame({'A': self.w_05, 'B': self.x, 'C': self.w_10})
        pdtest.assert_frame_equal(w, expected)

    def test_three_col(self):
        w = numutils.winsorize_dataframe(self.df, A=0.20, C=0.10, B=0.20)
        expected = pandas.DataFrame({'A': self.w_20,'B': self.w_20, 'C': self.w_10})
        pdtest.assert_frame_equal(w, expected)

    def test_tuple_limit(self):
        w = numutils.winsorize_dataframe(self.df, A=(0.05, 0.20), C=0.10, B=0.20)
        expected = pandas.DataFrame({'A': self.w_05_20, 'B': self.w_20, 'C': self.w_10})
        pdtest.assert_frame_equal(w, expected)
