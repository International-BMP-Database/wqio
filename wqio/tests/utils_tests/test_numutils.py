import types
from collections import namedtuple
from io import StringIO
from textwrap import dedent

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest
from wqio.tests import helpers

import numpy
from scipy import stats
import pandas
import statsmodels.api as sm

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
        with helpers.raises(ValueError):
            numutils.sigFigs(self.x, 0)

    def test_sigFigs_negative_n(self):
        with helpers.raises(ValueError):
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


@pytest.mark.parametrize(('fxn', 'pval', 'expected', 'error_to_raise'), [
    (numutils.process_p_vals, None, 'NA', None),
    (numutils.process_p_vals, 0.0005, '<0.001', None),
    (numutils.process_p_vals, 0.005, '0.005', None),
    (numutils.process_p_vals, 0.001, '0.001', None),
    (numutils.process_p_vals, 0.0012, '0.001', None),
    (numutils.process_p_vals, 0.06, '0.060', None),
    (numutils.process_p_vals, 1.01, None, ValueError),
    (numutils.translate_p_vals, None, 'ಠ_ಠ', None),
    (numutils.translate_p_vals, 0.005, '¯\(ツ)/¯', None),
    (numutils.translate_p_vals, 0.03, '¯\_(ツ)_/¯', None),
    (numutils.translate_p_vals, 0.06, '¯\__(ツ)__/¯', None),
    (numutils.translate_p_vals, 0.11, '(╯°□°)╯︵ ┻━┻', None),
    (numutils.translate_p_vals, 1.01, None, ValueError),
])
def test_p_values_handlers(fxn, pval, expected, error_to_raise):
    with helpers.raises(error_to_raise):
        result = fxn(pval)
        assert result == expected


def test_anderson_darling():
    data = helpers.getTestROSData()
    result = numutils.anderson_darling(data['res'])

    tolerance = 0.05
    known_anderson = (
        1.4392085,
        [0.527, 0.6, 0.719, 0.839, 0.998],
        [15., 10., 5., 2.5, 1.],
        0.00080268
    )

    # Ad stat
    assert abs(result[0] - known_anderson[0]) < 0.0001

    # critical values
    nptest.assert_allclose(result[1], known_anderson[1], rtol=tolerance)

    # significance level
    nptest.assert_allclose(result[2], known_anderson[2], rtol=tolerance)

    # p-value
    assert abs(result[3] - known_anderson[3]) < 0.0000001


class Test_processAndersonDarlingResults(object):
    def setup(self):
        fieldnames = ['statistic', 'critical_values', 'significance_level']
        AndersonResult = namedtuple('AndersonResult', fieldnames)
        self.bad = AndersonResult(
            statistic=0.30194681312357829,
            critical_values=numpy.array([0.529, 0.602, 0.722, 0.842, 1.002]),
            significance_level=numpy.array([15., 10., 5., 2.5, 1.])
        )

        self.good = AndersonResult(
            statistic=numpy.inf,
            critical_values=numpy.array([0.907, 1.061, 1.32, 1.58, 1.926]),
            significance_level=numpy.array([15.0, 10.0, 5.0, 2.50, 1.0])
        )

        self.known_good = '99.0%'
        self.known_bad = '<85.0%'

    def test_good(self):
        res = numutils.processAndersonDarlingResults(self.good)
        assert res == self.known_good

    def test_bad(self):
        res = numutils.processAndersonDarlingResults(self.bad)
        assert res == self.known_bad


@pytest.mark.parametrize(('AD', 'n_points', 'expected'), [
    (0.302, 1, 0.0035899),
    (0.302, 5, 0.4155200),
    (0.302, 15, 0.5325205),
    (0.150, 105, 0.9616770),
])
def test__anderson_darling_p_vals(AD, n_points, expected):
    ad_result = (AD, None, None)
    p_val = numutils._anderson_darling_p_vals(ad_result, n_points)
    assert abs(p_val - expected) < 0.0001


@pytest.fixture
def units_norm_data():
    data_csv = StringIO(dedent("""\
        storm,param,station,units,conc
        1,"Lead, Total",inflow,ug/L,10
        2,"Lead, Total",inflow,mg/L,0.02
        3,"Lead, Total",inflow,g/L,0.00003
        4,"Lead, Total",inflow,ug/L,40
        1,"Lead, Total",outflow,mg/L,0.05
        2,"Lead, Total",outflow,ug/L,60
        3,"Lead, Total",outflow,ug/L,70
        4,"Lead, Total",outflow,g/L,0.00008
        1,"Cadmium, Total",inflow,ug/L,10
        2,"Cadmium, Total",inflow,mg/L,0.02
        3,"Cadmium, Total",inflow,g/L,0.00003
        4,"Cadmium, Total",inflow,ug/L,40
        1,"Cadmium, Total",outflow,mg/L,0.05
        2,"Cadmium, Total",outflow,ug/L,60
        3,"Cadmium, Total",outflow,ug/L,70
        4,"Cadmium, Total",outflow,g/L,0.00008
    """))
    raw = pandas.read_csv(data_csv)

    known_csv = StringIO(dedent("""\
        storm,param,station,units,conc
        1,"Lead, Total",inflow,ug/L,10.
        2,"Lead, Total",inflow,ug/L,20.
        3,"Lead, Total",inflow,ug/L,30.
        4,"Lead, Total",inflow,ug/L,40.
        1,"Lead, Total",outflow,ug/L,50.
        2,"Lead, Total",outflow,ug/L,60.
        3,"Lead, Total",outflow,ug/L,70.
        4,"Lead, Total",outflow,ug/L,80.
        1,"Cadmium, Total",inflow,mg/L,0.010
        2,"Cadmium, Total",inflow,mg/L,0.020
        3,"Cadmium, Total",inflow,mg/L,0.030
        4,"Cadmium, Total",inflow,mg/L,0.040
        1,"Cadmium, Total",outflow,mg/L,0.050
        2,"Cadmium, Total",outflow,mg/L,0.060
        3,"Cadmium, Total",outflow,mg/L,0.070
        4,"Cadmium, Total",outflow,mg/L,0.080
    """))
    expected = pandas.read_csv(known_csv)
    return raw, expected


def test_normalize_units(units_norm_data):
    unitsmap = {
        'ug/L': 1e-6,
        'mg/L': 1e-3,
        'g/L': 1e+0,
    }

    targetunits = {
        "Lead, Total": 'ug/L',
        "Cadmium, Total": 'mg/L',
    }
    raw, expected = units_norm_data
    result = numutils.normalize_units(raw, unitsmap, targetunits,
                                      paramcol='param', rescol='conc',
                                      unitcol='units')
    pdtest.assert_frame_equal(result, expected)


def test_normalize_units_bad_targetunits(units_norm_data):
    unitsmap = {
        'ug/L': 1e-6,
        'mg/L': 1e-3,
        'g/L': 1e+0,
    }

    targetunits = {
        "Lead, Total": 'ug/L',
    }
    raw, expected = units_norm_data
    with helpers.raises(ValueError):
        numutils.normalize_units(raw, unitsmap, targetunits,
                                 paramcol='param', rescol='conc',
                                 unitcol='units', napolicy='raise')


def test_normalize_units_bad_normalization(units_norm_data):
    unitsmap = {
        'mg/L': 1e-3,
        'g/L': 1e+0,
    }

    targetunits = {
        "Lead, Total": 'ug/L',
        "Cadmium, Total": 'mg/L',
    }
    raw, expected = units_norm_data
    with helpers.raises(ValueError):
        numutils.normalize_units(raw, unitsmap, targetunits,
                                 paramcol='param', rescol='conc',
                                 unitcol='units', napolicy='raise')


def test_normalize_units_bad_conversion(units_norm_data):
    unitsmap = {
        'ug/L': 1e-6,
        'mg/L': 1e-3,
        'g/L': 1e+0,
    }

    targetunits = {
        "Lead, Total": 'ng/L',
        "Cadmium, Total": 'mg/L',
    }
    raw, expected = units_norm_data
    with helpers.raises(ValueError):
        numutils.normalize_units(raw, unitsmap, targetunits,
                                 paramcol='param', rescol='conc',
                                 unitcol='units', napolicy='raise')


class Test_test_pH2concentration(object):
    def setup(self):
        self.pH = 4
        self.known_conc = 0.10072764682551091

    def test_pH2concentration_normal(self):
        assert abs(numutils.pH2concentration(self.pH) - self.known_conc) < 0.0001

    def test_pH2concentration_raises_high(self):
        with helpers.raises(ValueError):
            numutils.pH2concentration(14.1)

    def test_pH2concentration_raises_low(self):
        with helpers.raises(ValueError):
            numutils.pH2concentration(-0.1)


class Test_fit_line(object):
    def setup(self):
        self.data = numpy.array([
            2.000, 4.000, 4.620, 5.000, 5.000, 5.500, 5.570, 5.660,
            5.750, 5.860, 6.650, 6.780, 6.790, 7.500, 7.500, 7.500,
            8.630, 8.710, 8.990, 9.500, 9.500, 9.850, 10.82, 11.00,
            11.25, 11.25, 12.20, 14.92, 16.77, 17.81, 19.16, 19.19,
            19.64, 20.18, 22.97
        ])

        self.zscores = numpy.array([
            -2.06188401, -1.66883254, -1.43353970, -1.25837339, -1.11509471,
            -0.99166098, -0.88174260, -0.78156696, -0.68868392, -0.60139747,
            -0.51847288, -0.43897250, -0.36215721, -0.28742406, -0.21426459,
            -0.14223572, -0.07093824, 0.00000000, 0.07093824, 0.14223572,
            0.21426459, 0.28742406, 0.36215721, 0.43897250, 0.51847288,
            0.60139747, 0.68868392, 0.78156696, 0.88174260, 0.99166098,
            1.11509471, 1.25837339, 1.43353970, 1.66883254, 2.06188401
        ])

        self.probs = stats.norm.cdf(self.zscores) * 100.

        self.y = numpy.array([
            0.07323274, 0.12319301, 0.16771455, 0.17796950, 0.21840761,
            0.25757016, 0.27402650, 0.40868106, 0.44872637, 0.53673530,
            0.55169933, 0.56211726, 0.62375442, 0.66631353, 0.68454978,
            0.72137134, 0.87602096, 0.94651962, 1.01927875, 1.06040448,
            1.07966792, 1.17969506, 1.21132273, 1.30751428, 1.45371899,
            1.76381932, 1.98832275, 2.09275652, 2.66552831, 2.86453334,
            3.23039631, 4.23953492, 4.25892247, 4.58347660, 6.53100725
        ])

        self.known_y_linlin = numpy.array([-0.89650596, 21.12622025])
        self.known_y_linlog = numpy.array([2.80190754, 27.64958934])
        self.known_y_linprob = numpy.array([8.48666156, 98.51899616])
        self.known_y_loglin = numpy.array([-2.57620461, 1.66767934])
        self.known_y_loglog = numpy.array([0.04681540, 5.73261406])
        self.known_y_logprob = numpy.array([0.49945757, 95.23103009])
        self.known_y_problin = numpy.array([-0.89650596, 21.12622025])
        self.known_y_problog = numpy.array([2.80190754, 27.64958934])
        self.known_y_probprob = numpy.array([1.96093902, 98.03906098])

        self.custom_xhat = [-2, -1, 0, 1, 2]
        self.known_custom_yhat = numpy.array([-0.56601826, 4.77441944, 10.11485714,
                                              15.45529485, 20.79573255])

    def test_xlinear_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = numutils.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linlin)
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)

    def test_xlinear_ylinear_through_origin(self):
        scales = {'fitlogs': None, 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = numutils.fit_line(x, y, through_origin=True, **scales)
        assert res.params[0] == 0

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
        with helpers.raises(ValueError):
            x, y = self.zscores, self.data
            x_, y_, res = numutils.fit_line(x, y, fitlogs='junk')

    def test_bad_fitprobs(self):
        with helpers.raises(ValueError):
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
            5.50, 6.50, 7.50, 8.50, 9.50, 10.5, 11.5, 12.5, 13.5,
            14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5,
            23.5, 24.5
        ])

        self.known_yloglin = numpy.array([
            3.50000000, 4.31093022, 4.88629436, 5.33258146, 5.69722458,
            6.00552594, 6.27258872, 6.50815479, 6.71887582, 6.90949618,
            7.08351894, 7.24360435, 7.39182030, 7.52980604, 7.65888308,
            7.78013233, 7.89444915, 8.00258360, 8.10517019, 8.20275051
        ])

        self.known_yloglog = numpy.array([
            33.1154519600, 74.5097669100, 132.461807830, 206.971574740,
            298.039067630, 405.664286490, 529.847231340, 670.587902160,
            827.886298970, 1001.74242175, 1192.15627051, 1399.12784525,
            1622.65714598, 1862.74417268, 2119.38892536, 2392.59140402,
            2682.35160865, 2988.66953927, 3311.54519587, 3650.97857845
        ])

        self.known_ylinlog = numpy.array([
            2.44691932e+02, 6.65141633e+02, 1.80804241e+03,
            4.91476884e+03, 1.33597268e+04, 3.63155027e+04,
            9.87157710e+04, 2.68337287e+05, 7.29416370e+05,
            1.98275926e+06, 5.38969848e+06, 1.46507194e+07,
            3.98247844e+07, 1.08254988e+08, 2.94267566e+08,
            7.99902177e+08, 2.17435955e+09, 5.91052206e+09,
            1.60664647e+10, 4.36731791e+10
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
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        ])

        self.w_05 = numpy.array([
            1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 19
        ])

        self.w_10 = numpy.array([
            2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 18, 18
        ])

        self.w_20 = numpy.array([
            4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 16, 16, 16, 16
        ])

        self.w_05_20 = numpy.array([
            1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
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
        expected = pandas.DataFrame({'A': self.w_20, 'B': self.w_20, 'C': self.w_10})
        pdtest.assert_frame_equal(w, expected)

    def test_tuple_limit(self):
        w = numutils.winsorize_dataframe(self.df, A=(0.05, 0.20), C=0.10, B=0.20)
        expected = pandas.DataFrame({'A': self.w_05_20, 'B': self.w_20, 'C': self.w_10})
        pdtest.assert_frame_equal(w, expected)


def test__comp_stat_generator():
    df = helpers.make_dc_data().reset_index()
    gen = numutils._comp_stat_generator(df, ['param', 'bmp'], 'loc', 'res', helpers.comp_statfxn)
    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen)

    expected = {
        'bmp': ['1', '1', '1', '1', '1', '7', '7', '7', '7', '7'],
        'param': ['A', 'A', 'A', 'A', 'A', 'H', 'H', 'H', 'H', 'H'],
        'loc_1': [
            'Inflow', 'Inflow', 'Outflow', 'Outflow', 'Reference',
            'Inflow', 'Outflow', 'Outflow', 'Reference', 'Reference',
        ],
        'loc_2': [
            'Outflow', 'Reference', 'Inflow', 'Reference', 'Inflow',
            'Reference', 'Inflow', 'Reference', 'Inflow', 'Outflow',
        ],
        'stat': [
            33.100128, 34.228203, 18.318497, 21.231383, 9.500766,
            2.990900, 7.498225, 7.757954, 3.067299, 3.290471,
        ],
        'pvalue': [
            8.275032, 8.557050, 4.579624, 5.307845, 2.375191,
            0.747725, 1.874556, 1.939488, 0.766824, 0.822617,
        ],
    }
    pdtest.assert_frame_equal(
        pandas.DataFrame(expected, index=[0, 1, 2, 3, 4, 331, 332, 333, 334, 335]),
        pandas.concat([result.head(), result.tail()])
    )


def test__comp_stat_generator_single_group_col():
    df = helpers.make_dc_data().reset_index()
    gen = numutils._comp_stat_generator(df, 'param', 'loc', 'res', helpers.comp_statfxn)
    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen)

    expected = {
        'stat': [
            35.78880963, 36.032519607, 24.07328047, 24.151276229, 13.107839675,
            16.69316696, 14.336154663, 14.33615466, 11.298920232, 11.298920232
        ],
        'loc_1': [
            'Inflow', 'Inflow', 'Outflow', 'Outflow', 'Reference',
            'Inflow', 'Outflow', 'Outflow', 'Reference', 'Reference'
        ],
        'loc_2': [
            'Outflow', 'Reference', 'Inflow', 'Reference', 'Inflow',
            'Reference', 'Inflow', 'Reference', 'Inflow', 'Outflow'
        ],
        'param': ['A', 'A', 'A', 'A', 'A', 'H', 'H', 'H', 'H', 'H'],
        'pvalue': [
            8.94720240, 9.008129901, 6.018320119, 6.037819057, 3.276959918,
            4.17329174, 3.584038665, 3.584038665, 2.824730058, 2.824730058
        ]
    }
    pdtest.assert_frame_equal(
        pandas.DataFrame(expected, index=[0, 1, 2, 3, 4, 43, 44, 45, 46, 47]),
        pandas.concat([result.head(), result.tail()])
    )


def test__paired_stat_generator():
    df = helpers.make_dc_data_complex().unstack(level='loc')
    gen = numutils._paired_stat_generator(df, ['param'], 'loc', 'res', helpers.comp_statfxn)
    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen)

    expected = {
        'loc_1': [
            'Inflow', 'Inflow', 'Outflow', 'Outflow', 'Reference',
            'Inflow', 'Outflow', 'Outflow', 'Reference', 'Reference',
        ],
        'loc_2': [
            'Outflow', 'Reference', 'Inflow', 'Reference', 'Inflow',
            'Reference', 'Inflow', 'Reference', 'Inflow', 'Outflow',
        ],
        'param': [
            'A', 'A', 'A', 'A', 'A',
            'H', 'H', 'H', 'H', 'H',
        ],
        'pvalue': [
            2.688485, 3.406661, 9.084853, 9.084853, 5.243408,
            9.399253, 20.234093, 20.23409, 2.801076, 2.801075,
        ],
        'stat': [
            10.753940, 13.626645, 36.339414, 36.339414, 20.973631,
            37.597010, 80.936373, 80.936373, 11.204302, 11.204302,
        ]
    }

    pdtest.assert_frame_equal(
        pandas.DataFrame(expected, index=[0, 1, 2, 3, 4, 43, 44, 45, 46, 47]),
        pandas.concat([result.head(), result.tail()])
    )


@helpers.seed
def test_remove_outliers():
    expected_shape = (32,)
    x = numpy.random.normal(0, 4, size=37)

    assert numutils.remove_outliers(x).shape == expected_shape
