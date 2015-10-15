from nose.tools import *
import numpy.testing as nptest
import numpy as np
import scipy.optimize as opt

from wqio import testing
from wqio.algo import bootstrap


def test__acceleration():
    data = testing.getTestROSData()['res']
    known_acceleration = -0.024051865664929263
    assert_almost_equal(bootstrap._acceleration(data), known_acceleration, places=5)


class test_Stat:
    def setup(self):
        np.random.seed(0)
        self.data = testing.getTestROSData()
        self.statfxn = np.mean
        self.alpha = 0.10
        self.NIter = 2500
        self.bsStat = bootstrap.Stat(np.array(self.data.res), statfxn=self.statfxn,
                                     alpha=self.alpha, NIter=self.NIter)

        self.known_secondary = 10.121394742857142
        self.known_primary = 10.120571428571429

    def test__boot_stats(self):
        assert_true(hasattr(self.bsStat, 'boot_stats'))
        assert_equal(self.bsStat.boot_stats.shape[0], self.NIter)

    def test_BCA(self):
        assert_true(hasattr(self.bsStat, 'BCA'))
        knownBCA_res, knownBCA_ci = (10.127740624999999, np.array([8.64771461, 11.63041132]))
        BCA_res, BCA_ci = self.bsStat.BCA()
        assert_almost_equal(knownBCA_res, BCA_res, places=1)
        nptest.assert_array_almost_equal(knownBCA_ci, BCA_ci, decimal=1)

    def test_percentile(self):
        assert_true(hasattr(self.bsStat, 'percentile'))
        knownPer_res, knownPer_ci = (10.102715015411377, np.array([8.67312818, 11.65695653]))
        Per_res, Per_ci = self.bsStat.percentile()
        assert_almost_equal(knownPer_res, Per_res, places=1)
        nptest.assert_array_almost_equal(knownPer_ci, Per_ci, decimal=1)

    def test_boot_array(self):
        assert_true(hasattr(self.bsStat, 'boot_array'))
        assert_tuple_equal(self.bsStat.boot_array.shape, (self.NIter, self.data.res.shape[0]))

    def test_data(self):
        assert_true(hasattr(self.bsStat, 'data'))
        nptest.assert_array_equal(np.array(self.data.res), self.bsStat.data)

    def test_statfxn(self):
        assert_true(hasattr(self.bsStat, 'statfxn'))
        assert_equal(self.statfxn, self.bsStat.statfxn)

    def test_alpha(self):
        assert_true(hasattr(self.bsStat, 'alpha'))
        assert_equal(self.alpha, self.bsStat.alpha)

    def test_NIter(self):
        assert_true(hasattr(self.bsStat, 'NIter'))
        assert_equal(self.NIter, self.bsStat.NIter)

    def test_primary_result(self):
        assert_true(hasattr(self.bsStat, 'primary_result'))
        assert_equal(self.known_primary, self.bsStat.primary_result)

    def test_boot_result(self):
        assert_true(hasattr(self.bsStat, 'boot_result'))
        assert_equal(self.known_secondary, self.bsStat.boot_result)

    def test_final_result(self):
        assert_true(hasattr(self.bsStat, 'final_result'))
        assert_equal(self.known_primary, self.bsStat.final_result)


class test_Fit:
    def setup(self):
        self.data = testing.getTestROSData()
        self.alpha = 0.10
        self.NIter = 2500

        def cf_line(x, m, b):
            return m*x + b

        self.curvefitfxn = cf_line
        self.statfxn = opt.curve_fit
        self.bsFit = bootstrap.Fit(np.array(self.data.index), np.array(self.data.res),
                                   curvefitfxn=cf_line, alpha=self.alpha, NIter=self.NIter)

    def test__boot_stats(self):
        assert_true(hasattr(self.bsFit, 'boot_stats'))
        assert_equal(self.bsFit.boot_stats.shape[0], self.NIter)

    def test_BCA(self):
        assert_true(hasattr(self.bsFit, 'BCA'))
        knownBCA_res, knownBCA_ci = (np.array([0.49729708, 1.62350472]),
                                     np.array([[0.4344499, 0.55369941],
                                               [0.56610068, 2.37581269]]))
        BCA_res, BCA_ci = self.bsFit.BCA()
        nptest.assert_array_almost_equal(knownBCA_res, BCA_res, decimal=1)
        nptest.assert_array_almost_equal(knownBCA_ci, BCA_ci, decimal=1)

    def test_percentile(self):
        assert_true(hasattr(self.bsFit, 'percentile'))
        knownPer_res, knownPer_ci = (np.array([0.49918029, 1.66417978]),
                                     np.array([[0.43425858, 0.55355437],
                                               [0.65338572, 2.43818938]]))
        Per_res, Per_ci = self.bsFit.percentile()
        nptest.assert_array_almost_equal(knownPer_res, Per_res, decimal=1)
        nptest.assert_array_almost_equal(knownPer_ci, Per_ci, decimal=1)

    def test_boot_array(self):
        assert_true(hasattr(self.bsFit, 'boot_array'))
        assert_tuple_equal(self.bsFit.boot_array.shape, (self.NIter, self.data.res.shape[0], 2))

    def test_data(self):
        assert_true(hasattr(self.bsFit, 'data'))
        nptest.assert_array_equal(np.array(self.data.index), self.bsFit.data)

    def test_outputdata(self):
        assert_true(hasattr(self.bsFit, 'outputdata'))
        nptest.assert_array_equal(np.array(self.data.res), self.bsFit.outputdata)

    def test_curvefitfxn(self):
        assert_true(hasattr(self.bsFit, 'curvefitfxn'))
        assert_equal(self.curvefitfxn, self.bsFit.curvefitfxn)

    def test_statfxn(self):
        assert_true(hasattr(self.bsFit, 'statfxn'))
        assert_equal(self.statfxn, self.bsFit.statfxn)

    def test_alpha(self):
        assert_true(hasattr(self.bsFit, 'alpha'))
        assert_equal(self.alpha, self.bsFit.alpha)

    def test_NIter(self):
        assert_true(hasattr(self.bsFit, 'NIter'))
        assert_equal(self.NIter, self.bsFit.NIter)

    def test_primary_result(self):
        assert_true(hasattr(self.bsFit, 'primary_result'))
        nptest.assert_array_almost_equal(np.array([0.4992521, 1.63328575]),
                                         self.bsFit.primary_result)
