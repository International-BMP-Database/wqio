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

    def test_final_result(self):
        assert_true(hasattr(self.bsStat, 'final_result'))
        assert_equal(self.known_primary, self.bsStat.final_result)
