import nose.tools as nt
import numpy.testing as nptest
import numpy as np
import scipy.optimize as opt

from wqio import testing
from wqio.algo import bootstrap


def test__acceleration():
    data = testing.getTestROSData()['res']
    known_acceleration = -0.024051865664929263
    nt.assert_almost_equal(bootstrap._acceleration(data), known_acceleration, places=5)


def test__make_boot_index():
    result = bootstrap._make_boot_index(5, 5000)
    nt.assert_tuple_equal(result.shape, (5000, 5))
    nt.assert_equal(result.min(), 0)
    nt.assert_equal(result.max(), 4)


class TestBootStrappers(object):
    def setup(self):
        np.random.seed(0)
        self.data = testing.getTestROSData()['res'].values
        self.statfxn = np.mean
        self.alpha = 0.10
        self.NIter = 2500
        self.known_res = self.statfxn(self.data)
        self.known_BCA_ci = np.array([8.74244187, 11.71177557])
        self.known_pct_ci = np.array([8.70867143, 11.68634286])

    @testing.seed
    def test_BCA(self):
        res, ci = bootstrap.BCA(self.data, self.statfxn, self.NIter, self.alpha)
        nt.assert_almost_equal(self.known_res, res, places=3)
        nptest.assert_array_almost_equal(self.known_BCA_ci, ci, decimal=3)

    @testing.seed
    def test_percentile(self):
        res, ci = bootstrap.percentile(self.data, self.statfxn, self.NIter, self.alpha)
        nt.assert_almost_equal(self.known_res, res, places=3)
        nptest.assert_array_almost_equal(self.known_pct_ci, ci, decimal=3)


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


    @testing.seed
    def test_BCA(self):
        nt.assert_true(hasattr(self.bsStat, 'BCA'))
        knownBCA_ci = np.array([8.74244187, 11.71177557])
        BCA_res, BCA_ci = self.bsStat.BCA()
        nt.assert_almost_equal(self.known_primary, BCA_res, places=1)
        nptest.assert_array_almost_equal(knownBCA_ci, BCA_ci, decimal=1)

    @testing.seed
    def test_percentile(self):
        nt.assert_true(hasattr(self.bsStat, 'percentile'))
        knownPer_ci = np.array([8.70867143, 11.68634286])
        Per_res, Per_ci = self.bsStat.percentile()
        nt.assert_almost_equal(self.known_primary, Per_res, places=1)
        nptest.assert_array_almost_equal(knownPer_ci, Per_ci, decimal=1)

    def test_data(self):
        nt.assert_true(hasattr(self.bsStat, 'data'))
        nptest.assert_array_equal(np.array(self.data.res), self.bsStat.data)

    def test_statfxn(self):
        nt.assert_true(hasattr(self.bsStat, 'statfxn'))
        nt.assert_equal(self.statfxn, self.bsStat.statfxn)

    def test_alpha(self):
        nt.assert_true(hasattr(self.bsStat, 'alpha'))
        nt.assert_equal(self.alpha, self.bsStat.alpha)

    def test_NIter(self):
        nt.assert_true(hasattr(self.bsStat, 'NIter'))
        nt.assert_equal(self.NIter, self.bsStat.NIter)

    def test_final_result(self):
        nt.assert_true(hasattr(self.bsStat, 'final_result'))
        nt.assert_equal(self.known_primary, self.bsStat.final_result)
