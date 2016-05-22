import pytest
import numpy.testing as nptest

import numpy

from wqio import testing
from wqio.algo import bootstrap


@pytest.fixture
def testdata():
    return testing.getTestROSData()['res'].values


def test__acceleration(testdata):
    known_acceleration = -0.024051865664929263
    assert abs(bootstrap._acceleration(testdata) - known_acceleration) < 0.00001


def test__make_boot_index():
    result = bootstrap._make_boot_index(5, 5000)
    assert result.shape == (5000, 5)
    assert result.min() == 0
    assert result.max() == 4


@testing.seed
@pytest.mark.parametrize(('bootstrapper', 'known_ci'), [
    (bootstrap.BCA, numpy.array([8.686, 11.661])),
    (bootstrap.percentile, numpy.array([8.670, 11.647]))
])
def test_bootstrappers(testdata, bootstrapper, known_ci):
    res, ci = bootstrapper(testdata, numpy.mean, 10000, 0.10)
    assert abs(res - numpy.mean(testdata)) < 0.001
    nptest.assert_array_almost_equal(known_ci, ci, decimal=3)


class Test_Stat:
    def setup(self):
        numpy.random.seed(0)
        self.data = testing.getTestROSData()
        self.statfxn = numpy.mean
        self.alpha = 0.10
        self.NIter = 10000
        self.bsStat = bootstrap.Stat(numpy.array(self.data.res), statfxn=self.statfxn,
                                     alpha=self.alpha, NIter=self.NIter)

        self.known_secondary = 10.121394742857142
        self.known_primary = 10.120571428571429


    @testing.seed
    def test_BCA(self):
        assert hasattr(self.bsStat, 'BCA')
        knownBCA_ci = numpy.array([8.74244187, 11.71177557])
        BCA_res, BCA_ci = self.bsStat.BCA()
        assert abs(self.known_primary - BCA_res) < 0.1
        nptest.assert_array_almost_equal(knownBCA_ci, BCA_ci, decimal=1)

    @testing.seed
    def test_percentile(self):
        assert hasattr(self.bsStat, 'percentile')
        knownPer_ci = numpy.array([8.70867143, 11.68634286])
        Per_res, Per_ci = self.bsStat.percentile()
        assert abs(self.known_primary - Per_res) < 0.1
        nptest.assert_array_almost_equal(knownPer_ci, Per_ci, decimal=1)

    def test_data(self):
        assert hasattr(self.bsStat, 'data')
        nptest.assert_array_equal(numpy.array(self.data.res), self.bsStat.data)

    def test_statfxn(self):
        assert hasattr(self.bsStat, 'statfxn')
        assert self.statfxn == self.bsStat.statfxn

    def test_alpha(self):
        assert hasattr(self.bsStat, 'alpha')
        assert self.alpha == self.bsStat.alpha

    def test_NIter(self):
        assert hasattr(self.bsStat, 'NIter')
        assert self.NIter == self.bsStat.NIter

    def test_final_result(self):
        assert hasattr(self.bsStat, 'final_result')
        assert self.known_primary == self.bsStat.final_result
