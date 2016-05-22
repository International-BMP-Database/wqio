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

