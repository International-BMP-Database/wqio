import numpy
import numpy.testing as nptest
import pytest

from wqio import bootstrap
from wqio.tests import helpers


@pytest.fixture
def testdata():
    return helpers.getTestROSData()["res"].values


def test__acceleration(testdata):
    known_acceleration = -0.024051865664929263
    assert abs(bootstrap._acceleration(testdata) - known_acceleration) < 0.00001


def test__make_boot_index():
    result = bootstrap._make_boot_index(5, 5000)
    assert result.shape == (5000, 5)
    assert result.min() == 0
    assert result.max() == 4


@helpers.seed
@pytest.mark.parametrize(
    ("bootstrapper", "known_ci"),
    [
        (bootstrap.BCA, numpy.array([8.686, 11.661])),
        (bootstrap.percentile, numpy.array([8.670, 11.647])),
    ],
)
def test_bootstrappers(testdata, bootstrapper, known_ci):
    ci = bootstrapper(testdata, numpy.mean, 10000, 0.10)
    nptest.assert_array_almost_equal(known_ci, ci, decimal=3)


@helpers.seed
@pytest.mark.parametrize("uselog", [True, False])
def test_fit(uselog):
    N = 10
    x = numpy.arange(1, N + 1, dtype=float)
    y = numpy.array([4.527, 3.519, 9.653, 8.036, 10.805, 14.329, 13.508, 11.822, 13.281, 10.410])
    bsfit = bootstrap.fit(
        x, y, numpy.polyfit, niter=2000, xlog=uselog, ylog=uselog, deg=1, full=False
    )

    nptest.assert_array_almost_equal(bsfit.xhat, x[numpy.argsort(x)])

    expected = {
        False: {
            "yhat": numpy.array(
                [
                    5.841,
                    6.763,
                    7.685,
                    8.606,
                    9.528,
                    10.450,
                    11.371,
                    12.293,
                    13.215,
                    14.136,
                ]
            ),
            "lower": numpy.array(
                [
                    3.449,
                    4.855,
                    6.166,
                    7.181,
                    8.088,
                    8.954,
                    9.816,
                    10.561,
                    11.119,
                    11.591,
                ]
            ),
            "upper": numpy.array(
                [
                    9.948,
                    10.233,
                    10.562,
                    10.913,
                    11.542,
                    12.527,
                    13.901,
                    15.545,
                    17.218,
                    18.940,
                ]
            ),
        },
        True: {
            "yhat": numpy.array(
                [
                    4.004,
                    5.858,
                    7.318,
                    8.570,
                    9.686,
                    10.706,
                    11.651,
                    12.537,
                    13.374,
                    14.170,
                ]
            ),
            "lower": numpy.array(
                [
                    2.172,
                    3.961,
                    5.636,
                    7.079,
                    8.255,
                    9.218,
                    10.060,
                    10.760,
                    11.280,
                    11.660,
                ]
            ),
            "upper": numpy.array(
                [
                    8.048,
                    9.080,
                    9.882,
                    10.589,
                    11.429,
                    12.494,
                    13.871,
                    15.438,
                    17.124,
                    18.713,
                ]
            ),
        },
    }
    nptest.assert_array_almost_equal(bsfit.yhat, expected[uselog]["yhat"], decimal=3)
    nptest.assert_array_almost_equal(bsfit.lower, expected[uselog]["lower"], decimal=3)
    nptest.assert_array_almost_equal(bsfit.upper, expected[uselog]["upper"], decimal=3)
    assert bsfit.xlog == uselog
    assert bsfit.ylog == uselog
