import numpy
import numpy.testing as nptest
import pytest

from wqio import theil
from wqio.tests import helpers
from wqio.utils import TheilStats


@pytest.fixture
def xy():
    N = 12
    x = numpy.arange(1, 1 + N)
    y = numpy.array(
        [
            8.01975462,
            5.04563433,
            7.29575852,
            7.81281615,
            10.28277921,
            10.96935877,
            10.40494743,
            11.37816252,
            10.86221425,
            10.51920353,
            12.90208785,
            14.15634897,
        ]
    )
    return x, y


@pytest.fixture(params=[(True, True), (True, False), (False, False), (False, True)])
def ts(request, xy):
    return theil.TheilSenFit(xy[0], xy[1], request.param[0], request.param[1])


def test_TheilSenFit_attr(ts):
    assert hasattr(ts, "infl")
    assert hasattr(ts, "effl")
    assert hasattr(ts, "log_infl")
    assert hasattr(ts, "log_effl")


@helpers.seed
def test_TheilSenFit_stats(ts):
    expected = {
        (True, True): TheilStats(
            slope=0.38133424889520207,
            intercept=1.6350912326575155,
            low_slope=0.16821253087446286,
            high_slope=0.550735777966933,
        ),
        (True, False): TheilStats(
            slope=3.7630661898013327,
            intercept=3.4295263953951016,
            low_slope=1.615052807537439,
            high_slope=5.084786656060021,
        ),
        (False, False): TheilStats(
            slope=0.6021444541666665,
            intercept=6.548136527916667,
            low_slope=0.38654581599999993,
            high_slope=0.8729392799999999,
        ),
        (False, True): TheilStats(
            slope=0.06239014306941063,
            intercept=1.9422060209595988,
            low_slope=0.03781976207544716,
            high_slope=0.09183488925940173,
        ),
    }
    assert abs(ts.theil_stats.slope - expected[(ts.log_infl, ts.log_effl)].slope) < 0.0001
    assert abs(ts.theil_stats.intercept - expected[(ts.log_infl, ts.log_effl)].intercept) < 0.0001
    assert abs(ts.theil_stats.low_slope - expected[(ts.log_infl, ts.log_effl)].low_slope) < 0.0001
    assert abs(ts.theil_stats.high_slope - expected[(ts.log_infl, ts.log_effl)].high_slope) < 0.0001
    assert abs(ts.med_slope - expected[(ts.log_infl, ts.log_effl)][0]) < 0.0001
    assert abs(ts.intercept - expected[(ts.log_infl, ts.log_effl)][1]) < 0.0001
    assert abs(ts.low_slope - expected[(ts.log_infl, ts.log_effl)][2]) < 0.0001
    assert abs(ts.high_slope - expected[(ts.log_infl, ts.log_effl)][3]) < 0.0001


def test_TheilSenFit_x_fit(ts):
    expected = {
        True: numpy.array(
            [
                1.00000000,
                1.05557428,
                1.11423707,
                1.17616000,
                1.24152424,
                1.31052107,
                1.38335233,
                1.46023115,
                1.54138245,
                1.62704368,
                1.71746546,
                1.81291237,
                1.91366368,
                2.02001417,
                2.13227501,
                2.25077467,
                2.37585986,
                2.50789657,
                2.64727112,
                2.79439132,
                2.94968761,
                3.11361439,
                3.28665128,
                3.46930457,
                3.66210869,
                3.86562775,
                4.08045725,
                4.30722573,
                4.54659672,
                4.79927058,
                5.06598660,
                5.34752518,
                5.64471006,
                5.95841078,
                6.28954519,
                6.63908215,
                7.00804439,
                7.39751144,
                7.80862284,
                8.24258146,
                8.70065702,
                9.18418980,
                9.69459457,
                10.23336472,
                10.80207663,
                11.40239430,
                12.03607420,
                12.70497040,
                13.41104003,
                14.15634897,
            ]
        ),
        False: numpy.array(
            [
                1.00000000,
                1.26849692,
                1.53699384,
                1.80549075,
                2.07398767,
                2.34248459,
                2.61098151,
                2.87947842,
                3.14797534,
                3.41647226,
                3.68496918,
                3.95346610,
                4.22196301,
                4.49045993,
                4.75895685,
                5.02745377,
                5.29595068,
                5.56444760,
                5.83294452,
                6.10144144,
                6.36993836,
                6.63843527,
                6.90693219,
                7.17542911,
                7.44392603,
                7.71242294,
                7.98091986,
                8.24941678,
                8.51791370,
                8.78641061,
                9.05490753,
                9.32340445,
                9.59190137,
                9.86039829,
                10.12889520,
                10.39739212,
                10.66588904,
                10.93438596,
                11.20288287,
                11.47137979,
                11.73987671,
                12.00837363,
                12.27687055,
                12.54536746,
                12.81386438,
                13.08236130,
                13.35085822,
                13.61935513,
                13.88785205,
                14.15634897,
            ]
        ),
    }
    nptest.assert_array_almost_equal(ts.x_fit, expected[ts.log_infl], decimal=5)


def test_TheilSenFit_errors(ts):
    expected = {
        (True, True): numpy.array(
            [
                0.44681659,
                -0.28088861,
                -0.06673657,
                -0.10796727,
                0.08164554,
                0.07675534,
                -0.03485201,
                0.00364244,
                -0.08767803,
                -0.15994330,
                0.00789832,
                0.06749197,
            ]
        ),
        (True, False): numpy.array(
            [
                4.59022822,
                -0.99225079,
                -0.26791863,
                -0.83342768,
                0.79683142,
                0.79732290,
                -0.34716766,
                0.12355997,
                -0.83561366,
                -1.57510298,
                0.44912283,
                1.37595438,
            ]
        ),
        (False, False): numpy.array(
            [
                0.86947364,
                -2.70679111,
                -1.05881137,
                -1.14389819,
                0.72392041,
                0.80835552,
                -0.35820028,
                0.01287036,
                -1.10522237,
                -2.05037754,
                -0.26963767,
                0.38247899,
            ]
        ),
        (False, True): numpy.array(
            [
                0.07731166,
                -0.44846293,
                -0.14208330,
                -0.13600111,
                0.07631384,
                0.07855894,
                -0.03665561,
                -0.00963122,
                -0.11842712,
                -0.21290496,
                -0.07110845,
                -0.04072452,
            ]
        ),
    }
    nptest.assert_array_almost_equal(ts.errors, expected[(ts.log_infl, ts.log_effl)], decimal=5)


def test_TheilSenFit_MAD(ts):
    expected = {
        (True, True): 1.0824244964865588,
        (True, False): 0.8153752902670126,
        (False, False): 0.8389145774999998,
        (False, True): 1.081052923343382,
    }
    nptest.assert_almost_equal(ts.MAD, expected[(ts.log_infl, ts.log_effl)], decimal=5)


def test_TheilSenFit_BCF(ts):
    expected = {
        (True, True): 0.9607687777606599,
        (True, False): -0.11897181057076929,
        (False, False): -0.4913199674999998,
        (False, True): 0.9567477081643322,
    }
    nptest.assert_almost_equal(ts.BCF, expected[(ts.log_infl, ts.log_effl)], decimal=5)


def test_all_theil(xy):
    all_tsf, best_tsf = theil.all_theil(*xy)
    assert len(all_tsf) == 3
    assert all([isinstance(x, theil.TheilSenFit) for x in all_tsf])
    assert isinstance(best_tsf, theil.TheilSenFit)
    assert (best_tsf.log_infl, best_tsf.log_effl) == (True, False)
