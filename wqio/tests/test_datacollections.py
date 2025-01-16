from io import StringIO
from textwrap import dedent
from unittest import mock

import numpy
import pandas
import pandas.testing as pdtest
import pytest
import scipy
from packaging.version import Version
from scipy import stats

from wqio.datacollections import DataCollection, _dist_compare
from wqio.features import Dataset, Location
from wqio.tests import helpers

OLD_SCIPY = Version(scipy.version.version) < Version("0.19")


def check_stat(expected_csv, result, comp=False):
    index_col = [0]
    if comp:
        index_col += [1]

    file_obj = StringIO(dedent(expected_csv))
    expected = pandas.read_csv(file_obj, header=[0, 1], index_col=index_col)

    if comp:
        expected = expected.stack(level=-1)

    pdtest.assert_frame_equal(
        expected.sort_index(axis="columns"),
        result.sort_index(axis="columns").round(6),
        atol=1e-5,
    )


def remove_g_and_h(group):
    return group.name[1] not in ["G", "H"]


@pytest.fixture
def dc():
    df = helpers.make_dc_data_complex()
    dc = DataCollection(
        df,
        rescol="res",
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval="<",
        othergroups=None,
        pairgroups=["state", "bmp"],
        useros=True,
        filterfxn=remove_g_and_h,
        bsiter=10000,
    )

    return dc


@pytest.fixture
def dc_small():
    df = helpers.make_dc_data_complex()
    dc = DataCollection(
        df,
        rescol="res",
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval="<",
        othergroups=None,
        pairgroups=["state", "bmp"],
        useros=True,
        filterfxn=lambda g: g.name[1] in ["A", "B", "C"],
        bsiter=10000,
    )

    return dc


@pytest.fixture
def dc_noNDs():
    df = helpers.make_dc_data_complex()
    dc = DataCollection(
        df,
        rescol="res",
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval="junk",
        othergroups=None,
        pairgroups=["state", "bmp"],
        useros=True,
        filterfxn=remove_g_and_h,
        bsiter=10000,
    )

    return dc


def test_basic_attr(dc):
    assert dc._raw_rescol == "res"
    assert isinstance(dc.data, pandas.DataFrame)
    assert dc.roscol == "ros_res"
    assert dc.rescol == "ros_res"
    assert dc.qualcol == "qual"
    assert dc.stationcol == "loc"
    assert dc.paramcol == "param"
    assert dc.ndval == ["<"]
    assert dc.bsiter == 10000
    assert dc.groupcols == ["loc", "param"]
    assert dc.tidy_columns == ["loc", "param", "res", "__censorship"]
    assert hasattr(dc, "filterfxn")


def test_data(dc):
    assert isinstance(dc.data, pandas.DataFrame)
    assert dc.data.shape == (519, 8)
    assert "G" in dc.data["param"].unique()
    assert "H" in dc.data["param"].unique()


@pytest.mark.parametrize("useros", [True, False])
def test_tidy(dc, useros):
    assert isinstance(dc.tidy, pandas.DataFrame)
    assert dc.tidy.shape == (388, 5)
    assert "G" not in dc.tidy["param"].unique()
    assert "H" not in dc.tidy["param"].unique()
    collist = ["loc", "param", "res", "__censorship", "ros_res"]
    assert dc.tidy.columns.tolist() == collist


def test_paired(dc):
    assert isinstance(dc.paired, pandas.DataFrame)
    assert dc.paired.shape == (164, 6)
    assert "G" not in dc.paired.index.get_level_values("param").unique()
    assert "H" not in dc.paired.index.get_level_values("param").unique()
    dc.paired.columns.tolist() == [
        ("res", "Inflow"),
        ("res", "Outflow"),
        ("res", "Reference"),
        ("__censorship", "Inflow"),
        ("__censorship", "Outflow"),
        ("__censorship", "Reference"),
    ]


def test_count(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,Count,Count,Count
        param,,,
        A,21,22,20
        B,24,22,19
        C,24,24,25
        D,24,25,21
        E,19,16,20
        F,21,24,17
    """
    check_stat(known_csv, dc.count)


def test_n_unique(dc):
    known_csv = """\
        loc,Inflow,Outflow,Reference
        result,bmp,bmp,bmp
        param,,,
        A,7,7,7
        B,7,7,7
        C,7,7,7
        D,7,7,7
        E,7,7,7
        F,7,7,7
        G,7,7,7
        H,7,7,7
    """
    check_stat(known_csv, dc.n_unique("bmp"))


@helpers.seed
def test_median(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        result,lower,median,upper,lower,median,upper,lower,median,upper
        param,,,,,,,,,
        A,0.334506,1.197251,2.013994,0.860493,2.231058,2.626023,1.073386,1.639472,1.717293
        B,1.366948,2.773989,3.297147,0.23201,1.546499,2.579206,0.204164,1.565076,2.196367
        C,0.17351,0.525957,0.68024,0.247769,0.396984,0.540742,0.136462,0.412693,0.559458
        D,0.374122,1.201892,2.098846,0.516989,1.362759,1.827087,0.314655,0.882695,1.24545
        E,0.276095,1.070858,1.152887,0.287914,0.516746,1.456859,0.366824,0.80716,2.040739
        F,0.05667,0.832488,1.310575,0.425237,1.510942,2.193997,0.162327,0.745993,1.992513
    """
    check_stat(known_csv, dc.median)


@helpers.seed
def test_mean(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        result,lower,mean,upper,lower,mean,upper,lower,mean,upper
        param,,,,,,,,,
        A,1.231607,2.646682,4.204054,1.930601,5.249281,9.081952,1.540167,3.777974,6.389439
        B,2.99031,7.647175,12.810844,1.545539,6.863835,12.705913,1.010374,4.504255,9.592572
        C,0.37496,0.513248,0.65948,0.411501,1.004637,1.706317,0.35779,0.541962,0.734751
        D,1.29141,3.021235,4.987855,1.285899,2.318808,3.451824,1.008364,1.945828,2.924812
        E,0.818641,1.914696,3.049554,0.584826,1.098241,1.640807,1.113589,2.283292,3.581946
        F,0.8379,9.825404,25.289933,1.497825,3.450184,5.61929,0.939917,2.491708,4.094258
    """
    check_stat(known_csv, dc.mean)


@helpers.seed
def test_std_dev(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,std. dev.,std. dev.,std. dev.
        param,,,
        A,3.675059,8.92456,5.671232
        B,12.625938,13.922531,11.054115
        C,0.361364,1.727582,0.503498
        D,4.915433,2.90815,2.303697
        E,2.620266,1.132665,2.861698
        F,35.298251,5.476337,3.502956
    """
    check_stat(known_csv, dc.std_dev)


@helpers.seed
def test_percentile_25(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,pctl 25,pctl 25,pctl 25
        param,,,
        A,0.522601,0.906029,1.094721
        B,1.472541,0.251126,0.314226
        C,0.164015,0.267521,0.136462
        D,0.35688,0.516989,0.383895
        E,0.364748,0.311508,0.394658
        F,0.120068,0.406132,0.224429
    """
    check_stat(known_csv, dc.percentile(25))


@helpers.seed
def test_percentile_75(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,pctl 75,pctl 75,pctl 75
        param,,,
        A,2.563541,3.838021,2.650648
        B,4.728871,2.849948,2.261847
        C,0.776388,0.853535,0.792612
        D,3.04268,2.79341,3.611793
        E,1.532775,1.59183,3.201534
        F,1.792985,2.80979,2.742249
    """
    check_stat(known_csv, dc.percentile(75))


@helpers.seed
def test_logmean(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        result,Log-mean,lower,upper,Log-mean,lower,upper,Log-mean,lower,upper
        param,,,,,,,,,
        A,0.140559,-0.55112,0.644202,0.733004,0.047053,1.22099,0.545205,-0.057683,1.029948
        B,1.026473,0.368659,1.541241,0.105106,-0.939789,0.860244,0.068638,-0.932357,0.661203
        C,-0.963004,-1.304115,-0.638446,-0.83221,-1.464092,-0.414379,-1.088377,-1.556795,-0.720706
        D,0.062317,-0.663241,0.58349,0.185757,-0.325074,0.598432,-0.063507,-0.670456,0.434214
        E,-0.103655,-0.751075,0.385909,-0.456202,-1.08692,0.029967,-0.068135,-0.787007,0.51226
        F,-0.442721,-1.874677,0.344704,0.211658,-0.504166,0.734283,-0.253352,-1.175917,0.467231
    """
    check_stat(known_csv, dc.logmean)


@helpers.seed
def test_logstd_dev(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,Log-std. dev.,Log-std. dev.,Log-std. dev.
        param,,,
        A,1.407957,1.375282,1.257185
        B,1.461146,2.125324,1.707543
        C,0.836108,1.290809,1.078977
        D,1.563796,1.211728,1.309485
        E,1.29905,1.157803,1.512734
        F,2.381456,1.548944,1.753965
    """
    check_stat(known_csv, dc.logstd_dev)


@helpers.seed
def test_geomean(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        Geo-mean,Log-mean,lower,upper,Log-mean,lower,upper,Log-mean,lower,upper
        param,,,,,,,,,
        A,1.150917,0.576304,1.904467,2.081323,1.048178,3.390543,1.724962,0.943949,2.800919
        B,2.791205,1.445795,4.670381,1.110829,0.39071,2.363737,1.071049,0.393625,1.937121
        C,0.381744,0.271413,0.528113,0.435087,0.231288,0.66075,0.336763,0.210811,0.486409
        D,1.064299,0.515179,1.792283,1.204129,0.722474,1.819264,0.938467,0.511475,1.543749
        E,0.901536,0.471859,1.470951,0.633686,0.337254,1.03042,0.934134,0.455205,1.66906
        F,0.642286,0.153405,1.411572,1.235726,0.604009,2.083988,0.776195,0.308536,1.595571
    """
    check_stat(known_csv, dc.geomean)


@helpers.seed
def test_geostd_dev(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        Geo-std. dev.,Log-std. dev.,Log-std. dev.,Log-std. dev.
        param,,,
        A,4.087597,3.956193,3.515511
        B,4.310897,8.375613,5.515395
        C,2.30737,3.635725,2.941668
        D,4.776921,3.359284,3.704266
        E,3.665813,3.182932,4.539124
        F,10.820642,4.706498,5.777465
    """
    check_stat(known_csv, dc.geostd_dev)


@helpers.seed
def test_shapiro(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,pvalue,statistic,pvalue,statistic,pvalue,statistic
        param,,,,,,
        A,1.8e-05,0.685783,1e-06,0.576069,4e-06,0.61735
        B,1e-06,0.594411,0.0,0.530962,0.0,0.41471
        C,0.028774,0.905906,0.0,0.546626,0.00279,0.860373
        D,1e-06,0.622915,1.5e-05,0.722374,0.000202,0.76518
        E,1.7e-05,0.654137,0.004896,0.818813,0.000165,0.74917
        F,0.0,0.292916,2e-06,0.634671,0.000167,0.713968
    """
    check_stat(known_csv, dc.shapiro())


@helpers.seed
def test_shapiro_log(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,statistic,pvalue,statistic,pvalue,statistic,pvalue
        param,,,,,,
        A,0.983521938,0.96662426,0.979861856,0.913820148,0.939460814,0.234214202
        B,0.957531095,0.390856266,0.97048676,0.722278714,0.967978418,0.735424638
        C,0.906479359,0.029602444,0.974698305,0.78197974,0.967106879,0.572929323
        D,0.989704251,0.995502174,0.990663111,0.997093379,0.964812279,0.617747009
        E,0.955088913,0.479993254,0.95211035,0.523841977,0.963425279,0.61430341
        F,0.97542423,0.847370088,0.982230783,0.933124721,0.966197193,0.749036908
    """
    check_stat(known_csv, dc.shapiro_log())


@helpers.seed
def test_lilliefors(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,lilliefors,pvalue,lilliefors,pvalue,lilliefors,pvalue
        param,,,,,,
        A,0.308131,0.001,0.340594,0.001,0.364453,0.001
        B,0.36764,0.001,0.420343,0.001,0.417165,0.001
        C,0.166799,0.082069,0.324733,0.001,0.161753,0.08817
        D,0.273012,0.001,0.240311,0.001,0.296919,0.001
        E,0.341398,0.001,0.239314,0.016156,0.233773,0.005575
        F,0.419545,0.001,0.331315,0.001,0.284249,0.001
    """
    check_stat(known_csv, dc.lilliefors())


@helpers.seed
def test_lilliefors_log(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,log-lilliefors,pvalue,log-lilliefors,pvalue,log-lilliefors,pvalue
        param,,,,,,
        A,0.085481,0.95458,0.154439,0.197157,0.201414,0.033257
        B,0.161628,0.10505,0.124479,0.496979,0.159343,0.229694
        C,0.169573,0.071882,0.123882,0.443797,0.117466,0.489157
        D,0.068855,0.99,0.060674,0.99,0.13402,0.419675
        E,0.135066,0.471868,0.145523,0.477979,0.091649,0.928608
        F,0.144208,0.306945,0.084633,0.927419,0.085869,0.980029
    """
    check_stat(known_csv, dc.lilliefors_log())


@helpers.seed
def test_anderson_darling(dc):
    with helpers.raises(NotImplementedError):
        _ = dc.anderson_darling()


@helpers.seed
def test_anderson_darling_log(dc):
    with helpers.raises(NotImplementedError):
        _ = dc.anderson_darling_log()


@helpers.seed
def test_mann_whitney(dc):
    known_csv = """\
        ,,mann_whitney,mann_whitney,mann_whitney,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,180.0,179.0,,0.2198330905,0.4263216587
        A,Outflow,282.0,,248.0,0.2198330905,,0.488580368
        A,Reference,241.0,192.0,,0.4263216587,0.488580368,
        B,Inflow,,345.0,317.0,,0.0766949991,0.0304383994
        B,Outflow,183.0,,216.0,0.0766949991,,0.8650586835
        B,Reference,139.0,202.0,,0.0304383994,0.8650586835,
        C,Inflow,,282.0,323.0,,0.9097070273,0.6527104406
        C,Outflow,294.0,,323.0,0.9097070273,,0.6527104406
        C,Reference,277.0,277.0,,0.6527104406,0.6527104406,
        D,Inflow,,285.0,263.0,,0.7718162376,0.8111960975
        D,Outflow,315.0,,293.0,0.7718162376,,0.5082395211
        D,Reference,241.0,232.0,,0.8111960975,0.5082395211,
        E,Inflow,,164.0,188.0,,0.7033493939,0.9663820218
        E,Outflow,140.0,,132.0,0.7033493939,,0.3813114322
        E,Reference,192.0,188.0,,0.9663820218,0.3813114322,
        F,Inflow,,201.0,172.0,,0.2505911218,0.8601783903
        F,Outflow,303.0,,236.0,0.2505911218,,0.4045186043
        F,Reference,185.0,172.0,,0.8601783903,0.4045186043
    """
    check_stat(known_csv, dc.mann_whitney(), comp=True)


@helpers.seed
def test_t_test(dc):
    known_csv = """\
    ,,t_test,t_test,t_test,pvalue,pvalue,pvalue
    loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
    param,loc_1,,,,,,
    A,Inflow,,-1.239311,-0.761726,,0.222278,0.450806
    A,Outflow,1.239311,,0.630254,0.222278,,0.532113
    A,Reference,0.761726,-0.630254,,0.450806,0.532113,
    B,Inflow,,0.200136,0.855661,,0.842296,0.397158
    B,Outflow,-0.200136,,0.594193,0.842296,,0.555814
    B,Reference,-0.855661,-0.594193,,0.397158,0.555814,
    C,Inflow,,-1.363936,-0.228508,,0.179225,0.820243
    C,Outflow,1.363936,,1.283982,0.179225,,0.205442
    C,Reference,0.228508,-1.283982,,0.820243,0.205442,
    D,Inflow,,0.61178,0.917349,,0.543631,0.364076
    D,Outflow,-0.61178,,0.475391,0.543631,,0.63686
    D,Reference,-0.917349,-0.475391,,0.364076,0.63686,
    E,Inflow,,1.156605,-0.418858,,0.255738,0.677741
    E,Outflow,-1.156605,,-1.558039,0.255738,,0.128485
    E,Reference,0.418858,1.558039,,0.677741,0.128485,
    F,Inflow,,0.874261,0.851029,,0.386833,0.400379
    F,Outflow,-0.874261,,0.63432,0.386833,,0.529575
    F,Reference,-0.851029,-0.63432,,0.400379,0.529575,
    """
    check_stat(known_csv, dc.t_test(), comp=True)


@helpers.seed
def test_levene(dc):
    known_csv = """\
        ,,levene,levene,levene,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,1.176282059,0.293152155,,0.284450688,0.591287419
        A,Outflow,1.176282059,,0.397705309,0.284450688,,0.531863542
        A,Reference,0.293152155,0.397705309,,0.591287419,0.531863542,
        B,Inflow,,0.003559637,0.402002411,,0.952694449,0.529578712
        B,Outflow,0.003559637,,0.408938588,0.952694449,,0.526247443
        B,Reference,0.402002411,0.408938588,,0.529578712,0.526247443,
        C,Inflow,,1.965613561,0.679535532,,0.167626459,0.413910674
        C,Outflow,1.965613561,,1.462364363,0.167626459,,0.232602352
        C,Reference,0.679535532,1.462364363,,0.413910674,0.232602352,
        D,Inflow,,0.643364813,0.983777911,,0.426532092,0.32681669
        D,Outflow,0.643364813,,0.116830634,0.426532092,,0.734124856
        D,Reference,0.983777911,0.116830634,,0.32681669,0.734124856,
        E,Inflow,,0.961616536,0.410491665,,0.333914902,0.525668596
        E,Outflow,0.961616536,,2.726351564,0.333914902,,0.107912818
        E,Reference,0.410491665,2.726351564,,0.525668596,0.107912818,
        F,Inflow,,0.841984453,0.734809611,,0.363948105,0.396999375
        F,Outflow,0.841984453,,0.25881357,0.363948105,,0.613802541
        F,Reference,0.734809611,0.25881357,,0.396999375,0.613802541,
    """
    check_stat(known_csv, dc.levene(), comp=True)


@helpers.seed
def test_wilcoxon(dc):
    known_csv = """\
        ,,wilcoxon,wilcoxon,wilcoxon,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,32.0,59.0,,0.03479,0.430679
        A,Outflow,32.0,,46.0,0.03479,,0.274445
        A,Reference,59.0,46.0,,0.430679,0.274445,
        B,Inflow,,38.0,22.0,,0.600179,0.182338
        B,Outflow,38.0,,31.0,0.600179,,0.858863
        B,Reference,22.0,31.0,,0.182338,0.858863,
        C,Inflow,,75.0,120.0,,0.167807,0.601046
        C,Outflow,75.0,,113.0,0.167807,,0.463381
        C,Reference,120.0,113.0,,0.601046,0.463381,
        D,Inflow,,44.0,31.0,,0.593618,0.530285
        D,Outflow,44.0,,45.0,0.593618,,0.972125
        D,Reference,31.0,45.0,,0.530285,0.972125,
        E,Inflow,,21.0,19.0,,0.910156,0.431641
        E,Outflow,21.0,,16.0,0.910156,,0.077148
        E,Reference,19.0,16.0,,0.431641,0.077148,
        F,Inflow,,62.0,22.0,,0.492459,1.0
        F,Outflow,62.0,,28.0,0.492459,,0.656642
        F,Reference,22.0,28.0,,1.0,0.656642,
    """
    check_stat(known_csv, dc.wilcoxon(), comp=True)


@helpers.seed
def test_ranksums(dc):
    known_csv = """\
        ,,pvalue,pvalue,pvalue,rank_sums,rank_sums,rank_sums
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,0.2153009,0.4187782,,-1.2391203,-0.8085428
        A,Outflow,0.2153009,,0.4807102,1.2391203,,0.7051607
        A,Reference,0.4187782,0.4807102,,0.8085428,-0.7051607,
        B,Inflow,,0.0748817,0.029513,,1.781188,2.1765661
        B,Outflow,0.0748817,,0.8547898,-1.781188,,0.1830104
        B,Reference,0.029513,0.8547898,,-2.1765661,-0.1830104,
        C,Inflow,,0.9015386,0.6455162,,-0.1237179,0.46
        C,Outflow,0.9015386,,0.6455162,0.1237179,,0.46
        C,Reference,0.6455162,0.6455162,,-0.46,-0.46,
        D,Inflow,,0.7641772,0.8023873,,-0.3,0.2502587
        D,Outflow,0.7641772,,0.5011969,0.3,,0.6726078
        D,Reference,0.8023873,0.5011969,,-0.2502587,-0.6726078,
        E,Inflow,,0.6911022,0.9551863,,0.3973597,-0.0561951
        E,Outflow,0.6911022,,0.3727144,-0.3973597,,-0.8914004
        E,Reference,0.9551863,0.3727144,,0.0561951,0.8914004,
        F,Inflow,,0.2459307,0.8486619,,-1.1602902,-0.190826
        F,Outflow,0.2459307,,0.3971011,1.1602902,,0.8468098
        F,Reference,0.8486619,0.3971011,,0.190826,-0.8468098,
    """
    check_stat(known_csv, dc.ranksums(), comp=True)


@helpers.seed
@pytest.mark.xfail(OLD_SCIPY, reason="Scipy < 0.19")
def test_kendall(dc):
    known_csv = """\
        ,,kendalltau,kendalltau,kendalltau,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,-0.051661,-0.00738,,0.772893,0.967114
        A,Outflow,-0.051661,,-0.083333,0.772893,,0.690095
        A,Reference,-0.00738,-0.083333,,0.967114,0.690095,
        B,Inflow,,0.441351,0.298246,,0.015267,0.119265
        B,Outflow,0.441351,,0.559855,0.015267,,0.004202
        B,Reference,0.298246,0.559855,,0.119265,0.004202,
        C,Inflow,,0.280223,0.084006,,0.078682,0.578003
        C,Outflow,0.280223,,-0.1417,0.078682,,0.352394
        C,Reference,0.084006,-0.1417,,0.578003,0.352394,
        D,Inflow,,0.403469,0.095299,,0.020143,0.634826
        D,Outflow,0.403469,,0.318337,0.020143,,0.094723
        D,Reference,0.095299,0.318337,,0.634826,0.094723,
        E,Inflow,,0.114286,0.640703,,0.673337,0.004476
        E,Outflow,0.114286,,0.167944,0.673337,,0.449603
        E,Reference,0.640703,0.167944,,0.004476,0.449603,
        F,Inflow,,0.0,0.07231,,1.0,0.763851
        F,Outflow,0.0,,0.388889,1.0,,0.063
        F,Reference,0.07231,0.388889,,0.763851,0.063,
    """
    check_stat(known_csv, dc.kendall(), comp=True)


@helpers.seed
def test_spearman(dc):
    known_csv = """\
        ,,pvalue,pvalue,pvalue,spearmanrho,spearmanrho,spearmanrho
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,0.7574884491,0.9627447553,,-0.0809319588,0.012262418
        A,Outflow,0.7574884491,,0.7617330788,-0.0809319588,,-0.0823529412
        A,Reference,0.9627447553,0.7617330788,,0.012262418,-0.0823529412,
        B,Inflow,,0.0110829791,0.0775159774,,0.5831305575,0.4537313433
        B,Outflow,0.0110829791,,0.0024069317,0.5831305575,,0.6850916941
        B,Reference,0.0775159774,0.0024069317,,0.4537313433,0.6850916941,
        C,Inflow,,0.1330504059,0.6063501968,,0.3387640122,0.1134228342
        C,Outflow,0.1330504059,,0.3431640379,0.3387640122,,-0.2070506455
        C,Reference,0.6063501968,0.3431640379,,0.1134228342,-0.2070506455,
        D,Inflow,,0.0195715066,0.4751861062,,0.4935814032,0.1858231711
        D,Outflow,0.0195715066,,0.1263974782,0.4935814032,,0.363209462
        D,Reference,0.4751861062,0.1263974782,,0.1858231711,0.363209462,
        E,Inflow,,0.9828818202,0.0013596162,,0.0084033613,0.8112988341
        E,Outflow,0.9828818202,,0.3413722947,0.0084033613,,0.3012263814
        E,Reference,0.0013596162,0.3413722947,,0.8112988341,0.3012263814,
        F,Inflow,,0.9645303744,0.6759971848,,-0.0106277141,0.1348767061
        F,Outflow,0.9645303744,,0.0560590794,-0.0106277141,,0.5028571429
        F,Reference,0.6759971848,0.0560590794,,0.1348767061,0.5028571429
    """
    check_stat(known_csv, dc.spearman(), comp=True)


@helpers.seed
def test_kruskal_wallis(dc):
    result = dc.kruskal_wallis()
    _expected = [
        {"K-W H": 1.798578, "pvalue": 0.406859, "param": "A"},
        {"K-W H": 5.421338, "pvalue": 0.066492, "param": "B"},
        {"K-W H": 0.29261, "pvalue": 0.863894, "param": "C"},
        {"K-W H": 0.39957, "pvalue": 0.818907, "param": "D"},
        {"K-W H": 0.585441, "pvalue": 0.746231, "param": "E"},
        {"K-W H": 1.483048, "pvalue": 0.476387, "param": "F"},
    ]
    expected = pandas.DataFrame(_expected).set_index("param")
    pandas.testing.assert_frame_equal(result, expected)


@helpers.seed
def test_f_test(dc):
    result = dc.f_test()
    _expected = [
        {"f-test": 0.861339, "pvalue": 0.427754, "param": "A"},
        {"f-test": 0.343445, "pvalue": 0.710663, "param": "B"},
        {"f-test": 1.653142, "pvalue": 0.198833, "param": "C"},
        {"f-test": 0.526782, "pvalue": 0.592927, "param": "D"},
        {"f-test": 1.114655, "pvalue": 0.335738, "param": "E"},
        {"f-test": 0.738629, "pvalue": 0.482134, "param": "F"},
    ]
    expected = pandas.DataFrame(_expected).set_index("param")
    pandas.testing.assert_frame_equal(result, expected)


def test_tukey_hsd_smoke_test(dc):
    hsd, scores = dc.tukey_hsd()
    assert isinstance(hsd, pandas.DataFrame)
    assert isinstance(scores, pandas.DataFrame)

    assert hsd.index.names == ["loc 1", "loc 2", "param"]
    assert hsd.columns.tolist() == [
        "HSD Stat",
        "p-value",
        "CI-Low",
        "CI-High",
        "is_diff",
        "sign_of_diff",
        "score",
    ]

    assert scores.index.names == ["param"]
    assert scores.columns.tolist() == ["Inflow", "Outflow", "Reference"]


def test_dunn(dc):
    dunnres = dc.dunn()
    assert dunnres.index.tolist() == list("ABCDEF")
    assert dunnres.columns.tolist() == ["Inflow", "Outflow", "Reference"]


@helpers.seed
def test_theilslopes(dc):
    with helpers.raises(NotImplementedError):
        _ = dc.theilslopes()


def test_inventory(dc):
    known_csv = StringIO(
        dedent(
            """\
    loc,param,Count,Non-Detect
    Inflow,A,21,3
    Inflow,B,24,6
    Inflow,C,24,0
    Inflow,D,24,11
    Inflow,E,19,4
    Inflow,F,21,8
    Outflow,A,22,1
    Outflow,B,22,9
    Outflow,C,24,4
    Outflow,D,25,12
    Outflow,E,16,2
    Outflow,F,24,8
    Reference,A,20,2
    Reference,B,19,6
    Reference,C,25,4
    Reference,D,21,12
    Reference,E,20,3
    Reference,F,17,7
    """
        )
    )
    expected = pandas.read_csv(known_csv, index_col=[0, 1]).astype(int)
    pdtest.assert_frame_equal(expected, dc.inventory.astype(int), check_names=False)


def test_inventory_noNDs(dc_noNDs):
    known_csv = StringIO(
        dedent(
            """\
    loc,param,Count,Non-Detect
    Inflow,A,21,0
    Inflow,B,24,0
    Inflow,C,24,0
    Inflow,D,24,0
    Inflow,E,19,0
    Inflow,F,21,0
    Outflow,A,22,0
    Outflow,B,22,0
    Outflow,C,24,0
    Outflow,D,25,0
    Outflow,E,16,0
    Outflow,F,24,0
    Reference,A,20,0
    Reference,B,19,0
    Reference,C,25,0
    Reference,D,21,0
    Reference,E,20,0
    Reference,F,17,0
    """
        )
    )
    expected = pandas.read_csv(known_csv, index_col=[0, 1]).astype(int)
    pdtest.assert_frame_equal(
        expected,
        dc_noNDs.inventory.astype(int),
        check_names=False,
    )


@helpers.seed
def test_stat_summary(dc):
    known_csv = StringIO(
        dedent(
            """\
    ros_res,loc,A,B,C,D,E,F
    Count,Inflow,21,24,24,24,19,21
    Count,Outflow,22,22,24,25,16,24
    Count,Reference,20,19,25,21,20,17
    Non-Detect,Inflow,3.0,6.0,0.0,11.0,4.0,8.0
    Non-Detect,Outflow,1.0,9.0,4.0,12.0,2.0,8.0
    Non-Detect,Reference,2.0,6.0,4.0,12.0,3.0,7.0
    mean,Inflow,2.64668,7.64717,0.51325,3.02124,1.9147,9.8254
    mean,Outflow,5.24928,6.86384,1.00464,2.31881,1.09824,3.45018
    mean,Reference,3.77797,4.50425,0.54196,1.94583,2.28329,2.49171
    std,Inflow,3.67506,12.62594,0.36136,4.91543,2.62027,35.29825
    std,Outflow,8.92456,13.92253,1.72758,2.90815,1.13267,5.47634
    std,Reference,5.67123,11.05411,0.5035,2.3037,2.8617,3.50296
    min,Inflow,0.0756,0.17404,0.10213,0.05365,0.08312,0.00803
    min,Outflow,0.11177,0.02106,0.03578,0.11678,0.07425,0.06377
    min,Reference,0.15575,0.04909,0.04046,0.08437,0.05237,0.03445
    10%,Inflow,0.1772,0.45233,0.13467,0.15495,0.1763,0.03548
    10%,Outflow,0.44852,0.08297,0.08222,0.26949,0.19903,0.18008
    10%,Reference,0.38448,0.13467,0.08241,0.19355,0.12777,0.09457
    25%,Inflow,0.5226,1.47254,0.16401,0.35688,0.36475,0.12007
    25%,Outflow,0.90603,0.25113,0.26752,0.51699,0.31151,0.40613
    25%,Reference,1.09472,0.31423,0.13646,0.3839,0.39466,0.22443
    50%,Inflow,1.19725,2.77399,0.52596,1.20189,1.07086,0.83249
    50%,Outflow,2.23106,1.5465,0.39698,1.36276,0.51675,1.51094
    50%,Reference,1.63947,1.56508,0.41269,0.8827,0.80716,0.74599
    75%,Inflow,2.56354,4.72887,0.77639,3.04268,1.53278,1.79299
    75%,Outflow,3.83802,2.84995,0.85354,2.79341,1.59183,2.80979
    75%,Reference,2.65065,2.26185,0.79261,3.61179,3.20153,2.74225
    90%,Inflow,6.02835,24.40655,0.99293,8.00691,6.28345,8.51706
    90%,Outflow,12.43052,23.90022,2.43829,5.66731,2.30348,10.32829
    90%,Reference,12.58278,6.67125,1.2205,4.78255,7.72012,8.57303
    max,Inflow,13.87664,45.97893,1.26657,21.75505,8.88365,163.01001
    max,Outflow,36.58941,47.49381,8.04948,12.39894,4.19118,23.29367
    max,Reference,21.22363,48.23615,1.94442,7.67751,8.75609,10.5095
    """
        )
    )

    expected = pandas.read_csv(known_csv, index_col=[0, 1]).T
    pdtest.assert_frame_equal(
        expected.round(5),
        dc.stat_summary().round(5),
        check_names=False,
        check_dtype=False,
        rtol=1e-4,
    )


def test_locations(dc):
    for loc in dc.locations:
        assert isinstance(loc, Location)
    assert len(dc.locations) == 18
    assert dc.locations[0].definition == {"loc": "Inflow", "param": "A"}
    assert dc.locations[1].definition == {"loc": "Inflow", "param": "B"}


def test_datasets(dc):
    _ds = []
    for d in dc.datasets("Inflow", "Outflow"):
        assert isinstance(d, Dataset)
        _ds.append(d)
    assert len(_ds) == 6
    assert _ds[0].definition == {"param": "A"}
    assert _ds[1].definition == {"param": "B"}


# this sufficiently tests dc._filter_collection
def test_selectLocations(dc):
    locs = dc.selectLocations(param="A", loc=["Inflow", "Outflow"])
    assert len(locs) == 2
    for n, (loc, loctype) in enumerate(zip(locs, ["Inflow", "Outflow"])):
        assert isinstance(loc, Location)
        assert loc.definition["param"] == "A"
        assert loc.definition["loc"] == loctype


def test_selectLocations_squeeze_False(dc):
    locs = dc.selectLocations(param="A", loc=["Inflow"], squeeze=False)
    assert len(locs) == 1
    for n, loc in enumerate(locs):
        assert isinstance(loc, Location)
        assert loc.definition["param"] == "A"
        assert loc.definition["loc"] == "Inflow"


def test_selectLocations_squeeze_True(dc):
    loc = dc.selectLocations(param="A", loc=["Inflow"], squeeze=True)
    assert isinstance(loc, Location)
    assert loc.definition["param"] == "A"
    assert loc.definition["loc"] == "Inflow"


def test_selectLocations_squeeze_True_None(dc):
    loc = dc.selectLocations(param="A", loc=["Junk"], squeeze=True)
    assert loc is None


# since the test_selectLocations* tests stress _filter_collection
# enough, we'll mock it out for datasets:
def test_selectDatasets(dc):
    with (
        mock.patch.object(dc, "_filter_collection") as _fc,
        mock.patch.object(dc, "datasets", return_value=["A", "B"]) as _ds,
    ):
        dc.selectDatasets("Inflow", "Reference", foo="A", bar="C")
        _ds.assert_called_once_with("Inflow", "Reference")
        _fc.assert_called_once_with(["A", "B"], foo="A", bar="C", squeeze=False)


@pytest.mark.parametrize("func", [stats.mannwhitneyu, stats.wilcoxon])
@pytest.mark.parametrize(("x", "all_same"), [([5, 5, 5, 5, 5], True), ([5, 6, 7, 7, 8], False)])
def test_dist_compare_wrapper(x, all_same, func):
    y = [5, 5, 5, 5, 5]
    with mock.patch.object(stats, func.__name__) as _test:
        result = _dist_compare(x, y, _test, alternative="two-sided")
        if all_same:
            assert numpy.isnan(result.stat)
            assert numpy.isnan(result.pvalue)
            assert _test.call_count == 0
        else:
            # assert result == (0, 0)
            _test.assert_called_once_with(x, y, alternative="two-sided")
