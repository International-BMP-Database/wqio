import numpy.testing as nptest
import pandas
import pytest
import scipy
from packaging.version import Version

from wqio.features import Dataset, Location
from wqio.tests import helpers

OLD_SCIPY = Version(scipy.version.version) < Version("0.19")
TOLERANCE = 0.05


@pytest.fixture(params=[True, False])
def location(request):
    data = helpers.getTestROSData()
    return Location(
        data,
        station_type="inflow",
        bsiter=1500,
        rescol="res",
        qualcol="qual",
        useros=request.param,
    )


@pytest.mark.parametrize(
    "attr",
    [
        "name",
        "station_type",
        "analysis_space",
        "rescol",
        "qualcol",
        "plot_marker",
        "scatter_marker",
        "hasData",
        "all_positive",
        "include",
        "exclude",
        "NUnique",
        "bsiter",
        "N",
        "ND",
    ],
)
def test_locations_strings_ints(location, attr):
    expected = {
        "name": "Influent",
        "station_type": "inflow",
        "analysis_space": "lognormal",
        "rescol": "res",
        "qualcol": "qual",
        "plot_marker": "o",
        "scatter_marker": "v",
        "hasData": True,
        "all_positive": True,
        "exclude": False,
        "include": True,
        "NUnique": 30,
        "bsiter": 1500,
        "N": 35,
        "ND": 7,
    }
    assert getattr(location, attr) == expected[attr]


@pytest.mark.parametrize("attr", ["fractionND", "min", "min_DL", "min_detect", "max"])
def test_locations_numbers(location, attr):
    expected = {
        "fractionND": 0.2,
        "min": 2.0,
        "min_DL": 5.0,
        "min_detect": 2.0,
        "max": 22.97,
    }
    nptest.assert_approx_equal(getattr(location, attr), expected[attr])


@pytest.mark.parametrize(
    "attr",
    [
        "useros",
        "cov",
        "geomean",
        "geostd",
        "logmean",
        "logstd",
        "mean",
        "median",
        "pctl10",
        "pctl25",
        "pctl75",
        "pctl90",
        "skew",
        "std",
    ],
)
@helpers.seed
def test_location_stats_scalars(location, attr):
    expected = {
        "useros": {True: True, False: False},
        "cov": {True: 0.5887644, False: 0.5280314},
        "geomean": {True: 8.0779865, False: 8.8140731},
        "geostd": {True: 1.8116975, False: 1.7094616},
        "logmean": {True: 2.0891426, False: 2.1763497},
        "logstd": {True: 0.5942642, False: 0.5361785},
        "mean": {True: 9.5888515, False: 10.120571},
        "median": {True: 7.5000000, False: 8.7100000},
        "pctl10": {True: 4.0460279, False: 5.0000000},
        "pctl25": {True: 5.6150000, False: 5.8050000},
        "pctl75": {True: 11.725000, False: 11.725000},
        "pctl90": {True: 19.178000, False: 19.178000},
        "skew": {True: 0.8692107, False: 0.8537566},
        "std": {True: 5.6455746, False: 5.3439797},
    }
    nptest.assert_approx_equal(
        getattr(location, attr), expected[attr][location.useros], significant=5
    )


@pytest.mark.parametrize(
    "attr",
    [
        "geomean_conf_interval",
        "logmean_conf_interval",
        "mean_conf_interval",
        "median_conf_interval",
        "shapiro",
        "shapiro_log",
        "lilliefors",
        "lilliefors_log",
        "color",
    ],
)
def test_location_stats_arrays(location, attr):
    expected = {
        "color": {
            True: [0.32157, 0.45271, 0.66667],
            False: [0.32157, 0.45271, 0.66667],
        },
        "geomean_conf_interval": {True: [6.55572, 9.79677], False: [7.25255, 10.34346]},
        "logmean_conf_interval": {True: [1.88631, 2.27656], False: [1.97075, 2.34456]},
        "mean_conf_interval": {True: [7.74564, 11.49393], False: [8.52743, 11.97627]},
        "median_conf_interval": {True: [5.66000, 8.71000], False: [6.65000, 9.850000]},
        "shapiro": {True: [0.886889, 0.001789], False: [0.896744, 0.003236]},
        "shapiro_log": {True: [0.972679, 0.520949], False: [0.964298, 0.306435]},
        "lilliefors": {True: [0.185180, 0.004200], False: [0.160353, 0.02425]},
        "lilliefors_log": {True: [0.091855, 0.64099], False: [0.08148, 0.80351]},
    }
    nptest.assert_array_almost_equal(
        getattr(location, attr), expected[attr][location.useros], decimal=5
    )


@pytest.mark.parametrize("attr", ["anderson", "anderson_log"])
@pytest.mark.parametrize("index", [0, 1, 2, pytest.param(3, marks=pytest.mark.skip), 4])
def test_location_anderson(location, attr, index):
    expected = {
        "anderson": {
            True: (
                1.54388800,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15.0, 10.0, 5.0, 2.5, 1.0],
                None,
                0.000438139,
            ),
            False: (
                1.4392085,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15.0, 10.0, 5.0, 2.5, 1.0],
                None,
                0.00080268,
            ),
        },
        "anderson_log": {
            True: (
                0.30409634,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15.0, 10.0, 5.0, 2.5, 1.0],
                None,
                0.552806894,
            ),
            False: (
                0.3684061,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15.0, 10.0, 5.0, 2.5, 1.0],
                None,
                0.41004028,
            ),
        },
    }
    result = expected[attr][location.useros][index]
    if index in [0, 4]:
        nptest.assert_approx_equal(getattr(location, attr)[index], result, significant=5)
    elif index != 3:
        nptest.assert_array_almost_equal(getattr(location, attr)[index], result, decimal=5)


@pytest.fixture
def dataset():
    known_bsiter = 750
    in_data = helpers.getTestROSData()
    in_data["res"] += 3

    out_data = helpers.getTestROSData()
    out_data["res"] -= 1.5
    influent = Location(
        in_data,
        station_type="inflow",
        bsiter=known_bsiter,
        rescol="res",
        qualcol="qual",
        useros=False,
    )

    effluent = Location(
        out_data,
        station_type="outflow",
        bsiter=known_bsiter,
        rescol="res",
        qualcol="qual",
        useros=False,
    )

    return Dataset(influent, effluent)


def test_ds_data(dataset):
    assert hasattr(dataset, "data")
    assert isinstance(dataset.data, pandas.DataFrame)


def test_paired_data(dataset):
    assert hasattr(dataset, "paired_data")
    assert isinstance(dataset.paired_data, pandas.DataFrame)


def test__non_paired_stats(dataset):
    assert hasattr(dataset, "_non_paired_stats")
    known__non_paired_stats = True
    assert dataset._non_paired_stats == known__non_paired_stats


def test__paired_stats(dataset):
    known__paired_stats = True
    assert hasattr(dataset, "_paired_stats")
    assert dataset._paired_stats == known__paired_stats


def test_name(dataset):
    assert hasattr(dataset, "name")
    assert dataset.name is None


def test_name_set(dataset):
    assert hasattr(dataset, "name")
    testname = "Test Name"
    dataset.name = testname
    assert dataset.name == testname


def test_defintion_default(dataset):
    assert hasattr(dataset, "definition")
    assert dataset.definition == {}


def test_defintion_set(dataset):
    known_definition = {"attr1": "test1", "attr2": "test2"}
    assert hasattr(dataset, "definition")
    dataset.definition = known_definition
    assert dataset.definition == known_definition


def test_include_exclude(dataset):
    known_include = True
    assert hasattr(dataset, "include")
    assert dataset.include == known_include

    assert hasattr(dataset, "exclude")
    assert dataset.exclude == (not dataset.include)


def test_wilcoxon(dataset):
    known_wilcoxon_stats = (0.0, 2.469027e-07)
    known_wilcoxon_z = known_wilcoxon_stats[0]
    known_wilcoxon_p = known_wilcoxon_stats[1]

    assert hasattr(dataset, "wilcoxon_z")
    nptest.assert_allclose(dataset.wilcoxon_z, known_wilcoxon_z, rtol=TOLERANCE)

    assert hasattr(dataset, "wilcoxon_p")
    nptest.assert_allclose(dataset.wilcoxon_p, known_wilcoxon_p, rtol=TOLERANCE)

    assert hasattr(dataset, "_wilcoxon_stats")
    nptest.assert_allclose(dataset._wilcoxon_stats, known_wilcoxon_stats, rtol=TOLERANCE)


def test_mannwhitney(dataset):
    known_mannwhitney_stats = (927.0, 2.251523e-04)
    known_mannwhitney_u = known_mannwhitney_stats[0]
    known_mannwhitney_p = known_mannwhitney_stats[1]
    assert hasattr(dataset, "mannwhitney_u")
    nptest.assert_allclose(dataset.mannwhitney_u, known_mannwhitney_u, rtol=TOLERANCE)

    assert hasattr(dataset, "mannwhitney_p")
    nptest.assert_allclose(dataset.mannwhitney_p, known_mannwhitney_p, rtol=TOLERANCE)

    assert hasattr(dataset, "_mannwhitney_stats")
    nptest.assert_allclose(
        dataset._mannwhitney_stats,
        known_mannwhitney_stats,
        rtol=TOLERANCE,
    )


@pytest.mark.xfail(OLD_SCIPY, reason="Scipy < 0.19")
def test_kendall(dataset):
    known_kendall_stats = (1.00, 5.482137e-17)
    known_kendall_tau = known_kendall_stats[0]
    known_kendall_p = known_kendall_stats[1]
    assert hasattr(dataset, "kendall_tau")
    nptest.assert_allclose(dataset.kendall_tau, known_kendall_tau, rtol=TOLERANCE)

    assert hasattr(dataset, "kendall_p")
    nptest.assert_allclose(dataset.kendall_p, known_kendall_p, rtol=TOLERANCE)

    assert hasattr(dataset, "_kendall_stats")
    nptest.assert_allclose(dataset._kendall_stats, known_kendall_stats, rtol=TOLERANCE)


def test_spearman(dataset):
    known_spearman_stats = (1.0, 0.0)
    known_spearman_rho = known_spearman_stats[0]
    known_spearman_p = known_spearman_stats[1]
    assert hasattr(dataset, "spearman_rho")
    nptest.assert_allclose(dataset.spearman_rho, known_spearman_rho, atol=0.0001)

    assert hasattr(dataset, "spearman_p")
    nptest.assert_allclose(dataset.spearman_p, known_spearman_p, atol=0.0001)

    assert hasattr(dataset, "_spearman_stats")
    nptest.assert_allclose(dataset._spearman_stats, known_spearman_stats, atol=0.0001)


def test_theil(dataset):
    known_theil_stats = (1.0, -4.5, 1.0, 1.0)
    known_theil_hislope = known_theil_stats[0]
    known_theil_intercept = known_theil_stats[1]
    known_theil_loslope = known_theil_stats[2]
    known_theil_medslope = known_theil_stats[3]
    assert hasattr(dataset, "theil_medslope")
    nptest.assert_allclose(dataset.theil_medslope, known_theil_medslope, rtol=TOLERANCE)

    assert hasattr(dataset, "theil_intercept")
    nptest.assert_allclose(dataset.theil_intercept, known_theil_intercept, rtol=TOLERANCE)

    assert hasattr(dataset, "theil_loslope")
    nptest.assert_allclose(dataset.theil_loslope, known_theil_loslope, rtol=TOLERANCE)

    assert hasattr(dataset, "theil_hislope")
    nptest.assert_allclose(dataset.theil_hislope, known_theil_hislope, rtol=TOLERANCE)

    assert hasattr(dataset, "_theil_stats")
    nptest.assert_almost_equal(dataset._theil_stats["medslope"], known_theil_stats[0], decimal=4)

    nptest.assert_almost_equal(dataset._theil_stats["intercept"], known_theil_stats[1], decimal=4)

    nptest.assert_almost_equal(dataset._theil_stats["loslope"], known_theil_stats[2], decimal=4)

    nptest.assert_almost_equal(dataset._theil_stats["hislope"], known_theil_stats[3], decimal=4)

    assert not dataset._theil_stats["is_inverted"]

    assert "estimated_effluent" in list(dataset._theil_stats.keys())
    assert "estimate_error" in list(dataset._theil_stats.keys())


def test_medianCIsOverlap(dataset):
    known_medianCIsOverlap = False
    assert known_medianCIsOverlap == dataset.medianCIsOverlap


def test__repr__normal(dataset):
    dataset.__repr__


def test_repr__None(dataset):
    dataset.definition = None
    dataset.__repr__
