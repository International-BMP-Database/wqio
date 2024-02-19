from io import StringIO
from textwrap import dedent

import numpy
import numpy.testing as nptest
import pandas
import pandas.testing as pdtest
import pytest

from wqio import ros
from wqio.tests import helpers


@pytest.fixture
def basic_data():
    df = (
        helpers.getTestROSData()
        .assign(conc=lambda df: df["res"])
        .assign(censored=lambda df: df["qual"] == "ND")
    )
    return df


@pytest.fixture
def intermediate_data():
    df = pandas.DataFrame(
        [
            {"censored": True, "conc": 5.0, "det_limit_index": 1, "rank": 1},
            {"censored": True, "conc": 5.0, "det_limit_index": 1, "rank": 2},
            {"censored": True, "conc": 5.5, "det_limit_index": 2, "rank": 1},
            {"censored": True, "conc": 5.75, "det_limit_index": 3, "rank": 1},
            {"censored": True, "conc": 9.5, "det_limit_index": 4, "rank": 1},
            {"censored": True, "conc": 9.5, "det_limit_index": 4, "rank": 2},
            {"censored": True, "conc": 11.0, "det_limit_index": 5, "rank": 1},
            {"censored": False, "conc": 2.0, "det_limit_index": 0, "rank": 1},
            {"censored": False, "conc": 4.2, "det_limit_index": 0, "rank": 2},
            {"censored": False, "conc": 4.62, "det_limit_index": 0, "rank": 3},
            {"censored": False, "conc": 5.57, "det_limit_index": 2, "rank": 1},
            {"censored": False, "conc": 5.66, "det_limit_index": 2, "rank": 2},
            {"censored": False, "conc": 5.86, "det_limit_index": 3, "rank": 1},
            {"censored": False, "conc": 6.65, "det_limit_index": 3, "rank": 2},
            {"censored": False, "conc": 6.78, "det_limit_index": 3, "rank": 3},
            {"censored": False, "conc": 6.79, "det_limit_index": 3, "rank": 4},
            {"censored": False, "conc": 7.5, "det_limit_index": 3, "rank": 5},
            {"censored": False, "conc": 7.5, "det_limit_index": 3, "rank": 6},
            {"censored": False, "conc": 7.5, "det_limit_index": 3, "rank": 7},
            {"censored": False, "conc": 8.63, "det_limit_index": 3, "rank": 8},
            {"censored": False, "conc": 8.71, "det_limit_index": 3, "rank": 9},
            {"censored": False, "conc": 8.99, "det_limit_index": 3, "rank": 10},
            {"censored": False, "conc": 9.85, "det_limit_index": 4, "rank": 1},
            {"censored": False, "conc": 10.82, "det_limit_index": 4, "rank": 2},
            {"censored": False, "conc": 11.25, "det_limit_index": 5, "rank": 1},
            {"censored": False, "conc": 11.25, "det_limit_index": 5, "rank": 2},
            {"censored": False, "conc": 12.2, "det_limit_index": 5, "rank": 3},
            {"censored": False, "conc": 14.92, "det_limit_index": 5, "rank": 4},
            {"censored": False, "conc": 16.77, "det_limit_index": 5, "rank": 5},
            {"censored": False, "conc": 17.81, "det_limit_index": 5, "rank": 6},
            {"censored": False, "conc": 19.16, "det_limit_index": 5, "rank": 7},
            {"censored": False, "conc": 19.19, "det_limit_index": 5, "rank": 8},
            {"censored": False, "conc": 19.64, "det_limit_index": 5, "rank": 9},
            {"censored": False, "conc": 20.18, "det_limit_index": 5, "rank": 10},
            {"censored": False, "conc": 22.97, "det_limit_index": 5, "rank": 11},
        ]
    )

    return df


@pytest.fixture
def advanced_data():
    df = pandas.DataFrame(
        [
            {
                "Zprelim": -1.44562021,
                "censored": True,
                "conc": 5.0,
                "det_limit_index": 1,
                "plot_pos": 0.07414187,
                "rank": 1,
            },
            {
                "Zprelim": -1.22010353,
                "censored": True,
                "conc": 5.0,
                "det_limit_index": 1,
                "plot_pos": 0.11121281,
                "rank": 2,
            },
            {
                "Zprelim": -1.04382253,
                "censored": True,
                "conc": 5.5,
                "det_limit_index": 2,
                "plot_pos": 0.14828375,
                "rank": 1,
            },
            {
                "Zprelim": -1.04382253,
                "censored": True,
                "conc": 5.75,
                "det_limit_index": 3,
                "plot_pos": 0.14828375,
                "rank": 1,
            },
            {
                "Zprelim": -0.81095536,
                "censored": True,
                "conc": 9.5,
                "det_limit_index": 4,
                "plot_pos": 0.20869565,
                "rank": 1,
            },
            {
                "Zprelim": -0.40467790,
                "censored": True,
                "conc": 9.5,
                "det_limit_index": 4,
                "plot_pos": 0.34285714,
                "rank": 2,
            },
            {
                "Zprelim": -0.20857169,
                "censored": True,
                "conc": 11.0,
                "det_limit_index": 5,
                "plot_pos": 0.41739130,
                "rank": 1,
            },
            {
                "Zprelim": -1.59276546,
                "censored": False,
                "conc": 2.0,
                "det_limit_index": 0,
                "plot_pos": 0.05560640,
                "rank": 1,
            },
            {
                "Zprelim": -1.22010353,
                "censored": False,
                "conc": 4.2,
                "det_limit_index": 0,
                "plot_pos": 0.11121281,
                "rank": 2,
            },
            {
                "Zprelim": -0.96681116,
                "censored": False,
                "conc": 4.62,
                "det_limit_index": 0,
                "plot_pos": 0.16681922,
                "rank": 3,
            },
            {
                "Zprelim": -0.68351863,
                "censored": False,
                "conc": 5.57,
                "det_limit_index": 2,
                "plot_pos": 0.24713958,
                "rank": 1,
            },
            {
                "Zprelim": -0.60721672,
                "censored": False,
                "conc": 5.66,
                "det_limit_index": 2,
                "plot_pos": 0.27185354,
                "rank": 2,
            },
            {
                "Zprelim": -0.44953240,
                "censored": False,
                "conc": 5.86,
                "det_limit_index": 3,
                "plot_pos": 0.32652381,
                "rank": 1,
            },
            {
                "Zprelim": -0.36788328,
                "censored": False,
                "conc": 6.65,
                "det_limit_index": 3,
                "plot_pos": 0.35648013,
                "rank": 2,
            },
            {
                "Zprelim": -0.28861907,
                "censored": False,
                "conc": 6.78,
                "det_limit_index": 3,
                "plot_pos": 0.38643644,
                "rank": 3,
            },
            {
                "Zprelim": -0.21113039,
                "censored": False,
                "conc": 6.79,
                "det_limit_index": 3,
                "plot_pos": 0.41639276,
                "rank": 4,
            },
            {
                "Zprelim": -0.13489088,
                "censored": False,
                "conc": 7.5,
                "det_limit_index": 3,
                "plot_pos": 0.44634907,
                "rank": 5,
            },
            {
                "Zprelim": -0.05942854,
                "censored": False,
                "conc": 7.5,
                "det_limit_index": 3,
                "plot_pos": 0.47630538,
                "rank": 6,
            },
            {
                "Zprelim": 0.015696403,
                "censored": False,
                "conc": 7.5,
                "det_limit_index": 3,
                "plot_pos": 0.50626170,
                "rank": 7,
            },
            {
                "Zprelim": 0.090910169,
                "censored": False,
                "conc": 8.63,
                "det_limit_index": 3,
                "plot_pos": 0.53621801,
                "rank": 8,
            },
            {
                "Zprelim": 0.166642511,
                "censored": False,
                "conc": 8.71,
                "det_limit_index": 3,
                "plot_pos": 0.56617432,
                "rank": 9,
            },
            {
                "Zprelim": 0.243344267,
                "censored": False,
                "conc": 8.99,
                "det_limit_index": 3,
                "plot_pos": 0.59613064,
                "rank": 10,
            },
            {
                "Zprelim": 0.374443298,
                "censored": False,
                "conc": 9.85,
                "det_limit_index": 4,
                "plot_pos": 0.64596273,
                "rank": 1,
            },
            {
                "Zprelim": 0.428450751,
                "censored": False,
                "conc": 10.82,
                "det_limit_index": 4,
                "plot_pos": 0.66583850,
                "rank": 2,
            },
            {
                "Zprelim": 0.558957865,
                "censored": False,
                "conc": 11.25,
                "det_limit_index": 5,
                "plot_pos": 0.71190476,
                "rank": 1,
            },
            {
                "Zprelim": 0.637484160,
                "censored": False,
                "conc": 11.25,
                "det_limit_index": 5,
                "plot_pos": 0.73809523,
                "rank": 2,
            },
            {
                "Zprelim": 0.720156617,
                "censored": False,
                "conc": 12.2,
                "det_limit_index": 5,
                "plot_pos": 0.76428571,
                "rank": 3,
            },
            {
                "Zprelim": 0.808074633,
                "censored": False,
                "conc": 14.92,
                "det_limit_index": 5,
                "plot_pos": 0.79047619,
                "rank": 4,
            },
            {
                "Zprelim": 0.902734791,
                "censored": False,
                "conc": 16.77,
                "det_limit_index": 5,
                "plot_pos": 0.81666666,
                "rank": 5,
            },
            {
                "Zprelim": 1.006269985,
                "censored": False,
                "conc": 17.81,
                "det_limit_index": 5,
                "plot_pos": 0.84285714,
                "rank": 6,
            },
            {
                "Zprelim": 1.121900467,
                "censored": False,
                "conc": 19.16,
                "det_limit_index": 5,
                "plot_pos": 0.86904761,
                "rank": 7,
            },
            {
                "Zprelim": 1.254875912,
                "censored": False,
                "conc": 19.19,
                "det_limit_index": 5,
                "plot_pos": 0.89523809,
                "rank": 8,
            },
            {
                "Zprelim": 1.414746425,
                "censored": False,
                "conc": 19.64,
                "det_limit_index": 5,
                "plot_pos": 0.92142857,
                "rank": 9,
            },
            {
                "Zprelim": 1.622193585,
                "censored": False,
                "conc": 20.18,
                "det_limit_index": 5,
                "plot_pos": 0.94761904,
                "rank": 10,
            },
            {
                "Zprelim": 1.939989611,
                "censored": False,
                "conc": 22.97,
                "det_limit_index": 5,
                "plot_pos": 0.97380952,
                "rank": 11,
            },
        ]
    )

    return df


@pytest.fixture
def basic_cohn():
    cohn = pandas.DataFrame(
        [
            {
                "lower_dl": 2.0,
                "ncen_equal": 0.0,
                "nobs_below": 0.0,
                "nuncen_above": 3.0,
                "prob_exceedance": 1.0,
                "upper_dl": 5.0,
            },
            {
                "lower_dl": 5.0,
                "ncen_equal": 2.0,
                "nobs_below": 5.0,
                "nuncen_above": 0.0,
                "prob_exceedance": 0.77757437070938218,
                "upper_dl": 5.5,
            },
            {
                "lower_dl": 5.5,
                "ncen_equal": 1.0,
                "nobs_below": 6.0,
                "nuncen_above": 2.0,
                "prob_exceedance": 0.77757437070938218,
                "upper_dl": 5.75,
            },
            {
                "lower_dl": 5.75,
                "ncen_equal": 1.0,
                "nobs_below": 9.0,
                "nuncen_above": 10.0,
                "prob_exceedance": 0.7034324942791762,
                "upper_dl": 9.5,
            },
            {
                "lower_dl": 9.5,
                "ncen_equal": 2.0,
                "nobs_below": 21.0,
                "nuncen_above": 2.0,
                "prob_exceedance": 0.37391304347826088,
                "upper_dl": 11.0,
            },
            {
                "lower_dl": 11.0,
                "ncen_equal": 1.0,
                "nobs_below": 24.0,
                "nuncen_above": 11.0,
                "prob_exceedance": 0.31428571428571428,
                "upper_dl": numpy.inf,
            },
            {
                "lower_dl": numpy.nan,
                "ncen_equal": numpy.nan,
                "nobs_below": numpy.nan,
                "nuncen_above": numpy.nan,
                "prob_exceedance": 0.0,
                "upper_dl": numpy.nan,
            },
        ]
    )
    return cohn


@pytest.fixture
def expected_sorted():
    expected_sorted = pandas.DataFrame(
        [
            {"censored": True, "conc": 5.0},
            {"censored": True, "conc": 5.0},
            {"censored": True, "conc": 5.5},
            {"censored": True, "conc": 5.75},
            {"censored": True, "conc": 9.5},
            {"censored": True, "conc": 9.5},
            {"censored": True, "conc": 11.0},
            {"censored": False, "conc": 2.0},
            {"censored": False, "conc": 4.2},
            {"censored": False, "conc": 4.62},
            {"censored": False, "conc": 5.57},
            {"censored": False, "conc": 5.66},
            {"censored": False, "conc": 5.86},
            {"censored": False, "conc": 6.65},
            {"censored": False, "conc": 6.78},
            {"censored": False, "conc": 6.79},
            {"censored": False, "conc": 7.5},
            {"censored": False, "conc": 7.5},
            {"censored": False, "conc": 7.5},
            {"censored": False, "conc": 8.63},
            {"censored": False, "conc": 8.71},
            {"censored": False, "conc": 8.99},
            {"censored": False, "conc": 9.85},
            {"censored": False, "conc": 10.82},
            {"censored": False, "conc": 11.25},
            {"censored": False, "conc": 11.25},
            {"censored": False, "conc": 12.2},
            {"censored": False, "conc": 14.92},
            {"censored": False, "conc": 16.77},
            {"censored": False, "conc": 17.81},
            {"censored": False, "conc": 19.16},
            {"censored": False, "conc": 19.19},
            {"censored": False, "conc": 19.64},
            {"censored": False, "conc": 20.18},
            {"censored": False, "conc": 22.97},
        ]
    )[["conc", "censored"]]
    return expected_sorted


@pytest.fixture
def expected_cohn():
    final_cols = [
        "lower_dl",
        "upper_dl",
        "nuncen_above",
        "nobs_below",
        "ncen_equal",
        "prob_exceedance",
    ]
    cohn = pandas.DataFrame(
        [
            {
                "lower_dl": 2.0,
                "ncen_equal": 0.0,
                "nobs_below": 0.0,
                "nuncen_above": 3.0,
                "prob_exceedance": 1.0,
                "upper_dl": 5.0,
            },
            {
                "lower_dl": 5.0,
                "ncen_equal": 2.0,
                "nobs_below": 5.0,
                "nuncen_above": 0.0,
                "prob_exceedance": 0.77757437070938218,
                "upper_dl": 5.5,
            },
            {
                "lower_dl": 5.5,
                "ncen_equal": 1.0,
                "nobs_below": 6.0,
                "nuncen_above": 2.0,
                "prob_exceedance": 0.77757437070938218,
                "upper_dl": 5.75,
            },
            {
                "lower_dl": 5.75,
                "ncen_equal": 1.0,
                "nobs_below": 9.0,
                "nuncen_above": 10.0,
                "prob_exceedance": 0.7034324942791762,
                "upper_dl": 9.5,
            },
            {
                "lower_dl": 9.5,
                "ncen_equal": 2.0,
                "nobs_below": 21.0,
                "nuncen_above": 2.0,
                "prob_exceedance": 0.37391304347826088,
                "upper_dl": 11.0,
            },
            {
                "lower_dl": 11.0,
                "ncen_equal": 1.0,
                "nobs_below": 24.0,
                "nuncen_above": 11.0,
                "prob_exceedance": 0.31428571428571428,
                "upper_dl": numpy.inf,
            },
            {
                "lower_dl": numpy.nan,
                "ncen_equal": numpy.nan,
                "nobs_below": numpy.nan,
                "nuncen_above": numpy.nan,
                "prob_exceedance": 0.0,
                "upper_dl": numpy.nan,
            },
        ]
    )[final_cols]
    return cohn


def test__ros_sort_baseline(basic_data, expected_sorted):
    result = ros._ros_sort(basic_data, "conc", "censored")
    pdtest.assert_frame_equal(result, expected_sorted, rtol=1e-5)


def test__ros_sort_warning(basic_data, expected_sorted):
    df = basic_data.copy()
    max_row = df["conc"].idxmax()
    df.loc[max_row, "censored"] = True
    with pytest.warns(UserWarning):
        result = ros._ros_sort(df, "conc", "censored", warn=True)
        pdtest.assert_frame_equal(result, expected_sorted.iloc[:-1], rtol=1e-5)


def test_cohn_numbers_baseline(basic_data, expected_cohn):
    result = ros.cohn_numbers(basic_data, result="conc", censorship="censored")
    pdtest.assert_frame_equal(result, expected_cohn, rtol=1e-5)


def test_cohn_numbers_no_NDs(basic_data):
    result = ros.cohn_numbers(basic_data.assign(qual=False), result="conc", censorship="qual")
    assert result.shape == (0, 6)


def test__detection_limit_index_empty():
    empty_cohn = pandas.DataFrame(numpy.empty((0, 7)))
    assert ros._detection_limit_index(None, empty_cohn) == 0


@pytest.mark.parametrize(("value", "expected"), [(3.5, 0), (6.0, 3), (12.0, 5)])
def test__detection_limit_index_populated(value, expected, basic_cohn):
    result = ros._detection_limit_index(value, basic_cohn)
    assert result == expected


def test__detection_limit_index_out_of_bounds(basic_cohn):
    with helpers.raises(IndexError):
        ros._detection_limit_index(0, basic_cohn)


def test__ros_group_rank():
    df = pandas.DataFrame(
        {
            "dl_idx": [1] * 12,
            "params": list("AABCCCDE") + list("DCBA"),
            "values": list(range(12)),
        }
    )

    result = ros._ros_group_rank(df, "dl_idx", "params")
    expected = pandas.Series([1, 2, 1, 1, 2, 3, 1, 1, 2, 4, 2, 3], name="rank")
    pdtest.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"censored": False, "det_limit_index": 2, "rank": 1}, 0.24713958810068648),
        ({"censored": False, "det_limit_index": 2, "rank": 12}, 0.51899313501144173),
        ({"censored": True, "det_limit_index": 5, "rank": 4}, 1.3714285714285714),
        ({"censored": True, "det_limit_index": 4, "rank": 2}, 0.41739130434782606),
    ],
)
def test__ros_plot_pos(row, expected, basic_cohn):
    result = ros._ros_plot_pos(row, "censored", basic_cohn)
    assert (abs(result - expected) / expected) < 0.00001


def test__norm_plot_pos():
    result = ros._norm_plot_pos([1, 2, 3, 4])
    expected = numpy.array([0.159104, 0.385452, 0.614548, 0.840896])
    nptest.assert_array_almost_equal(result, expected)


def test_plotting_positions(intermediate_data, basic_cohn):
    results = ros.plotting_positions(intermediate_data, "censored", basic_cohn)
    expected = numpy.array(
        [
            0.07414188,
            0.11121281,
            0.14828375,
            0.14828375,
            0.20869565,
            0.34285714,
            0.41739130,
            0.05560641,
            0.11121281,
            0.16681922,
            0.24713959,
            0.27185355,
            0.32652382,
            0.35648013,
            0.38643645,
            0.41639276,
            0.44634907,
            0.47630539,
            0.50626170,
            0.53621802,
            0.56617433,
            0.59613064,
            0.64596273,
            0.66583851,
            0.71190476,
            0.73809524,
            0.76428571,
            0.79047619,
            0.81666667,
            0.84285714,
            0.86904762,
            0.89523810,
            0.92142857,
            0.94761905,
            0.97380952,
        ]
    )
    nptest.assert_array_almost_equal(results, expected)


def test__ros_estimate(advanced_data):
    expected = numpy.array(
        [
            3.11279729,
            3.60634338,
            4.04602788,
            4.04602788,
            4.71008116,
            6.14010906,
            6.97841457,
            2.00000000,
            4.20000000,
            4.62000000,
            5.57000000,
            5.66000000,
            5.86000000,
            6.65000000,
            6.78000000,
            6.79000000,
            7.50000000,
            7.50000000,
            7.50000000,
            8.63000000,
            8.71000000,
            8.99000000,
            9.85000000,
            10.82000000,
            11.25000000,
            11.25000000,
            12.20000000,
            14.92000000,
            16.77000000,
            17.81000000,
            19.16000000,
            19.19000000,
            19.64000000,
            20.18000000,
            22.97,
        ]
    )
    df = advanced_data.pipe(ros._ros_estimate, "conc", "censored", numpy.log, numpy.exp)
    result = df["final"].values
    nptest.assert_array_almost_equal(result, expected)


def test__do_ros_basic(basic_data):
    expected = numpy.array(
        [
            3.11279729,
            3.60634338,
            4.04602788,
            4.04602788,
            4.71008116,
            6.14010906,
            6.97841457,
            2.00000000,
            4.20000000,
            4.62000000,
            5.57000000,
            5.66000000,
            5.86000000,
            6.65000000,
            6.78000000,
            6.79000000,
            7.50000000,
            7.50000000,
            7.50000000,
            8.63000000,
            8.71000000,
            8.99000000,
            9.85000000,
            10.82000000,
            11.25000000,
            11.25000000,
            12.20000000,
            14.92000000,
            16.77000000,
            17.81000000,
            19.16000000,
            19.19000000,
            19.64000000,
            20.18000000,
            22.97,
        ]
    )

    df = basic_data.pipe(ros._do_ros, "conc", "censored", numpy.log, numpy.exp)
    result = df["final"].values
    nptest.assert_array_almost_equal(result, expected)


def test__do_ros_basic_with_floor(basic_data):
    expected = numpy.array(
        [
            5.00000000,
            5.00000000,
            5.00000000,
            5.00000000,
            5.00000000,
            6.14010906,
            6.97841457,
            5.00000000,
            5.00000000,
            5.00000000,
            5.57000000,
            5.66000000,
            5.86000000,
            6.65000000,
            6.78000000,
            6.79000000,
            7.50000000,
            7.50000000,
            7.50000000,
            8.63000000,
            8.71000000,
            8.99000000,
            9.85000000,
            10.82000000,
            11.25000000,
            11.25000000,
            12.20000000,
            14.92000000,
            16.77000000,
            17.81000000,
            19.16000000,
            19.19000000,
            19.64000000,
            20.18000000,
            22.97,
        ]
    )

    df = basic_data.pipe(ros._do_ros, "conc", "censored", numpy.log, numpy.exp, floor=5)
    result = df["final"].values
    nptest.assert_array_almost_equal(result, expected)


def test__do_ros_all_equal_some_cen():
    expected = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    result = (
        pandas.DataFrame(
            {
                "res": numpy.array(expected),
                "cen": numpy.array([True, True, True, True, True, False, True, False]),
            }
        )
        .pipe(ros._do_ros, "res", "cen", numpy.log, numpy.exp)["final"]
        .values
    )
    nptest.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    ["as_obj", "expected"],
    [(True, {"enough_uncensored": True, "not_too_many_censored": True}), (False, True)],
)
def test_is_valid_to_ros(basic_data, as_obj, expected):
    result = ros.is_valid_to_ros(basic_data, "censored", as_obj=as_obj)
    assert result == expected


class HelselAppendixB:
    """
    Appendix B dataset from "Estimation of Descriptive Statists for
    Multiply Censored Water Quality Data", Water Resources Research,
    Vol 24, No 12, pp 1997 - 2004. December 1988.
    """

    decimal = 2
    res = numpy.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
            10.0,
            10.0,
            3.0,
            7.0,
            9.0,
            12.0,
            15.0,
            20.0,
            27.0,
            33.0,
            50.0,
        ]
    )
    cen = numpy.array(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    rescol = "obs"
    cencol = "cen"
    df = pandas.DataFrame({rescol: res, cencol: cen})
    values = numpy.array(
        [
            0.47,
            0.85,
            1.11,
            1.27,
            1.76,
            2.34,
            2.50,
            3.00,
            3.03,
            4.80,
            7.00,
            9.00,
            12.0,
            15.0,
            20.0,
            27.0,
            33.0,
            50.0,
        ]
    )

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([3.0, 6.0, numpy.nan]),
            "nobs_below": numpy.array([6.0, 12.0, numpy.nan]),
            "ncen_equal": numpy.array([6.0, 3.0, numpy.nan]),
            "prob_exceedance": numpy.array([0.5555, 0.3333, 0.0]),
        }
    )


class HelselArsenic:
    """
    Oahu arsenic data from Nondetects and Data Analysis by
    Dennis R. Helsel (John Wiley, 2005)

    Plotting positions are fudged since relative to source data since
    modeled data is what matters and (source data plot positions are
    not uniformly spaced, which seems weird)
    """

    decimal = 2
    res = numpy.array(
        [
            3.2,
            2.8,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.7,
            1.5,
            1.0,
            1.0,
            1.0,
            1.0,
            0.9,
            0.9,
            0.7,
            0.7,
            0.6,
            0.5,
            0.5,
            0.5,
        ]
    )

    cen = numpy.array(
        [
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    rescol = "obs"
    cencol = "cen"
    df = pandas.DataFrame({rescol: res, cencol: cen})
    values = numpy.array(
        [
            3.20,
            2.80,
            1.42,
            1.14,
            0.95,
            0.81,
            0.68,
            0.57,
            0.46,
            0.35,
            1.70,
            1.50,
            0.98,
            0.76,
            0.58,
            0.41,
            0.90,
            0.61,
            0.70,
            0.70,
            0.60,
            0.50,
            0.50,
            0.50,
        ]
    )

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([6.0, 1.0, 2.0, 2.0, numpy.nan]),
            "nobs_below": numpy.array([0.0, 7.0, 12.0, 22.0, numpy.nan]),
            "ncen_equal": numpy.array([0.0, 1.0, 4.0, 8.0, numpy.nan]),
            "prob_exceedance": numpy.array([1.0, 0.3125, 0.2143, 0.0833, 0.0]),
        }
    )


class RNADAdata:
    decimal = 3
    datastring = StringIO(
        dedent(
            """\
        res cen
        0.090  True
        0.090  True
        0.090  True
        0.101 False
        0.136 False
        0.340 False
        0.457 False
        0.514 False
        0.629 False
        0.638 False
        0.774 False
        0.788 False
        0.900  True
        0.900  True
        0.900  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000 False
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.100 False
        2.000 False
        2.000 False
        2.404 False
        2.860 False
        3.000 False
        3.000 False
        3.705 False
        4.000 False
        5.000 False
        5.960 False
        6.000 False
        7.214 False
       16.000 False
       17.716 False
       25.000 False
       51.000 False
    """
        )
    )
    rescol = "res"
    cencol = "cen"
    df = pandas.read_csv(datastring, sep=r"\s+")
    values = numpy.array(
        [
            0.01907990,
            0.03826254,
            0.06080717,
            0.10100000,
            0.13600000,
            0.34000000,
            0.45700000,
            0.51400000,
            0.62900000,
            0.63800000,
            0.77400000,
            0.78800000,
            0.08745914,
            0.25257575,
            0.58544205,
            0.01711153,
            0.03373885,
            0.05287083,
            0.07506079,
            0.10081573,
            1.00000000,
            0.13070334,
            0.16539309,
            0.20569039,
            0.25257575,
            0.30725491,
            0.37122555,
            0.44636843,
            0.53507405,
            0.64042242,
            0.76644378,
            0.91850581,
            1.10390531,
            1.10000000,
            2.00000000,
            2.00000000,
            2.40400000,
            2.86000000,
            3.00000000,
            3.00000000,
            3.70500000,
            4.00000000,
            5.00000000,
            5.96000000,
            6.00000000,
            7.21400000,
            16.00000000,
            17.71600000,
            25.00000000,
            51.00000000,
        ]
    )

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([9.0, 0.0, 18.0, numpy.nan]),
            "nobs_below": numpy.array([3.0, 15.0, 32.0, numpy.nan]),
            "ncen_equal": numpy.array([3.0, 3.0, 17.0, numpy.nan]),
            "prob_exceedance": numpy.array([0.84, 0.36, 0.36, 0]),
        }
    )


class NoOp_ZeroND:
    decimal = 2
    numpy.random.seed(0)
    N = 20
    res = numpy.random.lognormal(size=N)
    cen = [False] * N
    rescol = "obs"
    cencol = "cen"
    df = pandas.DataFrame({rescol: res, cencol: cen})
    values = numpy.array(
        [
            0.38,
            0.43,
            0.81,
            0.86,
            0.90,
            1.13,
            1.15,
            1.37,
            1.40,
            1.49,
            1.51,
            1.56,
            2.14,
            2.59,
            2.66,
            4.28,
            4.46,
            5.84,
            6.47,
            9.4,
        ]
    )

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([]),
            "nobs_below": numpy.array([]),
            "ncen_equal": numpy.array([]),
            "prob_exceedance": numpy.array([]),
        }
    )


class OneND:
    decimal = 3
    res = numpy.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
            10.0,
            10.0,
            3.0,
            7.0,
            9.0,
            12.0,
            15.0,
            20.0,
            27.0,
            33.0,
            50.0,
        ]
    )
    cen = numpy.array(
        [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    rescol = "conc"
    cencol = "cen"
    df = pandas.DataFrame({rescol: res, cencol: cen})
    values = numpy.array(
        [
            0.24,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
            10.0,
            10.0,
            3.00,
            7.0,
            9.0,
            12.0,
            15.0,
            20.0,
            27.0,
            33.0,
            50.0,
        ]
    )

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([17.0, numpy.nan]),
            "nobs_below": numpy.array([1.0, numpy.nan]),
            "ncen_equal": numpy.array([1.0, numpy.nan]),
            "prob_exceedance": numpy.array([0.9444, 0.0]),
        }
    )


class HalfDLs_80pctNDs:
    decimal = 3
    res = numpy.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
            10.0,
            10.0,
            3.0,
            7.0,
            9.0,
            12.0,
            15.0,
            20.0,
            27.0,
            33.0,
            50.0,
        ]
    )
    cen = numpy.array(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ]
    )
    rescol = "value"
    cencol = "qual"
    df = pandas.DataFrame({rescol: res, cencol: cen})
    values = numpy.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            5.0,
            5.0,
            5.0,
            1.5,
            3.5,
            4.5,
            6.0,
            7.5,
            10.0,
            27.0,
            33.0,
            50.0,
        ]
    )

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, numpy.nan]),
            "nobs_below": numpy.array([6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 15.0, numpy.nan]),
            "ncen_equal": numpy.array([6.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, numpy.nan]),
            "prob_exceedance": numpy.array([0.1667] * 8 + [0.0]),
        }
    )


class HaflDLs_OneUncensored:
    decimal = 3
    res = numpy.array([1.0, 1.0, 12.0, 15.0])
    cen = numpy.array([True, True, True, False])
    rescol = "value"
    cencol = "qual"
    df = pandas.DataFrame({rescol: res, cencol: cen})
    values = numpy.array([0.5, 0.5, 6.0, 15.0])

    cohn = pandas.DataFrame(
        {
            "nuncen_above": numpy.array([0.0, 1.0, numpy.nan]),
            "nobs_below": numpy.array([2.0, 3.0, numpy.nan]),
            "ncen_equal": numpy.array([2.0, 1.0, numpy.nan]),
            "prob_exceedance": numpy.array([0.25, 0.25, 0.0]),
        }
    )


class MaxCen_GT_MaxUncen(HelselAppendixB):
    res = numpy.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
            10.0,
            10.0,
            3.0,
            7.0,
            9.0,
            12.0,
            15.0,
            20.0,
            27.0,
            33.0,
            50.0,
            60,
            70,
        ]
    )
    cen = numpy.array(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ]
    )


class OnlyDL_GT_MaxUncen(NoOp_ZeroND):
    numpy.random.seed(0)
    N = 20
    res = [
        0.38,
        0.43,
        0.81,
        0.86,
        0.90,
        1.13,
        1.15,
        1.37,
        1.40,
        1.49,
        1.51,
        1.56,
        2.14,
        2.59,
        2.66,
        4.28,
        4.46,
        5.84,
        6.47,
        9.40,
        10.0,
        10.0,
    ]
    cen = ([False] * N) + [True, True]


@pytest.mark.parametrize(
    "case",
    [
        HelselAppendixB,
        HelselArsenic,
        RNADAdata,
        NoOp_ZeroND,
        OneND,
        HalfDLs_80pctNDs,
        HaflDLs_OneUncensored,
        MaxCen_GT_MaxUncen,
        OnlyDL_GT_MaxUncen,
    ],
)
@pytest.mark.parametrize("as_arrays", [True, False])
def test_ros_from_literature(as_arrays, case):
    if as_arrays:
        result = ros.ROS(case.df[case.rescol], case.df[case.cencol], df=None)
    else:
        result = ros.ROS(case.rescol, case.cencol, df=case.df)

    nptest.assert_array_almost_equal(sorted(result), sorted(case.values), decimal=case.decimal)


@pytest.mark.parametrize(
    "case",
    [
        HelselAppendixB,
        HelselArsenic,
        RNADAdata,
        NoOp_ZeroND,
        OneND,
        HalfDLs_80pctNDs,
        HaflDLs_OneUncensored,
        MaxCen_GT_MaxUncen,
        OnlyDL_GT_MaxUncen,
    ],
)
def test_cohn_from_literature(case):
    cols = ["nuncen_above", "nobs_below", "ncen_equal", "prob_exceedance"]
    result = ros.cohn_numbers(case.df, case.rescol, case.cencol)
    pdtest.assert_frame_equal(result[cols].round(5), case.cohn[cols].round(5), atol=1e-4)
