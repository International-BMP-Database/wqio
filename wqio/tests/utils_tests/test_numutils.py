import types
from collections import namedtuple
from io import StringIO
from textwrap import dedent
from typing import Any, Literal

import numpy
import numpy.testing as nptest
import pandas
import pandas.testing as pdtest
import pytest
import statsmodels.api as sm
from numpy._typing._array_like import NDArray
from pandas import DataFrame
from scipy import stats

from wqio.tests import helpers
from wqio.utils import numutils


@pytest.mark.parametrize(
    ("value", "N", "tex", "forceint", "pval", "expected", "error"),
    [
        (1234.56, 3, False, False, False, "1,230", None),
        (1234.56, 4, False, False, False, "1,235", None),
        (1234.56, 8, False, False, False, "1,234.5600", None),
        (1234.56, 0, False, False, False, None, ValueError),
        (1234.56, -1, False, False, False, None, ValueError),
        (1234.56 * 1e5, 3, False, False, False, "1.23e+08", None),
        (1234.56 * 1e5, 3, True, False, False, r"$1.23 \times 10 ^ {8}$", None),
        (1234.56, 3, False, True, False, "1,235", None),
        (0.123456, 3, False, False, False, "0.123", None),
        (0.123456, 4, False, False, False, "0.1235", None),
        (0.123456, 8, False, False, False, "0.12345600", None),
        (0.123456, -1, False, False, False, None, ValueError),
        (0.123456, 0, False, False, False, None, ValueError),
        (0.123456 * 1e-6, 3, False, False, False, "1.23e-07", None),
        (0.123456 * 1e-6, 3, True, False, False, r"$1.23 \times 10 ^ {-7}$", None),
        (0.123456, 3, False, True, False, "0", None),
        (0.001, 1, False, False, True, "0.001", None),
        (0.0005, 3, False, False, True, "<0.001", None),
        (0.0005, 3, True, False, True, "$<0.001$", None),
        ("1234", 3, False, False, False, "1234", None),
        (numpy.nan, 3, False, False, False, "NA", None),
        (numpy.inf, 3, False, False, False, "NA", None),
        (numpy.inf * -1, 3, False, False, False, "NA", None),
    ],
)
def test_sig_figs(value, N, tex, forceint, pval, expected, error):
    with helpers.raises(error):
        result = numutils.sig_figs(value, N, tex=tex, forceint=forceint, pval=pval)
        assert result == expected


@pytest.mark.parametrize(
    ("num", "qual", "N", "expected"),
    [
        (12498.124, "<", 3, "<12,500"),
        (12.540, ">", 6, ">12.5400"),
        (12.540, "", 3, "12.5"),
        (0.00257, "=", 2, "=0.0026"),
    ],
)
def test_format_result(num, qual, N, expected):
    assert numutils.format_result(num, qual, N) == expected


@pytest.mark.parametrize(
    ("fxn", "pval", "expected", "error_to_raise"),
    [
        (numutils.process_p_vals, None, "NA", None),
        (numutils.process_p_vals, 0.0005, "<0.001", None),
        (numutils.process_p_vals, 0.005, "0.005", None),
        (numutils.process_p_vals, 0.001, "0.001", None),
        (numutils.process_p_vals, 0.0012, "0.001", None),
        (numutils.process_p_vals, 0.06, "0.060", None),
        (numutils.process_p_vals, 1.01, None, ValueError),
    ],
)
def test_process_p_vals(fxn, pval, expected, error_to_raise):
    with helpers.raises(error_to_raise):
        result = fxn(pval)
        assert result == expected


@pytest.mark.parametrize("as_emoji", [False, True])
@pytest.mark.parametrize(
    ("pval", "expected", "error_to_raise"),
    [
        (None, ("NA", r"ಠ_ಠ"), None),
        (0.005, ("maybe", r"¯\(ツ)/¯"), None),
        (0.03, ("maybe (weak)", r"¯\_(ツ)_/¯"), None),
        (0.06, ("maybe (very weak)", r"¯\__(ツ)__/¯"), None),
        (0.11, ("nope", r"(╯°□°)╯︵ ┻━┻"), None),
        (1.01, (None, None), ValueError),
    ],
)
def test_translate_p_vals(pval, expected, as_emoji: bool, error_to_raise):
    with helpers.raises(error_to_raise):
        result = numutils.translate_p_vals(pval, as_emoji=as_emoji)
        assert result == expected[as_emoji]


def test_anderson_darling():
    data = helpers.getTestROSData()
    result = numutils.anderson_darling(data["res"])

    tolerance = 0.05
    known_anderson = (
        1.4392085,
        [0.527, 0.6, 0.719, 0.839, 0.998],
        [15.0, 10.0, 5.0, 2.5, 1.0],
        None,
        0.00080268,
    )

    # Ad stat
    assert abs(result[0] - known_anderson[0]) < 0.0001

    # critical values
    nptest.assert_allclose(result[1], known_anderson[1], rtol=tolerance)

    # significance level
    nptest.assert_allclose(result[2], known_anderson[2], rtol=tolerance)

    # p-value
    assert abs(result[4] - known_anderson[4]) < 0.0000001


@pytest.mark.parametrize("which", ["good", "bad"])
def test_processAndersonDarlingResults(which: Literal["good"] | Literal["bad"]):
    fieldnames = ["statistic", "critical_values", "significance_level"]
    AndersonResult = namedtuple("AndersonResult", fieldnames)
    ARs = {
        "bad": AndersonResult(
            statistic=0.30194681312357829,
            critical_values=numpy.array([0.529, 0.602, 0.722, 0.842, 1.002]),
            significance_level=numpy.array([15.0, 10.0, 5.0, 2.5, 1.0]),
        ),
        "good": AndersonResult(
            statistic=numpy.inf,
            critical_values=numpy.array([0.907, 1.061, 1.32, 1.58, 1.926]),
            significance_level=numpy.array([15.0, 10.0, 5.0, 2.50, 1.0]),
        ),
    }

    expected = {"bad": "<85.0%", "good": "99.0%"}

    assert numutils.process_AD_result(ARs[which]) == expected[which]


@pytest.mark.parametrize(
    ("AD", "n_points", "expected"),
    [
        (0.302, 1, 0.0035899),
        (0.302, 5, 0.4155200),
        (0.302, 15, 0.5325205),
        (0.150, 105, 0.9616770),
    ],
)
def test__anderson_darling_p_vals(AD, n_points, expected):
    ad_result = (AD, None, None)
    p_val = numutils._anderson_darling_p_vals(ad_result, n_points)
    assert abs(p_val - expected) < 0.0001


@pytest.fixture
def units_norm_data():
    data_csv = StringIO(
        dedent(
            """\
        storm,param,station,units,conc
        1,"Lead, Total",inflow,ug/L,10
        2,"Lead, Total",inflow,mg/L,0.02
        3,"Lead, Total",inflow,g/L,0.00003
        4,"Lead, Total",inflow,ug/L,40
        1,"Lead, Total",outflow,mg/L,0.05
        2,"Lead, Total",outflow,ug/L,60
        3,"Lead, Total",outflow,ug/L,70
        4,"Lead, Total",outflow,g/L,0.00008
        1,"Cadmium, Total",inflow,ug/L,10
        2,"Cadmium, Total",inflow,mg/L,0.02
        3,"Cadmium, Total",inflow,g/L,0.00003
        4,"Cadmium, Total",inflow,ug/L,40
        1,"Cadmium, Total",outflow,mg/L,0.05
        2,"Cadmium, Total",outflow,ug/L,60
        3,"Cadmium, Total",outflow,ug/L,70
        4,"Cadmium, Total",outflow,g/L,0.00008
    """
        )
    )
    raw = pandas.read_csv(data_csv)

    known_csv = StringIO(
        dedent(
            """\
        storm,param,station,units,conc
        1,"Lead, Total",inflow,ug/L,10.
        2,"Lead, Total",inflow,ug/L,20.
        3,"Lead, Total",inflow,ug/L,30.
        4,"Lead, Total",inflow,ug/L,40.
        1,"Lead, Total",outflow,ug/L,50.
        2,"Lead, Total",outflow,ug/L,60.
        3,"Lead, Total",outflow,ug/L,70.
        4,"Lead, Total",outflow,ug/L,80.
        1,"Cadmium, Total",inflow,mg/L,0.010
        2,"Cadmium, Total",inflow,mg/L,0.020
        3,"Cadmium, Total",inflow,mg/L,0.030
        4,"Cadmium, Total",inflow,mg/L,0.040
        1,"Cadmium, Total",outflow,mg/L,0.050
        2,"Cadmium, Total",outflow,mg/L,0.060
        3,"Cadmium, Total",outflow,mg/L,0.070
        4,"Cadmium, Total",outflow,mg/L,0.080
    """
        )
    )
    expected = pandas.read_csv(known_csv)
    return raw, expected


def test_normalize_units(units_norm_data: tuple[DataFrame, DataFrame]):
    unitsmap = {"ug/L": 1e-6, "mg/L": 1e-3, "g/L": 1e0}

    targetunits = {"Lead, Total": "ug/L", "Cadmium, Total": "mg/L"}
    raw, expected = units_norm_data
    result = numutils.normalize_units(
        raw, unitsmap, targetunits, paramcol="param", rescol="conc", unitcol="units"
    )
    pdtest.assert_frame_equal(result, expected)


def test_normalize_units_bad_targetunits(units_norm_data: tuple[DataFrame, DataFrame]):
    unitsmap = {"ug/L": 1e-6, "mg/L": 1e-3, "g/L": 1e0}

    targetunits = {"Lead, Total": "ug/L"}
    raw, expected = units_norm_data
    with helpers.raises(ValueError):
        numutils.normalize_units(
            raw,
            unitsmap,
            targetunits,
            paramcol="param",
            rescol="conc",
            unitcol="units",
            napolicy="raise",
        )


def test_normalize_units_bad_normalization(units_norm_data: tuple[DataFrame, DataFrame]):
    unitsmap = {"mg/L": 1e-3, "g/L": 1e0}

    targetunits = {"Lead, Total": "ug/L", "Cadmium, Total": "mg/L"}
    raw, expected = units_norm_data
    with helpers.raises(ValueError):
        numutils.normalize_units(
            raw,
            unitsmap,
            targetunits,
            paramcol="param",
            rescol="conc",
            unitcol="units",
            napolicy="raise",
        )


def test_normalize_units_bad_conversion(units_norm_data: tuple[DataFrame, DataFrame]):
    unitsmap = {"ug/L": 1e-6, "mg/L": 1e-3, "g/L": 1e0}

    targetunits = {"Lead, Total": "ng/L", "Cadmium, Total": "mg/L"}
    raw, expected = units_norm_data
    with helpers.raises(ValueError):
        numutils.normalize_units(
            raw,
            unitsmap,
            targetunits,
            paramcol="param",
            rescol="conc",
            unitcol="units",
            napolicy="raise",
        )


@pytest.mark.parametrize(
    ("pH", "expected", "error"),
    [
        (4, 0.10072764682551091, None),
        (14.1, None, ValueError),
        (-0.1, None, ValueError),
    ],
)
def test_test_pH2concentration(pH, expected, error):
    with helpers.raises(error):
        assert abs(numutils.pH_to_concentration(pH) - expected) < 0.0001


@helpers.seed
@pytest.mark.parametrize("error", [None, ValueError])
def test_compute_theilslope_default(error: types.NoneType | type[ValueError]):
    with helpers.raises(error):
        y = helpers.getTestROSData()["res"].values
        x = numpy.arange(len(y) - 1) if error else None
        assert tuple(numutils.compute_theilslope(y, x)) == stats.mstats.theilslopes(y, x)


@pytest.fixture
def fit_data():
    data = {
        "data": numpy.array(
            [
                2.000,
                4.000,
                4.620,
                5.000,
                5.000,
                5.500,
                5.570,
                5.660,
                5.750,
                5.860,
                6.650,
                6.780,
                6.790,
                7.500,
                7.500,
                7.500,
                8.630,
                8.710,
                8.990,
                9.500,
                9.500,
                9.850,
                10.82,
                11.00,
                11.25,
                11.25,
                12.20,
                14.92,
                16.77,
                17.81,
                19.16,
                19.19,
                19.64,
                20.18,
                22.97,
            ]
        ),
        "zscores": numpy.array(
            [
                -2.06188401,
                -1.66883254,
                -1.43353970,
                -1.25837339,
                -1.11509471,
                -0.99166098,
                -0.88174260,
                -0.78156696,
                -0.68868392,
                -0.60139747,
                -0.51847288,
                -0.43897250,
                -0.36215721,
                -0.28742406,
                -0.21426459,
                -0.14223572,
                -0.07093824,
                0.00000000,
                0.07093824,
                0.14223572,
                0.21426459,
                0.28742406,
                0.36215721,
                0.43897250,
                0.51847288,
                0.60139747,
                0.68868392,
                0.78156696,
                0.88174260,
                0.99166098,
                1.11509471,
                1.25837339,
                1.43353970,
                1.66883254,
                2.06188401,
            ]
        ),
        "y": numpy.array(
            [
                0.07323274,
                0.12319301,
                0.16771455,
                0.17796950,
                0.21840761,
                0.25757016,
                0.27402650,
                0.40868106,
                0.44872637,
                0.53673530,
                0.55169933,
                0.56211726,
                0.62375442,
                0.66631353,
                0.68454978,
                0.72137134,
                0.87602096,
                0.94651962,
                1.01927875,
                1.06040448,
                1.07966792,
                1.17969506,
                1.21132273,
                1.30751428,
                1.45371899,
                1.76381932,
                1.98832275,
                2.09275652,
                2.66552831,
                2.86453334,
                3.23039631,
                4.23953492,
                4.25892247,
                4.58347660,
                6.53100725,
            ]
        ),
    }
    z2, _y = stats.probplot(data["y"], fit=False)
    data["probs"] = stats.norm.cdf(data["zscores"]) * 100.0
    data["p2"] = stats.norm.cdf(z2) * 100
    return data


@pytest.mark.parametrize(
    ("fitlogs", "fitprobs", "error"),
    [
        (None, None, None),
        (None, None, None),
        ("y", None, None),
        (None, "y", None),
        ("x", None, None),
        ("both", None, None),
        ("x", "y", None),
        (None, "x", None),
        ("y", "x", None),
        (None, "both", None),
        ("junk", None, ValueError),
        (None, "junk", ValueError),
    ],
)
def test_fit_line(fit_data: dict[str, NDArray[Any]], fitlogs, fitprobs, error):
    xy = {
        (None, None): (fit_data["zscores"], fit_data["data"]),
        ("y", None): (fit_data["zscores"], fit_data["data"]),
        (None, "y"): (fit_data["data"], fit_data["probs"]),
        ("x", None): (fit_data["data"], fit_data["zscores"]),
        ("both", None): (fit_data["data"], fit_data["y"]),
        ("x", "y"): (fit_data["data"], fit_data["probs"]),
        (None, "x"): (fit_data["probs"], fit_data["data"]),
        ("y", "x"): (fit_data["probs"], fit_data["data"]),
        (None, "both"): (fit_data["probs"], fit_data["p2"]),
        ("junk", None): (fit_data["zscores"], fit_data["data"]),
        (None, "junk"): (fit_data["zscores"], fit_data["data"]),
    }

    expected = {
        (None, None): numpy.array([-0.89650596, 21.12622025]),
        ("y", None): numpy.array([2.80190754, 27.64958934]),
        (None, "y"): numpy.array([8.48666156, 98.51899616]),
        ("x", None): numpy.array([-2.57620461, 1.66767934]),
        ("both", None): numpy.array([0.04681540, 5.73261406]),
        ("x", "y"): numpy.array([0.49945757, 95.23103009]),
        (None, "x"): numpy.array([-0.89650596, 21.12622025]),
        ("y", "x"): numpy.array([2.80190754, 27.64958934]),
        (None, "both"): numpy.array([1.96093902, 98.03906098]),
        ("junk", None): None,
        (None, "junk"): None,
    }

    with helpers.raises(error):
        x_, y_, res = numutils.fit_line(
            xy[(fitlogs, fitprobs)][0],
            xy[(fitlogs, fitprobs)][1],
            fitlogs=fitlogs,
            fitprobs=fitprobs,
        )
        nptest.assert_array_almost_equal(y_, expected[(fitlogs, fitprobs)])
        assert isinstance(res, sm.regression.linear_model.RegressionResultsWrapper)


def test_fit_line_through_origin(fit_data: dict[str, NDArray[Any]]):
    x, y = fit_data["zscores"], fit_data["data"]
    x_, y_, res = numutils.fit_line(x, y, through_origin=True)
    assert res.params[0] == 0


def test_fit_line_with_xhat(fit_data: dict[str, NDArray[Any]]):
    x, y = fit_data["zscores"], fit_data["data"]
    x_, y_, res = numutils.fit_line(x, y, xhat=[-2, -1, 0, 1, 2])
    expected = [-0.566018, 4.774419, 10.114857, 15.455295, 20.795733]
    nptest.assert_array_almost_equal(y_, expected)


@pytest.mark.parametrize(
    ("oneway", "expected"), [(True, [False, False, True]), (False, [False, True, True])]
)
def test_checkIntervalOverlap_single(oneway, expected):
    result = [
        numutils.check_interval_overlap([1, 2], [3, 4], oneway=oneway),
        numutils.check_interval_overlap([1, 4], [2, 3], oneway=oneway),
        numutils.check_interval_overlap([1, 3], [2, 4], oneway=oneway),
    ]
    assert result == expected


@pytest.mark.parametrize(("oneway", "expected"), [(True, [0, 0, 1]), (False, [0, 1, 1])])
def test_checkIntervalOverlap(oneway, expected):
    x = numpy.array([[1, 2], [1, 4], [1, 3]])
    y = numpy.array([[3, 4], [2, 3], [2, 4]])
    nptest.assert_array_equal(
        numutils.check_interval_overlap(x, y, oneway=oneway, axis=1),
        numpy.array(expected, dtype=bool),
    )


@pytest.mark.parametrize(
    ("opts", "expected_key"),
    [
        (dict(), "no-op"),
        (dict(A=0.05), "one-col"),
        (dict(A=0.05, C=0.10), "two-col"),
        (dict(A=0.20, B=0.20, C=0.10), "three-col"),
        (dict(A=(0.05, 0.20), B=0.20, C=0.10), "tuple-limit"),
    ],
)
def test_winsorize_dataframe(opts, expected_key):
    x = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    w_05 = numpy.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19])

    w_10 = numpy.array([2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18])

    w_20 = numpy.array([4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16])

    w_05_20 = numpy.array(
        [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16]
    )

    df = pandas.DataFrame({"A": x, "B": x, "C": x})
    expected = {
        "no-op": df.copy(),
        "one-col": pandas.DataFrame({"A": w_05, "B": x, "C": x}),
        "two-col": pandas.DataFrame({"A": w_05, "B": x, "C": w_10}),
        "three-col": pandas.DataFrame({"A": w_20, "B": w_20, "C": w_10}),
        "tuple-limit": pandas.DataFrame({"A": w_05_20, "B": w_20, "C": w_10}),
    }

    result = numutils.winsorize_dataframe(df, **opts)
    assert result.eq(expected[expected_key]).all().all()


def test__comp_stat_generator():
    df = helpers.make_dc_data().reset_index()
    gen = numutils._comp_stat_generator(df, ["param", "bmp"], "loc", "res", helpers.comp_statfxn)
    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen)

    expected = {
        "bmp": ["1", "1", "1", "1", "1", "7", "7", "7", "7", "7"],
        "param": ["A", "A", "A", "A", "A", "H", "H", "H", "H", "H"],
        "loc_1": [
            "Inflow",
            "Inflow",
            "Outflow",
            "Outflow",
            "Reference",
            "Inflow",
            "Outflow",
            "Outflow",
            "Reference",
            "Reference",
        ],
        "loc_2": [
            "Outflow",
            "Reference",
            "Inflow",
            "Reference",
            "Inflow",
            "Reference",
            "Inflow",
            "Reference",
            "Inflow",
            "Outflow",
        ],
        "stat": [
            33.100128,
            34.228203,
            18.318497,
            21.231383,
            9.500766,
            2.990900,
            7.498225,
            7.757954,
            3.067299,
            3.290471,
        ],
        "pvalue": [
            8.275032,
            8.557050,
            4.579624,
            5.307845,
            2.375191,
            0.747725,
            1.874556,
            1.939488,
            0.766824,
            0.822617,
        ],
    }
    pdtest.assert_frame_equal(
        pandas.DataFrame(expected, index=[0, 1, 2, 3, 4, 331, 332, 333, 334, 335]).sort_index(
            axis="columns"
        ),
        pandas.concat([result.head(), result.tail()]).sort_index(axis="columns"),
    )


def test__comp_stat_generator_single_group_col():
    df = helpers.make_dc_data().reset_index()
    gen = numutils._comp_stat_generator(df, "param", "loc", "res", helpers.comp_statfxn)
    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen)

    expected = {
        "stat": [
            35.78880963,
            36.032519607,
            24.07328047,
            24.151276229,
            13.107839675,
            16.69316696,
            14.336154663,
            14.33615466,
            11.298920232,
            11.298920232,
        ],
        "loc_1": [
            "Inflow",
            "Inflow",
            "Outflow",
            "Outflow",
            "Reference",
            "Inflow",
            "Outflow",
            "Outflow",
            "Reference",
            "Reference",
        ],
        "loc_2": [
            "Outflow",
            "Reference",
            "Inflow",
            "Reference",
            "Inflow",
            "Reference",
            "Inflow",
            "Reference",
            "Inflow",
            "Outflow",
        ],
        "param": ["A", "A", "A", "A", "A", "H", "H", "H", "H", "H"],
        "pvalue": [
            8.94720240,
            9.008129901,
            6.018320119,
            6.037819057,
            3.276959918,
            4.17329174,
            3.584038665,
            3.584038665,
            2.824730058,
            2.824730058,
        ],
    }
    pdtest.assert_frame_equal(
        pandas.DataFrame(expected, index=[0, 1, 2, 3, 4, 43, 44, 45, 46, 47]).sort_index(
            axis="columns"
        ),
        pandas.concat([result.head(), result.tail()]).sort_index(axis="columns"),
    )


@pytest.mark.parametrize(
    ("control", "statfxn"),
    [(None, helpers.group_statfxn), ("Reference", helpers.group_statfxn_with_control)],
)
def test__group_comp_stat_generator(control, statfxn):
    df = helpers.make_dc_data_complex().reset_index().loc[lambda df: df["state"].isin(["CA", "OR"])]
    gen = numutils._group_comp_stat_generator(
        df, ["bmp", "state"], "loc", "res", statfxn, statname="max_of_max", control=control
    )

    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen).sort_values(by=["bmp", "state"]).reset_index(drop=True)

    expected_vals = [
        10.50949742,
        14.77713121,
        23.29366747,
        25.31733815,
        15.0478646,
        163.01000855,
        81.9363735,
        12.89370847,
        16.76919938,
        5.25222615,
        8.75794985,
        24.71916248,
        13.87664484,
        15.05442994,
    ]
    expected = (
        pandas.DataFrame(
            {
                "bmp": list(map(str, range(1, 8))) * 2,
                "state": (["CA"] * 7) + (["OR"] * 7),
                "max_of_max": expected_vals,
                "pvalue": [0.25] * 14,
            }
        )
        .sort_values(by=["bmp", "state"])
        .reset_index(drop=True)
    )
    pandas.testing.assert_frame_equal(result, expected, rtol=0.01)


def test__paired_stat_generator():
    df = helpers.make_dc_data_complex().unstack(level="loc")
    gen = numutils._paired_stat_generator(df, ["param"], "loc", "res", helpers.comp_statfxn)
    assert isinstance(gen, types.GeneratorType)
    result = pandas.DataFrame(gen).sort_index(axis="columns")

    expected = pandas.DataFrame(
        {
            "loc_1": [
                "Inflow",
                "Inflow",
                "Outflow",
                "Outflow",
                "Reference",
                "Inflow",
                "Outflow",
                "Outflow",
                "Reference",
                "Reference",
            ],
            "loc_2": [
                "Outflow",
                "Reference",
                "Inflow",
                "Reference",
                "Inflow",
                "Reference",
                "Inflow",
                "Reference",
                "Inflow",
                "Outflow",
            ],
            "param": ["A", "A", "A", "A", "A", "H", "H", "H", "H", "H"],
            "pvalue": [
                2.688485,
                3.406661,
                9.084853,
                9.084853,
                5.243408,
                9.399253,
                20.234093,
                20.23409,
                2.801076,
                2.801075,
            ],
            "stat": [
                10.753940,
                13.626645,
                36.339414,
                36.339414,
                20.973631,
                37.597010,
                80.936373,
                80.936373,
                11.204302,
                11.204302,
            ],
        },
        index=[0, 1, 2, 3, 4, 43, 44, 45, 46, 47],
    ).sort_index(axis="columns")

    pdtest.assert_frame_equal(expected, pandas.concat([result.head(), result.tail()]))


@helpers.seed
def test_remove_outliers():
    expected_shape = (35,)
    x = numpy.random.normal(0, 4, size=37)

    assert numutils.remove_outliers(x).shape == expected_shape


def test_tukey_hsd_functions():
    expected_records = [
        {
            "chemical_name": "Copper",
            "Loc_0": -2.0,
            "Loc_1": 6.0,
            "Loc_2": -2.0,
            "Loc_3": -4.0,
            "Loc_4": 3.0,
            "Loc_5": -1.0,
            "Loc_6": 0.0,
        },
        {
            "chemical_name": "Di(2-ethylhexyl)phthalate",
            "Loc_0": 3.0,
            "Loc_1": 5.0,
            "Loc_2": -2.0,
            "Loc_3": -2.0,
            "Loc_4": -2.0,
            "Loc_5": -1.0,
            "Loc_6": -1.0,
        },
        {
            "chemical_name": "Indeno(1,2,3-cd)pyrene",
            "Loc_0": 2.0,
            "Loc_1": 0.0,
            "Loc_2": 6.0,
            "Loc_3": -2.0,
            "Loc_4": -2.0,
            "Loc_5": -4.0,
            "Loc_6": 0.0,
        },
        {
            "chemical_name": "Lead",
            "Loc_0": 0.0,
            "Loc_1": 6.0,
            "Loc_2": -2.0,
            "Loc_3": -3.0,
            "Loc_4": 4.0,
            "Loc_5": -3.0,
            "Loc_6": -2.0,
        },
        {
            "chemical_name": "Phenanthrene",
            "Loc_0": 1.0,
            "Loc_1": 0.0,
            "Loc_2": 1.0,
            "Loc_3": -3.0,
            "Loc_4": 0.0,
            "Loc_5": 0.0,
            "Loc_6": 1.0,
        },
        {
            "chemical_name": "Pyrene",
            "Loc_0": 0.0,
            "Loc_1": -1.0,
            "Loc_2": 3.0,
            "Loc_3": -2.0,
            "Loc_4": -2.0,
            "Loc_5": -2.0,
            "Loc_6": 4.0,
        },
        {
            "chemical_name": "Total Suspended Solids",
            "Loc_0": -1.0,
            "Loc_1": -1.0,
            "Loc_2": -1.0,
            "Loc_3": -1.0,
            "Loc_4": -1.0,
            "Loc_5": -1.0,
            "Loc_6": 6.0,
        },
        {
            "chemical_name": "Zinc",
            "Loc_0": 0.0,
            "Loc_1": 1.0,
            "Loc_2": -1.0,
            "Loc_3": -6.0,
            "Loc_4": -1.0,
            "Loc_5": 4.0,
            "Loc_6": 3.0,
        },
    ]
    expected = pandas.DataFrame(expected_records).set_index("chemical_name")
    wq = pandas.read_pickle(helpers.test_data_path("wq.pkl"))
    hsd = numutils.tukey_hsd(wq, "res", "location", "chemical_name")
    result = numutils.process_tukey_hsd_scores(hsd, "location", "chemical_name")
    pandas.testing.assert_frame_equal(result, expected, check_names=False)


def test_rank_stats():
    copper = pandas.read_pickle(helpers.test_data_path("sed.pkl")).loc[
        lambda df: df["chemical_name"] == "Copper"
    ]
    result = numutils.rank_stats(copper, "res", "location")
    expected = pandas.DataFrame(
        [
            {"location": "Loc_0", "count": 8, "sum": 178.0, "mean": 22.25},
            {"location": "Loc_1", "count": 11, "sum": 459.5, "mean": 41.77272727272727},
            {"location": "Loc_2", "count": 10, "sum": 248.0, "mean": 24.8},
            {"location": "Loc_3", "count": 11, "sum": 78.0, "mean": 7.090909090909091},
            {"location": "Loc_4", "count": 9, "sum": 503.0, "mean": 55.888888888888886},
            {"location": "Loc_5", "count": 12, "sum": 424.5, "mean": 35.375},
        ]
    )
    pandas.testing.assert_frame_equal(result, expected)


def test_dunn_scores():
    copper = pandas.read_pickle(helpers.test_data_path("sed.pkl")).loc[
        lambda df: df["chemical_name"] == "Copper"
    ]
    dunnres = numutils.dunn_test(copper, "res", "location")
    expected = pandas.Series(
        data=[-1, 1, -1, -3, 3, 1], index=["Loc_0", "Loc_1", "Loc_2", "Loc_3", "Loc_4", "Loc_5"]
    )
    pandas.testing.assert_series_equal(dunnres.scores, expected, check_names=False)
