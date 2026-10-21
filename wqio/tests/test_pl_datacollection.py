import polars as pl
import pytest
from polars import col as c

from wqio.pl.datacollection import DataCollection
from wqio.tests import helpers
from wqio.tests.parity import assert_matches_golden, flatten

STATIONS = ["Inflow", "Outflow", "Reference"]
COUNT_NAMES = {"Count": "count", "Non-Detect": "non_detect"}


@pytest.fixture(scope="module")
def dc() -> DataCollection:
    return DataCollection(
        helpers.read_golden("input_dc_data_complex"),
        rescol="res",
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval="<",
        pairgroups=["state", "bmp"],
        filterfxn=~c("param").is_in(["G", "H"]),
        bsiter=2000,
        showpbar=False,
    )


def rename_fields(**mapping: str):
    """Rename the field part (after the last ``__``) of golden columns."""

    def renamer(col: str) -> str:
        head, sep, tail = col.rpartition("__")
        return f"{head}{sep}{mapping.get(tail, tail)}"

    return renamer


def test_tidy(dc):
    assert_matches_golden(dc.tidy, "dc_tidy", sort_by=["loc", "param", "ros_res"])


def test_tidy_without_nds():
    dc = DataCollection(
        helpers.read_golden("input_dc_data_complex"),
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval="junk",
        filterfxn=~c("param").is_in(["G", "H"]),
    )
    assert_matches_golden(dc.tidy, "dc_noNDs_tidy", sort_by=["loc", "param", "res"])
    assert_matches_golden(
        dc.inventory, "dc_noNDs_inventory", rename=lambda col: COUNT_NAMES.get(col, col)
    )


def test_wide(dc):
    assert_matches_golden(dc.wide, "dc_paired", swap=True)


def test_paired_has_all_stations(dc):
    assert dc.paired.height > 0
    assert dc.paired.select(pl.all_horizontal(c(STATIONS).is_not_null()).all()).item()


def test_inventory(dc):
    assert_matches_golden(
        dc.inventory, "dc_inventory", rename=lambda col: COUNT_NAMES.get(col, col)
    )


def test_count(dc):
    assert_matches_golden(dc.count, "dc_count", rename=rename_fields(Count="count"))


def test_n_unique(dc):
    assert_matches_golden(dc.n_unique("bmp"), "dc_n_unique_bmp")


def test_stat_summary(dc):
    def renamer(col: str) -> str:
        if "__" not in col:
            return col
        stat, _, station = col.partition("__")
        stat = COUNT_NAMES.get(stat, stat)
        if stat.endswith("%"):
            stat = f"pctl_{stat[:-1]}"
        return f"{station}__{stat}"

    assert_matches_golden(dc.stat_summary(), "dc_stat_summary", rename=renamer)


@pytest.mark.parametrize(
    ("prop", "golden", "fields"),
    [
        ("std_dev", "dc_std_dev", {"std. dev.": "std_dev"}),
        ("logstd_dev", "dc_logstd_dev", {"Log-std. dev.": "logstd_dev"}),
        ("geostd_dev", "dc_geostd_dev", {"Log-std. dev.": "geostd_dev"}),
    ],
)
def test_deterministic_stats(dc, prop, golden, fields):
    assert_matches_golden(getattr(dc, prop), golden, rename=rename_fields(**fields))


def test_percentile(dc):
    assert_matches_golden(
        dc.percentile(25), "dc_percentile_25", rename=rename_fields(**{"pctl 25": "pctl_25"})
    )


@pytest.mark.parametrize(
    ("prop", "golden", "stat", "goldenstat"),
    [
        ("median", "dc_median", "median", "median"),
        ("mean", "dc_mean", "mean", "mean"),
        ("logmean", "dc_logmean", "logmean", "Log-mean"),
        ("geomean", "dc_geomean", "geomean", "Log-mean"),
    ],
)
def test_bootstrapped_stats(dc, prop, golden, stat, goldenstat):
    result = flatten(getattr(dc, prop)).sort("param")
    expected = helpers.read_golden(golden).sort("param")
    assert result["param"].to_list() == expected["param"].to_list()

    for station in STATIONS:
        point = result[f"{station}__{stat}"]
        assert point.to_numpy() == pytest.approx(
            expected[f"{station}__{goldenstat}"].to_numpy(), rel=1e-9
        )
        # the CI is stochastic and uses a different RNG than the goldens
        width = expected[f"{station}__upper"] - expected[f"{station}__lower"]
        for side in ("lower", "upper"):
            diff = (result[f"{station}__{side}"] - expected[f"{station}__{side}"]).abs()
            assert (diff <= width).all()
        assert (result[f"{station}__lower"] <= point).all()
        assert (point <= result[f"{station}__upper"]).all()


@pytest.mark.parametrize(
    ("method", "goldenstat"),
    [
        ("shapiro", "statistic"),
        ("shapiro_log", "statistic"),
        ("lilliefors", "lilliefors"),
        ("lilliefors_log", "log-lilliefors"),
    ],
)
def test_normality_tests(dc, method, goldenstat):
    assert_matches_golden(
        getattr(dc, method)(), f"dc_{method}", rename=rename_fields(**{goldenstat: "statistic"})
    )


@pytest.mark.parametrize(
    "method",
    ["mann_whitney", "ranksums", "t_test", "levene", "wilcoxon", "kendall", "spearman"],
)
def test_twoway(dc, method):
    assert_matches_golden(
        getattr(dc, method)(), f"dc_{method}", sort_by=["param", "loc_1", "loc_2"]
    )


@pytest.mark.parametrize(
    ("method", "name", "golden_name"),
    [("kruskal_wallis", "kruskal_wallis", "K-W H"), ("f_test", "f_test", "f-test")],
)
def test_allway(dc, method, name, golden_name):
    assert_matches_golden(
        getattr(dc, method)(), f"dc_{method}", rename={golden_name: name}, sort_by=["param"]
    )


def test_tukey_hsd(dc):
    hsd, scores = dc.tukey_hsd()
    assert_matches_golden(hsd, "dc_tukey_hsd", sort_by=["param", "loc 1", "loc 2"])
    assert_matches_golden(scores, "dc_tukey_scores", sort_by=["param"])


def test_dunn(dc):
    assert_matches_golden(dc.dunn(), "dc_dunn", sort_by=["param"])


def test_pairedstations_default():
    dc = DataCollection(
        helpers.read_golden("input_dc_data_complex"),
        stationcol="loc",
        paramcol="param",
        ndval="<",
        pairgroups=["state", "bmp"],
    )
    assert dc.pairedstations == STATIONS
