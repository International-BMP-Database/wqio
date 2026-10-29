import itertools
from collections.abc import Callable
from functools import cached_property, partial
from typing import Any, NoReturn

import numpy
import polars as pl
import polars.selectors as cs
from polars import col as c
from polars_ros import bootstrap, ros
from scipy import stats

from wqio import utils, validate
from wqio.utils import _pandas_boundary as boundary

_STAT = {"statistic": pl.Float64, "pvalue": pl.Float64}


def _dist_compare(x: Any, y: Any, stat_comp_func: Callable[..., Any], **test_opts: Any) -> Any:
    if (len(x) == len(y)) and numpy.equal(x, y).all():
        return (numpy.nan, numpy.nan)

    return stat_comp_func(x, y, **test_opts)


def _scipy_expr(
    fxn: Callable[..., Any], *, min_obs: int = 0, log: bool = False, **opts: Any
) -> pl.Expr:
    """Wrap a scipy-style test returning (statistic, pvalue) as an aggregation expression.

    Groups with fewer than ``min_obs`` observations give a null struct.
    """

    def _apply(s: pl.Series) -> pl.Series:
        x = s.to_numpy()
        if len(x) < min_obs:
            result = {"statistic": None, "pvalue": None}
        else:
            res = fxn(numpy.log(x) if log else x, **opts)
            result = {"statistic": float(res[0]), "pvalue": float(res[1])}
        return pl.Series([result], dtype=pl.Struct(_STAT))

    return c("__x").map_batches(_apply, return_dtype=pl.Struct(_STAT), returns_scalar=True)


class DataCollection:
    """Polars-based water quality comparison object.

    Results that the pandas version reported with ``(station, result)`` column
    MultiIndexes are returned here with one *struct* column per station, with
    the statistics as the struct's fields. Pairwise comparisons are tidy frames
    with ``<stationcol>_1`` and ``<stationcol>_2`` columns.

    Parameters
    ----------
    filterfxn : polars.Expr, optional
        Boolean expression evaluated against the raw data; rows where it is
        false are dropped from ``tidy`` and ``wide``. Use ``.over(...)`` to
        make group-level decisions, e.g. ``pl.len().over("station", "parameter") > 3``.
    pairedstations : list of str, optional
        Stations to include in paired tests. Defaults to every station.
    showpbar : bool
        Accepted for API compatibility; polars has no progress bar hook.
    """

    cencol: str = "__censored"

    def __init__(
        self,
        dataframe: pl.DataFrame,
        pairedstations: list[str] | None = None,
        rescol: str = "res",
        qualcol: str = "qual",
        stationcol: str = "station",
        paramcol: str = "parameter",
        ndval: str | list[str] = "ND",
        othergroups: list[str] | None = None,
        pairgroups: list[str] | None = None,
        useros: bool = True,
        filterfxn: pl.Expr | None = None,
        bsiter: int = 10000,
        showpbar: bool = True,
    ) -> None:
        self.raw_data = dataframe
        self._raw_rescol = rescol
        self.qualcol = qualcol
        self.stationcol = stationcol
        self.paramcol = paramcol
        self.ndval: list[str] = validate.at_least_empty_list(ndval)
        self.othergroups: list[str] = validate.at_least_empty_list(othergroups)
        self.pairgroups: list[str] = validate.at_least_empty_list(pairgroups)
        self.useros = useros
        self.filterfxn = filterfxn
        self.bsiter = bsiter
        self.showpbar = showpbar

        # column that stores ROS'd values, and the column that holds the "final" values
        self.roscol: str = "ros_" + rescol
        self.rescol: str = self.roscol if self.useros else rescol

        # columns to group by when doing general stats and comparisons
        self.groupcols = [self.stationcol, self.paramcol] + self.othergroups
        self.groupcols_comparison = [self.paramcol] + self.othergroups

        # columns to pivot by when doing paired stats
        self.pairgroups = self.pairgroups + [self.paramcol]

        self.tidy_columns = list(
            dict.fromkeys(self.groupcols + [self._raw_rescol, self.rescol, self.cencol])
        )

        # the "raw" data with the censorship column added
        self.data = dataframe.with_columns(c(self.qualcol).is_in(self.ndval).alias(self.cencol))
        self._kept = self.data if filterfxn is None else self.data.filter(filterfxn)

        self.pairedstations: list[str] = (
            list(pairedstations) if pairedstations is not None else self.stations
        )

    @cached_property
    def stations(self) -> list[str]:
        return sorted(self.data[self.stationcol].unique().to_list())

    @cached_property
    def tidy(self) -> pl.DataFrame:
        if self.useros:
            imputed = (
                ros.impute(c(self._raw_rescol), c(self.cencol))
                .over(self.groupcols)
                .fill_null(
                    pl.when(c(self.cencol))
                    .then(pl.lit(0.5) * c(self._raw_rescol))
                    .otherwise(c(self._raw_rescol))
                )
            )
        else:
            imputed = pl.lit(None, dtype=pl.Float64)

        return (
            self._kept.with_columns(imputed.alias(self.roscol))
            .select(self.tidy_columns)
            .sort(self.groupcols)
        )

    @cached_property
    def wide(self) -> pl.DataFrame:
        """One struct column (raw result, censorship) per station."""
        return (
            self._kept.with_columns(values=pl.struct(self._raw_rescol, self.cencol))
            .pivot(
                on=self.stationcol,
                index=self.pairgroups,
                values="values",
                sort_columns=True,
            )
            .sort(self.pairgroups)
        )

    @cached_property
    def paired(self) -> pl.DataFrame:
        """Rows of ``wide`` where every one of ``pairedstations`` has a result."""
        return self.wide.filter(pl.all_horizontal(cs.by_name(self.pairedstations).is_not_null()))

    @cached_property
    def comp_paired(self) -> pl.DataFrame:
        """Raw results, as a list per group, for each of ``pairedstations``."""
        return self.paired.group_by(self.groupcols_comparison, maintain_order=True).agg(
            cs.by_name(self.pairedstations).struct.field(self._raw_rescol).name.keep()
        )

    @cached_property
    def comp_unpaired(self) -> pl.DataFrame:
        """Final results, as a list per group, for every station."""
        return (
            self.tidy.group_by(self.groupcols, maintain_order=True)
            .agg(self.rescol)
            .pivot(
                on=self.stationcol,
                index=self.groupcols_comparison,
                values=self.rescol,
                sort_columns=True,
            )
            .sort(self.groupcols_comparison)
        )

    # single-sample statistics
    def _by_station(self, df: pl.DataFrame, groupcols: list[str] | None = None) -> pl.DataFrame:
        """Pivot a frame with a ``stats`` struct column into one struct column per station."""
        index = [g for g in (groupcols or self.groupcols) if g != self.stationcol]
        return df.pivot(on=self.stationcol, index=index, values="stats", sort_columns=True).sort(
            index
        )

    def generic_stat(
        self,
        statfxn: pl.Expr,
        statname: str | None = None,
        bootstrap_statistic: str | None = None,
        log: bool = False,
    ) -> pl.DataFrame:
        """Generic function to estimate a statistic and its CIs for every group.

        Parameters
        ----------
        statfxn : polars.Expr
            Aggregating expression written in terms of ``pl.col("__x")``, the
            (ROS'd, unless ``useros=False``) results of a group.
        statname : str, optional
            Field name for the statistic. Omit when ``statfxn`` already returns a struct.
        bootstrap_statistic : str, optional
            A statistic known to ``polars_ros.bootstrap.ci`` ("mean", "median", "geomean", ...).
            When given, the field ``lower`` and ``upper`` hold the BCA 95% CI.
        log : bool
            Bootstrap the log of the results.

        Returns
        -------
        pl.DataFrame
            ``groupcols_comparison`` plus one struct column per station.
        """
        aggs = [statfxn.alias("__stat")]
        if bootstrap_statistic:
            values = c("__x").log() if log else c("__x")
            aggs.append(
                bootstrap.ci(
                    values, statistic=bootstrap_statistic, niter=self.bsiter, method="bca"
                ).alias("__ci")
            )

        agg = self.tidy.rename({self.rescol: "__x"}).group_by(self.groupcols).agg(*aggs)

        name = statname or "stat"
        if bootstrap_statistic:
            stats_ = pl.struct(
                c("__ci").struct.field("lower"),
                c("__stat").alias(name),
                c("__ci").struct.field("upper"),
            )
        elif statname:
            stats_ = pl.struct(c("__stat").alias(name))
        else:
            stats_ = c("__stat")

        return self._by_station(agg.select(*self.groupcols, stats_.alias("stats")))

    @cached_property
    def count(self) -> pl.DataFrame:
        return self.generic_stat(pl.len(), "count")

    @cached_property
    def inventory(self) -> pl.DataFrame:
        return (
            self.tidy.group_by(self.groupcols)
            .agg(count=pl.len(), non_detect=c(self.cencol).sum())
            .sort(self.groupcols)
        )

    @cached_property
    def median(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").median(), "median", "median")

    @cached_property
    def mean(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").mean(), "mean", "mean")

    @cached_property
    def std_dev(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").std(ddof=1), "std_dev")

    def percentile(self, percentile: float) -> pl.DataFrame:
        """Return the percentiles (0 - 100) for the data."""
        return self.generic_stat(
            c("__x").quantile(percentile / 100, interpolation="linear"), f"pctl_{percentile:g}"
        )

    @cached_property
    def logmean(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").log().mean(), "logmean", "mean", log=True)

    @cached_property
    def logstd_dev(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").log().std(ddof=1), "logstd_dev")

    @cached_property
    def geomean(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").log().mean().exp(), "geomean", "geomean")

    @cached_property
    def geostd_dev(self) -> pl.DataFrame:
        return self.generic_stat(c("__x").log().std(ddof=1).exp(), "geostd_dev")

    def shapiro(self, **opts: Any) -> pl.DataFrame:
        """Shapiro-Wilk test for normality; see `scipy.stats.shapiro`."""
        return self.generic_stat(_scipy_expr(stats.shapiro, min_obs=4, **opts))

    def shapiro_log(self, **opts: Any) -> pl.DataFrame:
        """Shapiro-Wilk test for normality of the logs; see `scipy.stats.shapiro`."""
        return self.generic_stat(_scipy_expr(stats.shapiro, min_obs=4, log=True, **opts))

    def lilliefors(self, **opts: Any) -> pl.DataFrame:
        """Lilliefors test for normality; see `statsmodels.stats.diagnostic.lilliefors`."""
        from statsmodels.stats.diagnostic import lilliefors

        return self.generic_stat(_scipy_expr(lilliefors, **opts))

    def lilliefors_log(self, **opts: Any) -> pl.DataFrame:
        """Lilliefors test for normality of the logs."""
        from statsmodels.stats.diagnostic import lilliefors

        return self.generic_stat(_scipy_expr(lilliefors, log=True, **opts))

    def anderson_darling(self, **opts: Any) -> NoReturn:
        raise NotImplementedError

    def anderson_darling_log(self, **opts: Any) -> NoReturn:
        raise NotImplementedError

    def theilslopes(self, logs: bool = False) -> NoReturn:
        raise NotImplementedError

    # comparisons
    def comparison_stat_twoway(
        self,
        statfxn: Callable[..., Any],
        statname: str = "stat",
        paired: bool = False,
        **statopts: Any,
    ) -> pl.DataFrame:
        """Apply a two-sample test to every ordered pair of stations.

        Each pair is tested on the lists of results of its two stations: the ROS'd
        results of ``comp_unpaired``, or (``paired=True``) the raw results of the rows
        of ``wide`` where both stations of the pair have a result.

        Returns
        -------
        pl.DataFrame
            ``groupcols_comparison``, ``<stationcol>_1``, ``<stationcol>_2``,
            ``statname`` and ``pvalue``.
        """
        if paired:
            stations = self.pairedstations
            samples = self._paired_samples
        else:
            stations = [
                c_ for c_ in self.comp_unpaired.columns if c_ not in self.groupcols_comparison
            ]
            samples = self._unpaired_samples

        names = [f"{self.stationcol}_1", f"{self.stationcol}_2"]
        dtype = pl.Struct({statname: pl.Float64, "pvalue": pl.Float64})

        def _apply(row: dict[str, Any]) -> dict[str, float]:
            res = statfxn(numpy.asarray(row["x"]), numpy.asarray(row["y"]), **statopts)
            return {statname: float(res[0]), "pvalue": float(res[1])}

        frames = [
            samples(a, b).select(
                *self.groupcols_comparison,
                pl.lit(a).alias(names[0]),
                pl.lit(b).alias(names[1]),
                pl.struct("x", "y").map_elements(_apply, return_dtype=dtype).alias("__result"),
            )
            for a, b in itertools.permutations(stations, 2)
        ]
        return pl.concat(frames).unnest("__result").sort(self.groupcols_comparison + names)

    def _unpaired_samples(self, a: str, b: str) -> pl.DataFrame:
        return self.comp_unpaired.filter(c(a).is_not_null() & c(b).is_not_null()).select(
            *self.groupcols_comparison, x=c(a), y=c(b)
        )

    def _paired_samples(self, a: str, b: str) -> pl.DataFrame:
        # complete cases for just this pair of stations
        return (
            self.wide.filter(c(a).is_not_null() & c(b).is_not_null())
            .group_by(self.groupcols_comparison, maintain_order=True)
            .agg(
                x=c(a).struct.field(self._raw_rescol),
                y=c(b).struct.field(self._raw_rescol),
            )
        )

    def comparison_stat_allway(
        self,
        statfxn: Callable[..., Any],
        statname: str,
        control: str | None = None,
        **statopts: Any,
    ) -> pl.DataFrame:
        """Apply a k-sample test to all of the stations (or against ``control``)."""
        comp = self.comp_unpaired
        stations = [col for col in comp.columns if col not in self.groupcols_comparison]
        dtype = pl.Struct({statname: pl.Float64, "pvalue": pl.Float64})

        def _apply(row: dict[str, Any]) -> dict[str, float]:
            samples = {k: numpy.asarray(v) for k, v in row.items() if v is not None}
            opts = dict(statopts)
            if control is not None:
                opts["control"] = samples.pop(control)
            res = statfxn(*samples.values(), **opts)
            return {statname: float(res.statistic), "pvalue": float(res.pvalue)}

        return (
            comp.select(
                *self.groupcols_comparison,
                pl.struct(stations).map_elements(_apply, return_dtype=dtype).alias("__result"),
            )
            .unnest("__result")
            .sort(self.groupcols_comparison)
        )

    def mann_whitney(self, **opts: Any) -> pl.DataFrame:
        """Mann-Whitney U test across stations; see `scipy.stats.mannwhitneyu`."""
        return self.comparison_stat_twoway(
            partial(_dist_compare, stat_comp_func=stats.mannwhitneyu, **opts),
            statname="mann_whitney",
        )

    def ranksums(self, **opts: Any) -> pl.DataFrame:
        """Unpaired Wilcoxon rank-sum test; see `scipy.stats.ranksums`."""
        return self.comparison_stat_twoway(stats.ranksums, "rank_sums", **opts)

    def t_test(self, **opts: Any) -> pl.DataFrame:
        """T-test for independent samples; see `scipy.stats.ttest_ind`."""
        return self.comparison_stat_twoway(stats.ttest_ind, "t_test", **opts)

    def levene(self, **opts: Any) -> pl.DataFrame:
        """Levene test for equal variances; see `scipy.stats.levene`."""
        return self.comparison_stat_twoway(stats.levene, "levene", **opts)

    def wilcoxon(self, **opts: Any) -> pl.DataFrame:
        """Paired Wilcoxon signed-rank test; see `scipy.stats.wilcoxon`."""
        return self.comparison_stat_twoway(
            partial(_dist_compare, stat_comp_func=stats.wilcoxon), "wilcoxon", paired=True, **opts
        )

    def kendall(self, **opts: Any) -> pl.DataFrame:
        """Paired Kendall-tau test; see `scipy.stats.kendalltau`."""
        return self.comparison_stat_twoway(stats.kendalltau, "kendalltau", paired=True, **opts)

    def spearman(self, **opts: Any) -> pl.DataFrame:
        """Paired Spearman-rho test; see `scipy.stats.spearmanr`."""
        return self.comparison_stat_twoway(stats.spearmanr, "spearmanrho", paired=True, **opts)

    def kruskal_wallis(self, **opts: Any) -> pl.DataFrame:
        """Kruskal-Wallis H-test across stations; see `scipy.stats.kruskal`."""
        return self.comparison_stat_allway(stats.kruskal, "kruskal_wallis", **opts)

    def f_test(self, **opts: Any) -> pl.DataFrame:
        """One-way ANOVA across stations; see `scipy.stats.f_oneway`."""
        return self.comparison_stat_allway(stats.f_oneway, "f_test", **opts)

    # The tests below are implemented with pandas; they cross the boundary here.
    def tukey_hsd(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Tukey HSD across stations + other groups for each parameter."""
        tidy = boundary.to_pandas(self.tidy)
        hsd = utils.tukey_hsd(tidy, self.rescol, self.stationcol, self.paramcol, *self.othergroups)
        scores = utils.process_tukey_hsd_scores(hsd, self.stationcol, self.paramcol)
        return boundary.from_pandas(hsd), boundary.from_pandas(scores)

    def dunn(self) -> pl.DataFrame:
        """Dunn test across the different stations for each parameter."""
        tidy = boundary.to_pandas(self.tidy)
        scores = tidy.groupby(by=[self.paramcol]).apply(
            lambda g: utils.dunn_test(g, self.rescol, self.stationcol, *self.othergroups).scores
        )
        return boundary.from_pandas(scores)

    # summaries
    def n_unique(self, column: str) -> pl.DataFrame:
        """Number of unique values of ``column`` for each group, one struct per station."""
        agg = self.data.group_by(self.groupcols).agg(pl.struct(c(column).n_unique()).alias("stats"))
        return self._by_station(agg)

    def stat_summary(
        self,
        percentiles: list[float] | None = None,
        groupcols: list[str] | None = None,
    ) -> pl.DataFrame:
        """A high-level summary of the data, one struct column per station.

        Parameters
        ----------
        percentiles : list of float, optional
            Fractions in (0, 1). The median is always included.
        groupcols : list of str, optional
            Columns to group ``tidy`` by. Must include ``stationcol``.
        """
        groupcols = validate.at_least_empty_list(groupcols) or self.groupcols
        ptiles = sorted({*(percentiles or [0.1, 0.25, 0.5, 0.75, 0.9]), 0.5})
        x = c(self.rescol)
        summary = [
            pl.len().alias("count"),
            c(self.cencol).sum().alias("non_detect"),
            x.mean().alias("mean"),
            x.std(ddof=1).alias("std"),
            x.min().alias("min"),
            *(x.quantile(p, interpolation="linear").alias(f"pctl_{p * 100:g}") for p in ptiles),
            x.max().alias("max"),
        ]
        agg = self.tidy.group_by(groupcols).agg(pl.struct(summary).alias("stats"))
        return self._by_station(agg, groupcols)
