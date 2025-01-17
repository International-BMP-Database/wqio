import warnings
from collections import namedtuple
from functools import partial

import numpy
import pandas
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.decorators import cache_readonly

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from wqio import bootstrap, utils, validate
from wqio.features import Dataset, Location
from wqio.ros import ROS

_Stat = namedtuple("_stat", ["stat", "pvalue"])


def _dist_compare(x, y, stat_comp_func, **test_opts):
    if (len(x) == len(y)) and numpy.equal(x, y).all():
        return _Stat(numpy.nan, numpy.nan)

    return stat_comp_func(x, y, **test_opts)


class DataCollection:
    """Generalized water quality comparison object.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe all of the data to analyze.
    rescol, qualcol, stationcol, paramcol : string
        Column labels for the results, qualifiers, stations (monitoring
        locations), and parameters (pollutants), respectively.

        .. note::

           Non-detect results should be reported as the detection
           limit of that observation.

    ndval : string or list of strings, options
        The values found in ``qualcol`` that indicates that a
        result is a non-detect.
    othergroups : list of strings, optional
        The columns (besides ``stationcol`` and ``paramcol``) that
        should be considered when grouping into subsets of data.
    pairgroups : list of strings, optional
        Other columns besides ``stationcol`` and ``paramcol`` that
        can be used define a unique index on ``dataframe`` such that it
        can be "unstack" (i.e., pivoted, cross-tabbed) to place the
        ``stationcol`` values into columns. Values of ``pairgroups``
        may overlap with ``othergroups``.
    useros : bool (default = True)
        Toggles the use of regression-on-order statistics to estimate
        non-detect values when computing statistics.
    filterfxn : callable, optional
        Function that will be passed to the ``filter`` method of a
        ``pandas.Groupby`` object to remove groups that should not be
        analyzed (for whatever reason). If not provided, all groups
        returned by ``dataframe.groupby(by=groupcols)`` will be used.
    bsiter : int
        Number of iterations the bootstrapper should use when estimating
        confidence intervals around a statistic.
    showpbar : bool (True)
        When True and the `tqdm` module is available, this will toggle the
        appears of progress bars in long-running group by-apply operations.

    """

    # column that stores the censorsip status of an observation
    cencol = "__censorship"

    def __init__(
        self,
        dataframe,
        rescol="res",
        qualcol="qual",
        stationcol="station",
        paramcol="parameter",
        ndval="ND",
        othergroups=None,
        pairgroups=None,
        useros=True,
        filterfxn=None,
        bsiter=10000,
        showpbar=True,
    ):
        # cache for all of the properties
        self._cache = {}

        # basic input
        self.raw_data = dataframe
        self._raw_rescol = rescol
        self.qualcol = qualcol
        self.stationcol = stationcol
        self.paramcol = paramcol
        self.ndval = validate.at_least_empty_list(ndval)
        self.othergroups = validate.at_least_empty_list(othergroups)
        self.pairgroups = validate.at_least_empty_list(pairgroups)
        self.useros = useros
        self.filterfxn = filterfxn or utils.non_filter
        self.bsiter = bsiter
        self.showpbar = showpbar

        # column that stores ROS'd values
        self.roscol = "ros_" + rescol

        # column stators "final" values
        if self.useros:
            self.rescol = self.roscol
        else:
            self.rescol = rescol

        # columns to group by when ROS'd, doing general stats
        self.groupcols = [self.stationcol, self.paramcol] + self.othergroups
        self.groupcols_comparison = [self.paramcol] + self.othergroups

        # columns to group and pivot by when doing paired stats
        self.pairgroups = self.pairgroups + [self.stationcol, self.paramcol]

        # final column list of the tidy dataframe
        self.tidy_columns = self.groupcols + list({self._raw_rescol, self.rescol, self.cencol})

        # the "raw" data with the censorship column added
        self.data = dataframe.assign(
            **{self.cencol: dataframe[self.qualcol].isin(self.ndval)}
        ).reset_index()

        self.pbarfxn = tqdm if (self.showpbar and tqdm) else utils.misc.no_op

    @cache_readonly
    def tidy(self):
        if self.useros:

            def fxn(g):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rosdf = (
                        ROS(
                            df=g,
                            result=self._raw_rescol,
                            censorship=self.cencol,
                            as_array=False,
                        )
                        .rename(columns={"final": self.roscol})
                        .loc[:, [self._raw_rescol, self.roscol, self.cencol]]
                    )
                return rosdf

        else:

            def fxn(g):
                g[self.roscol] = numpy.nan
                return g

        if tqdm and self.showpbar:

            def make_tidy(df):
                tqdm.pandas(desc="Tidying the DataCollection")
                return df.groupby(self.groupcols).progress_apply(fxn, include_groups=False)

        else:

            def make_tidy(df):
                return df.groupby(self.groupcols).apply(fxn, include_groups=False)

        # keep_cols = self.tidy_columns
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            _tidy = (
                self.data.reset_index()  # [self.tidy_columns]
                .groupby(by=self.groupcols)
                .filter(self.filterfxn)
                .pipe(make_tidy)
                .reset_index()
                .sort_values(by=self.groupcols)
            )

        return _tidy[self.tidy_columns]

    @cache_readonly
    def paired(self):
        _pairs = (
            self.data.reset_index()
            .groupby(by=self.groupcols)
            .filter(self.filterfxn)
            .set_index(self.pairgroups)
            .unstack(level=self.stationcol)
            .rename_axis(["value", self.stationcol], axis="columns")
        )[[self._raw_rescol, self.cencol]]
        return _pairs

    def generic_stat(
        self,
        statfxn,
        use_bootstrap=True,
        statname=None,
        has_pvalue=False,
        filterfxn=None,
        **statopts,
    ):
        """Generic function to estimate a statistic and its CIs.

        Parameters
        ----------
        statfxn : callable
            A function that takes a 1-D sequnce and returns a scalar
            results. Its call signature should be in the form:
            ``statfxn(seq, **kwargs)``.
        use_bootstrap : bool, optional
            Toggles using a BCA bootstrapping method to estimate the
            95% confidence interval around the statistic.
        statname : string, optional
            Name of the statistic. Included as a column name in the
            final dataframe.
        has_pvalue : bool, optional
            Set to ``True`` if ``statfxn`` returns a tuple of the
            statistic and it's p-value.
        **statopts : optional kwargs
            Additional keyword arguments that will be passed to
            ``statfxn``.

        Returns
        -------
        stat_df : pandas.DataFrame
            A dataframe all the results of the ``statfxn`` when applied
            to ``self.tidy.groupby(self.groupcols)``.

        Examples
        --------
        This actually demonstrates how ``DataCollection.mean`` is
        implemented.

        >>> import numpy
        >>> import wqio
        >>> from wqio.tests import helpers
        >>> df = helpers.make_dc_data_complex()
        >>> dc = DataCollection(df, rescol='res', qualcol='qual',
        ...                     stationcol='loc', paramcol='param',
        ...                     ndval='<')
        >>> means = dc.generic_stat(numpy.mean, statname='Arith. Mean')

        You can also use ``lambda`` objects

        >>> pctl35 = dc.generic_stat(lambda x: numpy.percentile(x, 35),
        ...                          statname='pctl35', use_bootstrap=False)

        """

        if statname is None:
            statname = "stat"

        if filterfxn is None:
            filterfxn = utils.non_filter

        def fxn(x):
            data = x[self.rescol].values
            if use_bootstrap:
                stat = statfxn(data)
                lci, uci = bootstrap.BCA(data, statfxn=statfxn)
                values = [lci, stat, uci]
                statnames = ["lower", statname, "upper"]
            else:
                values = validate.at_least_empty_list(statfxn(data, **statopts))
                if hasattr(values, "_fields"):  # nametuple
                    statnames = values._fields
                else:  # tuple
                    statnames = [statname]
                    if has_pvalue:
                        statnames.append("pvalue")

            return pandas.Series(values, index=statnames)

        groups = self.tidy.groupby(by=self.groupcols).filter(filterfxn).groupby(by=self.groupcols)

        if tqdm and self.showpbar:
            tqdm.pandas(desc="Computing stats")
            vals = groups.progress_apply(fxn, include_groups=False)
        else:
            vals = groups.apply(fxn, include_groups=False)

        results = (
            vals.unstack(level=self.stationcol)
            .pipe(utils.swap_column_levels, 0, 1)
            .rename_axis(["station", "result"], axis="columns")
        )
        return results

    @cache_readonly
    def count(self):
        return (
            self.generic_stat(lambda x: x.shape[0], use_bootstrap=False, statname="Count")
            .fillna(0)
            .astype(int)
        )

    @cache_readonly
    def inventory(self):
        counts = (
            self.tidy.groupby(by=self.groupcols + [self.cencol])
            .size()
            .unstack(level=self.cencol)
            .fillna(0)
            .astype(int)
            .rename_axis(None, axis="columns")
            .rename(columns={False: "Detect", True: "Non-Detect"})
            .assign(Count=lambda df: df.sum(axis="columns"))
        )
        if "Non-Detect" not in counts.columns:
            counts["Non-Detect"] = 0

        return counts[["Count", "Non-Detect"]]

    @cache_readonly
    def median(self):
        return self.generic_stat(numpy.median, statname="median")

    @cache_readonly
    def mean(self):
        return self.generic_stat(numpy.mean, statname="mean")

    @cache_readonly
    def std_dev(self):
        return self.generic_stat(numpy.std, statname="std. dev.", use_bootstrap=False, ddof=1)

    def percentile(self, percentile):
        """Return the percentiles (0 - 100) for the data."""
        return self.generic_stat(
            lambda x: numpy.percentile(x, percentile),
            statname=f"pctl {percentile}",
            use_bootstrap=False,
        )

    @cache_readonly
    def logmean(self):
        return self.generic_stat(
            lambda x, axis=0: numpy.mean(numpy.log(x), axis=axis), statname="Log-mean"
        )

    @cache_readonly
    def logstd_dev(self):
        return self.generic_stat(
            lambda x, axis=0: numpy.std(numpy.log(x), axis=axis, ddof=1),
            use_bootstrap=False,
            statname="Log-std. dev.",
        )

    @cache_readonly
    def geomean(self):
        geomean = numpy.exp(self.logmean)
        geomean.columns.names = ["station", "Geo-mean"]
        return geomean

    @cache_readonly
    def geostd_dev(self):
        geostd = numpy.exp(self.logstd_dev)
        geostd.columns.names = ["station", "Geo-std. dev."]
        return geostd

    def shapiro(self, **opts):
        """
        Run the Shapiro-Wilk test for normality on the datasets.

        Requires at least 3 observations in each dataset.

        See `scipy.stats.shapiro` for info on kwargs you can pass.
        """
        return self.generic_stat(
            stats.shapiro,
            use_bootstrap=False,
            has_pvalue=True,
            statname="shapiro",
            filterfxn=lambda x: x.shape[0] > 3,
            **opts,
        )

    def shapiro_log(self, **opts):
        """
        Run the Shapiro-Wilk test for normality on log-transformed datasets.

        Requires at least 3 observations in each dataset.

        See `scipy.stats.shapiro` for info on kwargs you can pass.
        """
        return self.generic_stat(
            lambda x: stats.shapiro(numpy.log(x)),
            use_bootstrap=False,
            has_pvalue=True,
            filterfxn=lambda x: x.shape[0] > 3,
            statname="log-shapiro",
            **opts,
        )

    def lilliefors(self, **opts):
        """
        Run the Lilliefors test for normality on the datasets.

        Requires at least 3 observations in each dataset.

        See `statsmodels.api.stats.lilliefors` for info on kwargs you can pass.
        """
        return self.generic_stat(
            sm.stats.lilliefors,
            use_bootstrap=False,
            has_pvalue=True,
            statname="lilliefors",
            **opts,
        )

    def lilliefors_log(self, **opts):
        """
        Run the Lilliefors test for normality on the log-transformed datasets.

        Requires at least 3 observations in each dataset.

        See `statsmodels.api.stats.lilliefors` for info on kwargs you can pass.
        """
        return self.generic_stat(
            lambda x: sm.stats.lilliefors(numpy.log(x)),
            use_bootstrap=False,
            has_pvalue=True,
            statname="log-lilliefors",
            **opts,
        )

    def anderson_darling(self, **opts):
        raise NotImplementedError
        return self.generic_stat(
            utils.anderson_darling,
            use_bootstrap=False,
            has_pvalue=True,
            statname="anderson-darling",
            **opts,
        )

    def anderson_darling_log(self, **opts):
        raise NotImplementedError
        return self.generic_stat(
            lambda x: utils.anderson_darling(numpy.log(x)),
            use_bootstrap=False,
            has_pvalue=True,
            statname="log-anderson-darling",
            **opts,
        )

    def comparison_stat_twoway(self, statfxn, statname=None, paired=False, **statopts):
        """Generic function to apply comparative hypothesis tests to
        the groups of the ``DataCollection``.

        Parameters
        ----------
        statfxn : callable
            A function that takes a 1-D sequnce and returns a scalar
            results. Its call signature should be in the form:
            ``statfxn(seq, **kwargs)``.
        statname : string, optional
            Name of the statistic. Included as a column name in the
            final dataframe.
        paired : bool, optional
            Set to ``True`` if ``statfxn`` requires paired data.
        **statopts : optional kwargs
            Additional keyword arguments that will be passed to
            ``statfxn``.

        Returns
        -------
        stat_df : pandas.DataFrame
            A dataframe all the results of the ``statfxn`` when applied
            to ``self.tidy.groupby(self.groupcols)`` or
            ``self.paired.groupby(self.groupcols)`` when necessary.

        Examples
        --------
        This actually demonstrates how ``DataCollection.mann_whitney``
        is implemented.

        >>> from scipy import stats
        >>> import wqio
        >>> from wqio.tests import helpers
        >>> df = helpers.make_dc_data_complex()
        >>> dc = DataCollection(df, rescol='res', qualcol='qual',
        ...                     stationcol='loc', paramcol='param',
        ...                     ndval='<')
        >>> mwht = dc.comparison_stat_twoway(stats.mannwhitneyu,
        ...                                  statname='mann_whitney',
        ...                                  alternative='two-sided')

        """

        if paired:
            data = self.paired
            generator = utils.numutils._paired_stat_generator
            rescol = self._raw_rescol
        else:
            data = self.tidy
            generator = utils.numutils._comp_stat_generator
            rescol = self.rescol

        station_columns = [self.stationcol + "_1", self.stationcol + "_2"]
        meta_columns = self.groupcols_comparison
        index_cols = meta_columns + station_columns

        results = generator(
            data,
            meta_columns,
            self.stationcol,
            rescol,
            statfxn,
            statname=statname,
            pbarfxn=self.pbarfxn,
            **statopts,
        )
        return pandas.DataFrame.from_records(results).set_index(index_cols)

    def comparison_stat_allway(self, statfxn, statname, control=None, **statopts):
        results = utils.numutils._group_comp_stat_generator(
            self.tidy,
            self.groupcols_comparison,
            self.stationcol,
            self.rescol,
            statfxn,
            statname=statname,
            control=control,
            pbarfxn=self.pbarfxn,
            **statopts,
        )
        return pandas.DataFrame.from_records(results).set_index(self.groupcols_comparison)

    def mann_whitney(self, **opts):
        """
        Run the Mann-Whitney U test across datasets.

        See `scipy.stats.mannwhitneyu` for available options.
        """
        return self.comparison_stat_twoway(
            partial(_dist_compare, stat_comp_func=stats.mannwhitneyu, **opts),
            statname="mann_whitney",
        )

    def ranksums(self, **opts):
        """
        Run the unpaired Wilcoxon rank-sum test across datasets.

        See `scipy.stats.ranksums` for available options.
        """
        return self.comparison_stat_twoway(stats.ranksums, statname="rank_sums", **opts)

    def t_test(self, **opts):
        """
        Run the T-test for independent scores.

        See `scipy.stats.ttest_ind` for available options.
        """
        return self.comparison_stat_twoway(stats.ttest_ind, statname="t_test", **opts)

    def levene(self, **opts):
        """
        Run the Levene test for equal variances

        See `scipy.stats.levene` for available options.
        """
        return self.comparison_stat_twoway(stats.levene, statname="levene", **opts)

    def wilcoxon(self, **opts):
        """
        Run the paired Wilcoxon rank-sum test across paired dataset.

        See `scipy.stats.wilcoxon` for available options.
        """
        return self.comparison_stat_twoway(
            partial(_dist_compare, stat_comp_func=stats.wilcoxon),
            statname="wilcoxon",
            paired=True,
            **opts,
        )

    def kendall(self, **opts):
        """
        Run the paired Kendall-tau test across paired dataset.

        See `scipy.stats.kendalltau` for available options.
        """
        return self.comparison_stat_twoway(
            stats.kendalltau, statname="kendalltau", paired=True, **opts
        )

    def spearman(self, **opts):
        """
        Run the paired Spearman-rho test across paired dataset.

        See `scipy.stats.spearmanr` for available options.
        """
        return self.comparison_stat_twoway(
            stats.spearmanr, statname="spearmanrho", paired=True, **opts
        )

    def kruskal_wallis(self, **opts):
        """
        Run the paired Kruskal-Wallos H-test across paired dataset.

        See `scipy.stats.kruskal` for available options.
        """
        return self.comparison_stat_allway(stats.kruskal, statname="K-W H", control=None, **opts)

    def f_test(self, **opts):
        """
        One-way ANOVA test across datasets

        See `scipy.stats.f_oneway` for available options.
        """
        return self.comparison_stat_allway(stats.f_oneway, statname="f-test", control=None, **opts)

    def tukey_hsd(self) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Tukey Honestly Significant Difference (HSD) test across stations + other groups
        for each parameter
        """
        hsd = utils.tukey_hsd(
            self.tidy, self.rescol, self.stationcol, self.paramcol, *self.othergroups
        )
        scores = utils.process_tukey_hsd_scores(hsd, self.stationcol, self.paramcol)
        return hsd, scores

    def dunn(self):
        """
        Dunn test across the different stations for each pollutant
        """
        return self.tidy.groupby(by=[self.paramcol]).apply(
            lambda g: utils.dunn_test(g, self.rescol, self.stationcol, *self.othergroups).scores
        )

    def theilslopes(self, logs=False):
        raise NotImplementedError

    @cache_readonly
    def locations(self):
        _locations = []
        groups = (
            self.data.groupby(by=self.groupcols).filter(self.filterfxn).groupby(by=self.groupcols)
        )
        cols = [self._raw_rescol, self.qualcol]
        for names, data in groups:
            loc_dict = dict(zip(self.groupcols, names))
            loc = (
                data.set_index(self.pairgroups)[cols]
                .reset_index(level=self.stationcol, drop=True)
                .pipe(
                    Location,
                    station_type=loc_dict[self.stationcol].lower(),
                    rescol=self._raw_rescol,
                    qualcol=self.qualcol,
                    ndval=self.ndval,
                    bsiter=self.bsiter,
                    useros=self.useros,
                )
            )

            loc.definition = loc_dict
            _locations.append(loc)

        return _locations

    def datasets(self, loc1, loc2):
        """Generate ``Dataset`` objects from the raw data of the
        ``DataColletion``.

        Data are first grouped by ``self.groupcols`` and
        ``self.stationcol``. Data frame each group are then queried
        for into separate ``Lcoations`` from ``loc1`` and ``loc2``.
        The resulting ``Locations`` are used to create a ``Dataset``.

        Parameters
        ----------
        loc1, loc2 : string
            Values found in the ``self.stationcol`` property that will
            be used to distinguish the two ``Location`` objects for the
            ``Datasets``.

        Yields
        ------
        ``Dataset`` objects

        """

        groupcols = list(filter(lambda g: g != self.stationcol, self.groupcols))

        for names, data in self.data.groupby(by=groupcols):
            ds_dict = dict(zip(groupcols, names))

            ds_dict[self.stationcol] = loc1
            infl = self.selectLocations(squeeze=True, **ds_dict)

            ds_dict[self.stationcol] = loc2
            effl = self.selectLocations(squeeze=True, **ds_dict)

            ds_dict.pop(self.stationcol)
            dsname = "_".join(names).replace(", ", "")

            if effl:
                ds = Dataset(infl, effl, useros=self.useros, name=dsname)
                ds.definition = ds_dict
                yield ds

    @staticmethod
    def _filter_collection(collection, squeeze, **kwargs):
        items = list(collection)
        for key, value in kwargs.items():
            if numpy.isscalar(value):
                items = [r for r in filter(lambda x: x.definition[key] == value, items)]
            else:
                items = [r for r in filter(lambda x: x.definition[key] in value, items)]

        if squeeze:
            if len(items) == 1:
                items = items[0]
            elif len(items) == 0:
                items = None

        return items

    def selectLocations(self, squeeze=False, **conditions):
        """Select ``Location`` objects meeting specified criteria
        from the ``DataColletion``.

        Parameters
        ----------
        squeeze : bool, optional
            When True and only one object is found, it returns the bare
            object. Otherwise, a list is returned.
        **conditions : optional parameters
            The conditions to be applied to the definitions of the
            ``Locations`` to filter them out. If a scalar is provided
            as the value, normal comparison (==) is used. If a sequence
            is provided, the ``in`` operator is used.

        Returns
        -------
        locations : list of ``wqio.Location`` objects

        Example
        -------
        >>> from wqio.tests.helpers import make_dc_data_complex
        >>> import wqio
        >>> df = make_dc_data_complex()
        >>> dc = wqio.DataCollection(df, rescol='res', qualcol='qual',
        ...                          stationcol='loc', paramcol='param',
        ...                          ndval='<', othergroups=None,
        ...                          pairgroups=['state', 'bmp'],
        ...                          useros=True, bsiter=10000)
        >>> locs = dc.selectLocations(param=['A', 'B'], loc=['Inflow', 'Reference'])
        >>> len(locs)
        4
        >>> locs[0].definition
        {'loc': 'Inflow', 'param': 'A'}

        """

        locations = self._filter_collection(self.locations.copy(), squeeze=squeeze, **conditions)
        return locations

    def selectDatasets(self, loc1, loc2, squeeze=False, **conditions):
        """Select ``Dataset`` objects meeting specified criteria
        from the ``DataColletion``.

        Parameters
        ----------
        loc1, loc2 : string
            Values found in the ``self.stationcol`` property that will
            be used to distinguish the two ``Location`` objects for the
            ``Datasets``.
        squeeze : bool, optional
            When True and only one object is found, it returns the bare
            object. Otherwise, a list is returned.
        **conditions : optional parameters
            The conditions to be applied to the definitions of the
            ``Locations`` to filter them out. If a scalar is provided
            as the value, normal comparison (==) is used. If a sequence
            is provided, the ``in`` operator is used.

        Returns
        -------
        locations : list of ``wqio.Location`` objects

        Example
        -------
        >>> from wqio.tests.helpers import make_dc_data_complex
        >>> import wqio
        >>> df = make_dc_data_complex()
        >>> dc = wqio.DataCollection(df, rescol='res', qualcol='qual',
        ...                          stationcol='loc', paramcol='param',
        ...                          ndval='<', othergroups=None,
        ...                          pairgroups=['state', 'bmp'],
        ...                          useros=True, bsiter=10000)
        >>> dsets = dc.selectDatasets('Inflow', 'Outflow', squeeze=False,
        ... param=['A', 'B'])
        >>> len(dsets)
        2
        >>> dsets[0].definition
        {'param': 'A'}
        """

        datasets = self._filter_collection(self.datasets(loc1, loc2), squeeze=squeeze, **conditions)
        return datasets

    def n_unique(self, column):
        return (
            self.data.loc[:, self.groupcols + [column]]
            .drop_duplicates()
            .groupby(self.groupcols)
            .size()
            .unstack(level=self.stationcol)
            .pipe(utils.add_column_level, column, "result")
            .swaplevel(axis="columns")
            .fillna(0)
            .astype(int)
        )

    def stat_summary(self, percentiles=None, groupcols=None, useros=True):
        """A generic, high-level summary of the data collection.

        Parameters
        ----------
        groupcols : list of strings, optional
            The columns by which ``self.tidy`` will be grouped when
            computing the statistics.
        useros : bool, optional
            Toggles of the use of the ROS'd (``True``) or raw
            (``False``) data.

        Returns
        -------
        stat_df : pandas.DataFrame

        """
        groupcols = validate.at_least_empty_list(groupcols) or self.groupcols
        ptiles = percentiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        summary = (
            self.tidy.groupby(by=groupcols)
            .apply(lambda g: g[self.rescol].describe(percentiles=ptiles).T)
            .drop("count", axis="columns")
        )
        return self.inventory.join(summary).unstack(level=self.stationcol)
