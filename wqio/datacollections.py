import numpy
from scipy import stats
from matplotlib import pyplot
import pandas
import statsmodels.api as sm
from statsmodels.tools.decorators import resettable_cache, cache_readonly

from wqio import utils
from wqio import bootstrap
from wqio.ros import ROS
from wqio import validate
from wqio.features import Location, Dataset


class DataCollection(object):
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

    """

    # column that stores the censorsip status of an observation
    cencol = '__censorship'

    def __init__(self, dataframe, rescol='res', qualcol='qual',
                 stationcol='station', paramcol='parameter', ndval='ND',
                 othergroups=None, pairgroups=None, useros=True,
                 filterfxn=None, bsiter=10000):

        # cache for all of the properties
        self._cache = resettable_cache()

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

        # column that stores ROS'd values
        self.roscol = 'ros_' + rescol

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
        self.tidy_columns = self.groupcols + [self._raw_rescol, self.cencol]

        # the "raw" data with the censorship column added
        self.data = (
            dataframe
            .assign(**{self.cencol: dataframe[self.qualcol].isin(self.ndval)})
            .reset_index()
        )

    @cache_readonly
    def tidy(self):
        if self.useros:
            def fxn(g):
                rosdf = (
                    ROS(df=g, result=self._raw_rescol, censorship=self.cencol, as_array=False)
                    .rename(columns={'final': self.roscol})
                    [[self._raw_rescol, self.roscol, self.cencol]]
                )
                return rosdf
        else:
            def fxn(g):
                g[self.roscol] = numpy.nan
                return g

        keep_cols = self.tidy_columns + [self.roscol]
        _tidy = (
            self.data
            .reset_index()[self.tidy_columns]
            .groupby(by=self.groupcols)
            .filter(self.filterfxn)
            .groupby(by=self.groupcols)
            .apply(fxn)
            .reset_index()
            .sort_values(by=self.groupcols)
        )

        return _tidy[keep_cols]

    @cache_readonly
    def paired(self):
        _pairs = (
            self.data
                .reset_index()
                .groupby(by=self.groupcols)
                .filter(self.filterfxn)
                .set_index(self.pairgroups)
                .unstack(level=self.stationcol)
                .rename_axis(['value', self.stationcol], axis='columns')
        )[[self._raw_rescol, self.cencol]]
        return _pairs

    def generic_stat(self, statfxn, use_bootstrap=True, statname=None,
                     has_pvalue=False, filterfxn=None, **statopts):
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
            statname = 'stat'

        if filterfxn is None:
            filterfxn = utils.non_filter

        def fxn(x):
            data = x[self.rescol].values
            if use_bootstrap:
                stat = statfxn(data)
                lci, uci = bootstrap.BCA(data, statfxn=statfxn)
                values = [lci, stat, uci]
                statnames = ['lower', statname, 'upper']
            else:
                values = validate.at_least_empty_list(statfxn(data, **statopts))
                if hasattr(values, '_fields'):  # nametuple
                    statnames = values._fields
                else:  # tuple
                    statnames = [statname]
                    if has_pvalue:
                        statnames.append('pvalue')

            return pandas.Series(values, index=statnames)

        stat = (
            self.tidy
                .groupby(by=self.groupcols)
                .filter(filterfxn)
                .groupby(by=self.groupcols)
                .apply(fxn)
                .unstack(level=self.stationcol)
                .pipe(utils.swap_column_levels, 0, 1)
                .rename_axis(['station', 'result'], axis='columns')
        )

        return stat

    @cache_readonly
    def count(self):
        return self.generic_stat(lambda x: x.shape[0], use_bootstrap=False, statname='Count')

    @cache_readonly
    def inventory(self):
        counts = (
            self.tidy
                .groupby(by=self.groupcols + [self.cencol])
                .size()
                .unstack(level=self.cencol)
                .fillna(0)
                .astype(int)
                .rename_axis(None, axis='columns')
                .rename(columns={False: 'Detect', True: 'Non-Detect'})
                .assign(Count=lambda df: df.sum(axis='columns'))
        )
        if 'Non-Detect' not in counts.columns:
            counts['Non-Detect'] = 0

        return counts[['Count', 'Non-Detect']]

    @cache_readonly
    def median(self):
        return self.generic_stat(numpy.median, statname='median')

    @cache_readonly
    def mean(self):
        return self.generic_stat(numpy.mean, statname='mean')

    @cache_readonly
    def std_dev(self):
        return self.generic_stat(numpy.std, statname='std. dev.', use_bootstrap=False, )

    def percentile(self, percentile):
        """Return the percentiles (0 - 100) for the data."""
        return self.generic_stat(lambda x: numpy.percentile(x, percentile),
                                 statname='pctl {}'.format(percentile),
                                 use_bootstrap=False)

    @cache_readonly
    def logmean(self):
        return self.generic_stat(lambda x, axis=0: numpy.mean(numpy.log(x), axis=axis),
                                 statname='Log-mean')

    @cache_readonly
    def logstd_dev(self):
        return self.generic_stat(lambda x, axis=0: numpy.std(numpy.log(x), axis=axis),
                                 use_bootstrap=False, statname='Log-std. dev.')

    @cache_readonly
    def geomean(self):
        geomean = numpy.exp(self.logmean)
        geomean.columns.names = ['station', 'Geo-mean']
        return geomean

    @cache_readonly
    def geostd_dev(self):
        geostd = numpy.exp(self.logstd_dev)
        geostd.columns.names = ['station', 'Geo-std. dev.']
        return geostd

    @cache_readonly
    def shapiro(self):
        return self.generic_stat(stats.shapiro, use_bootstrap=False,
                                 has_pvalue=True, statname='shapiro',
                                 filterfxn=lambda x: x.shape[0] > 3)

    @cache_readonly
    def shapiro_log(self):
        return self.generic_stat(lambda x: stats.shapiro(numpy.log(x)),
                                 use_bootstrap=False, has_pvalue=True,
                                 filterfxn=lambda x: x.shape[0] > 3,
                                 statname='log-shapiro')

    @cache_readonly
    def lilliefors(self):
        return self.generic_stat(sm.stats.lilliefors, use_bootstrap=False,
                                 has_pvalue=True, statname='lilliefors')

    @cache_readonly
    def lilliefors_log(self):
        return self.generic_stat(lambda x: sm.stats.lilliefors(numpy.log(x)),
                                 use_bootstrap=False, has_pvalue=True,
                                 statname='log-lilliefors')

    @cache_readonly
    def anderson_darling(self):
        raise NotImplementedError
        return self.generic_stat(utils.anderson_darling, use_bootstrap=False,
                                 has_pvalue=True, statname='anderson-darling')

    @cache_readonly
    def anderson_darling_log(self):
        raise NotImplementedError
        return self.generic_stat(lambda x: utils.anderson_darling(numpy.log(x)),
                                 use_bootstrap=False, has_pvalue=True,
                                 statname='log-anderson-darling')

    def comparison_stat(self, statfxn, statname=None, paired=False, **statopts):
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
        apired : bool, optional
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
        >>> mwht = dc.comparison_stat(stats.mannwhitneyu,
        ...                           statname='mann_whitney',
        ...                           alternative='two-sided')

        """

        if paired:
            data = self.paired
            generator = utils.numutils._paired_stat_generator
            rescol = self._raw_rescol
        else:
            data = self.tidy
            generator = utils.numutils._comp_stat_generator
            rescol = self.rescol

        station_columns = [self.stationcol + '_1', self.stationcol + '_2']
        meta_columns = self.groupcols_comparison
        index_cols = meta_columns + station_columns

        results = generator(
            data,
            meta_columns,
            self.stationcol,
            rescol,
            statfxn,
            statname=statname,
            **statopts
        )
        return pandas.DataFrame.from_records(results).set_index(index_cols)

    @cache_readonly
    def mann_whitney(self):
        return self.comparison_stat(stats.mannwhitneyu, statname='mann_whitney',
                                    alternative='two-sided')

    @cache_readonly
    def ranksums(self):
        return self.comparison_stat(stats.ranksums, statname='rank_sums')

    @cache_readonly
    def t_test(self):
        return self.comparison_stat(stats.ttest_ind, statname='t_test', equal_var=False)

    @cache_readonly
    def levene(self):
        return self.comparison_stat(stats.levene, statname='levene', center='median')

    @cache_readonly
    def wilcoxon(self):
        return self.comparison_stat(stats.wilcoxon, statname='wilcoxon', paired=True)

    @cache_readonly
    def kendall(self):
        return self.comparison_stat(stats.kendalltau, statname='kendalltau', paired=True)

    @cache_readonly
    def spearman(self):
        return self.comparison_stat(stats.spearmanr, statname='spearmanrho', paired=True)

    @cache_readonly
    def theilslopes(self, logs=False):
        raise NotImplementedError

    @cache_readonly
    def locations(self):
        _locations = []
        groups = (
            self.data
                .groupby(by=self.groupcols)
                .filter(self.filterfxn)
                .groupby(by=self.groupcols)
        )
        cols = [self._raw_rescol, self.qualcol]
        for names, data in groups:
            loc_dict = dict(zip(self.groupcols, names))
            loc = (
                data.set_index(self.pairgroups)[cols]
                    .reset_index(level=self.stationcol, drop=True)
                    .pipe(Location, station_type=loc_dict[self.stationcol].lower(),
                          rescol=self._raw_rescol, qualcol=self.qualcol,
                          ndval=self.ndval, bsiter=self.bsiter, useros=self.useros)
            )

            loc.definition = loc_dict
            _locations.append(loc)

        return _locations

    def datasets(self, loc1, loc2):
        """ Generate ``Dataset`` objects from the raw data of the
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
            dsname = '_'.join(names).replace(', ', '')

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
        """ Select ``Location`` objects meeting specified criteria
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

        locations = self._filter_collection(
            self.locations.copy(), squeeze=squeeze, **conditions
        )
        return locations

    def selectDatasets(self, loc1, loc2, squeeze=False, **conditions):
        """ Select ``Dataset`` objects meeting specified criteria
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

        datasets = self._filter_collection(
            self.datasets(loc1, loc2), squeeze=squeeze, **conditions
        )
        return datasets

    def stat_summary(self, percentiles=None, groupcols=None, useros=True):
        """ A generic, high-level summary of the data collection.

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

        if useros:
            col = self.roscol
        else:
            col = self.rescol

        if groupcols is None:
            groupcols = self.groupcols
        else:
            groupcols = validate.at_least_empty_list(groupcols)

        ptiles = percentiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        summary = (
            self.tidy
            .groupby(by=groupcols)
            .apply(lambda g: g[col].describe(percentiles=ptiles).T)
            .drop('count', axis='columns')
        )
        return self.inventory.join(summary).unstack(level=self.stationcol)
