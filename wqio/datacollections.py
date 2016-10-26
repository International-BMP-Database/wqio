import numpy
from scipy import stats
from matplotlib import pyplot
import pandas
import statsmodels.api as sm
from statsmodels.tools.decorators import resettable_cache, cache_readonly
import seaborn.apionly as seaborn

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
    ndval : string (default = 'ND')
        The *only* value found in ``qualcol`` that indicates that a
        result is a non-detect. Non-detect results should be reported
        as the detection limits.
    othergroups : list of strings, optional
        Other columns besides ``stationcol`` and ``paramcol`` that
        should be considered when grouping into subsets of data.
    pairgroups : list of strings, optional
        Other columns besides ``stationcol`` and ``paramcol`` that
        can be sued define a unique index on ``dataframe`` such that it
        can be "unstack" (i.e., pivoted, cross-tabbed) to place the
        ``stationcol`` values into columns.
    useros : bool (default = True)
        Toggles the use of Regression On Order Statistics to
        estimate non-detect values when computing statistics.
    filterfxn : callable, optional
        Function that will we passes to a pandas.Groupby object that
        will filter out groups that should not be analyzed (for
        whatever reason).
    bsiter : int
        Number of iterations the bootstrapper should use when estimating
        confidence intervals around a statistic.

    """

    cencol = '__censorship'

    def __init__(self, dataframe, rescol='res', qualcol='qual',
                 stationcol='station', paramcol='parameter', ndval='ND',
                 othergroups=None, pairgroups=None, useros=True,
                 filterfxn=None, bsiter=10000):

        # cache for all of the properties
        self._cache = resettable_cache()

        # basic input
        self._raw_data = dataframe
        self._raw_rescol = rescol
        self.filterfxn = filterfxn
        self.qualcol = qualcol
        self.stationcol = stationcol
        self.paramcol = paramcol
        self.ndval = validate.at_least_empty_list(ndval)
        self.othergroups = validate.at_least_empty_list(othergroups)
        self.pairgroups = validate.at_least_empty_list(pairgroups)
        self.bsiter = bsiter

        self.useros = useros

        self.roscol = 'ros_' + rescol
        if self.useros:
            self.rescol = self.roscol
        else:
            self.rescol = rescol

        self.groupcols = [self.stationcol, self.paramcol] + self.othergroups
        self.groupcols_comparison = [self.paramcol] + self.othergroups

        self.pairgroups = self.pairgroups + [self.stationcol, self.paramcol]

        self.columns = self.groupcols + [self._raw_rescol, self.cencol]
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

        keep_cols = self.columns + [self.roscol]
        _tidy = (
            self.data
                .reset_index()[self.columns]
                .groupby(by=self.groupcols)
                .filter(self.filterfxn)
                .groupby(by=self.groupcols)
                .apply(fxn)
                .reset_index()
                .sort_values(by=self.groupcols)
        )[keep_cols]

        return _tidy

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

    def _generic_stat(self, statfxn, use_bootstrap=True, statname=None,
                      has_pvalue=False, **statopts):
        if statname is None:
            statname = 'stat'

        def fxn(x):
            data = x[self.rescol].values
            if use_bootstrap:
                stat = statfxn(data)
                lci, uci = bootstrap.BCA(data, statfxn=statfxn)
                values = [lci, stat, uci]
                statnames = ['lower', statname, 'upper']
            else:
                values = validate.at_least_empty_list(statfxn(data, **statopts))
                # nametuple
                if hasattr(values, '_fields'):
                    statnames = values._fields
                # tuple
                else:
                    statnames = [statname]
                    if has_pvalue:
                        statnames.append('pvalue')

            return pandas.Series(values, index=statnames)

        stat = (
            self.tidy
                .groupby(by=self.groupcols)
                .apply(fxn)
                .unstack(level=self.stationcol)
                .pipe(utils.swap_column_levels, 0, 1)
                .rename_axis(['station', 'result'], axis='columns')
        )

        return stat

    @cache_readonly
    def count(self):
        return self._generic_stat(lambda x: x.shape[0], use_bootstrap=False, statname='Count')

    @cache_readonly
    def median(self):
        return self._generic_stat(numpy.median, statname='median')

    @cache_readonly
    def mean(self):
        return self._generic_stat(numpy.mean, statname='mean')

    @cache_readonly
    def std_dev(self):
        return self._generic_stat(numpy.std, statname='std. dev.', use_bootstrap=False, )

    def percentile(self, percentile):
        """Return the percentiles (0 - 100) for the data."""
        return self._generic_stat(lambda x: numpy.percentile(x, percentile),
                                  statname='pctl {}'.format(percentile),
                                  use_bootstrap=False)

    @cache_readonly
    def logmean(self):
        return self._generic_stat(lambda x, axis=0: numpy.mean(numpy.log(x), axis=axis), statname='Log-mean')

    @cache_readonly
    def logstd_dev(self):
        return self._generic_stat(lambda x, axis=0: numpy.std(numpy.log(x), axis=axis),
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
        return self._generic_stat(stats.shapiro, use_bootstrap=False,
                                  has_pvalue=True, statname='shapiro')

    @cache_readonly
    def shapiro_log(self):
        return self._generic_stat(lambda x: stats.shapiro(numpy.log(x)),
                                  use_bootstrap=False, has_pvalue=True,
                                  statname='log-shapiro')

    @cache_readonly
    def lillifors(self):
        return self._generic_stat(sm.stats.lillifors, use_bootstrap=False,
                                  has_pvalue=True, statname='lillifors')

    @cache_readonly
    def lillifors_log(self):
        return self._generic_stat(lambda x: sm.stats.lillifors(numpy.log(x)),
                                  use_bootstrap=False, has_pvalue=True,
                                  statname='log-lillifors')

    @cache_readonly
    def anderson_darling(self):
        raise NotImplementedError
        return self._generic_stat(utils.anderson_darling, use_bootstrap=False,
                                  has_pvalue=True, statname='anderson-darling')

    @cache_readonly
    def anderson_darling_log(self):
        raise NotImplementedError
        return self._generic_stat(lambda x: utils.anderson_darling(numpy.log(x)),
                                  use_bootstrap=False, has_pvalue=True,
                                  statname='log-anderson-darling')

    def _comparison_stat(self, statfxn, statname=None, paired=False, **statopts):
        if paired:
            data = self.paired
            meta_columns = self.groupcols_comparison
            generator = utils.misc._paired_stat_generator
            rescol = self._raw_rescol
        else:
            data = self.tidy
            meta_columns = self.groupcols_comparison
            generator = utils.misc._comp_stat_generator
            rescol = self.rescol

        station_columns = [self.stationcol + '_1', self.stationcol + '_2']
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
        return self._comparison_stat(stats.mannwhitneyu, statname='mann_whitney', alternative='two-sided')

    @cache_readonly
    def t_test(self):
        return self._comparison_stat(stats.ttest_ind, statname='t_test', equal_var=False)

    @cache_readonly
    def levene(self):
        return self._comparison_stat(stats.levene, statname='levene', center='median')

    @cache_readonly
    def wilcoxon(self):
        return self._comparison_stat(stats.wilcoxon, statname='wilcoxon', paired=True)

    @cache_readonly
    def kendall(self):
        return self._comparison_stat(stats.kendalltau, statname='kendalltau', paired=True)

    @cache_readonly
    def spearman(self):
        return self._comparison_stat(stats.spearmanr, statname='spearmanrho', paired=True)

    @cache_readonly
    def theilslopes(self):
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
        for names, data in groups:
            loc_dict = dict(zip(self.groupcols, names))
            loc = Location(
                data.copy(), station_type=loc_dict[self.stationcol].lower(),
                rescol=self._raw_rescol, qualcol=self.qualcol,
                ndval=self.ndval, bsiter=self.bsiter, useros=self.useros
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

            ds = Dataset(infl, effl, useros=self.useros, name=dsname)
            ds.definition = ds_dict

            yield ds

    @staticmethod
    def _filter_collection(collection, squeeze, **kwargs):
        items = collection.copy()
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
        >>> from wqio.tests import make_dc_data_complex
        >>> import wqio
        >>> df = make_dc_data_complex()
        >>> dc = wqio.DataCollection(df, rescol='res', qualcol='qual',
        ...                          stationcol='loc', paramcol='param',
        ...                          ndval='<', othergroups=None,
        ...                          pairgroups=['state', 'bmp'],
        ...                          useros=True, bsiter=10000)
        >>> dc.selectLocations(param='A', loc=['Inflow', 'Reference'])

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
        >>> from wqio.tests import make_dc_data_complex
        >>> import wqio
        >>> df = make_dc_data_complex()
        >>> dc = wqio.DataCollection(df, rescol='res', qualcol='qual',
        ...                          stationcol='loc', paramcol='param',
        ...                          ndval='<', othergroups=None,
        ...                          pairgroups=['state', 'bmp'],
        ...                          useros=True, bsiter=10000)
        >>> dc.selectDatasets('Inflow', 'Outflow', param='A',
        ...                   state=['OR', 'CA'])

        """
        datasets = self._filter_collection(
            self.datasets(loc1, loc2), squeeze=squeeze, **conditions
        )
        return datasets

    def stat_summary(self, groupcols=None, useros=True):
        if useros:
            col = self.roscol
        else:
            col = self.rescol

        if groupcols is None:
            groupcols = self.groupcols

        ptiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        summary = (
            self.tidy
                .groupby(by=groupcols)
                .apply(lambda g: g[col].describe(percentiles=ptiles).T)
                .unstack(level=self.stationcol)
        )
        return summary
