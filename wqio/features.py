import itertools

import numpy
from scipy import stats
from matplotlib import pyplot
import pandas
import statsmodels.api as sm
from statsmodels.tools.decorators import resettable_cache, cache_readonly
import seaborn
from probscale.algo import _estimate_from_fit

from wqio import utils
from wqio import bootstrap
from wqio.ros import ROS
from wqio import validate
from wqio import viz


# meta data mappings based on station
station_names = {
    'inflow': 'Influent',
    'outflow': 'Effluent',
    'reference': 'Reference Flow',
}

markers = {
    'Influent': ['o', 'v'],
    'Effluent': ['s', '<'],
    'Reference Flow': ['D', 'd'],
}

palette = seaborn.color_palette(palette='deep', n_colors=3, desat=0.88)
colors = {
    'Influent': palette[0],
    'Effluent': palette[1],
    'Reference Flow': palette[2],
}


class Location(object):
    """ Object providing convenient access to statistical and
    graphical methods for summarizing a single set of water quality
    observations for a single pollutant.

    Parameters
    -----------
    dataframe : pandas.DataFrame
        A dataframe that contains at least two columns: one for the
        analytical values and another for the data qualfiers. Can
        contain any type of row index, but the column index must be
        simple (i.e., not a pandas.MultiIndex).
    rescol : string, optional (default = 'res')
        Name of the column in `dataframe` that contains the
        analytical values.
    qualcol : string, optional (default = 'qual')
        Name of the column in `dataframe` containing qualifiers.
    ndval : string, optional (default = 'ND')
        The *only* value in the `qualcol` of `dataframe` that
        indicates that the corresponding value in `rescol` is
        non-detect.
    station_type : string, optional
        Type of location being analyzed. Valid values are:
        'inflow' (default) or 'outflow'.
    useros : bool, optional (default = True)
        Toggles the use of Regression On Order Statistics to
        estimate non-detect values when computing statistics.
    cencol : string, optional (default = 'cen')
        Name of the column indicaticating if a results is censored.
        These values will be computed from ``qualcol`` and ``ndval``.
    bsiter : int, optional (default = 1e4)
        Number of interations to use when using a bootstrap
        algorithm to refine a statistic.
    include : bool, optional (default = True)
        Toggles the inclusion of the location when programmatically
        creating many `Location` objects.

    Settable Properties
    -------------------
    .name : string
        A human-readable name for the data.
    .definition : dict
        A dictionary of key-value pairs that define what makes this
        data distinct within a larger collection of `Location` objects.
    .include : bool
        Same as input.

    Statistical Attributes
    ----------------------
    .N : int
        Total number of results.
    .ND : int
        Number of non-detect results.
    .NUnique : int
        Number of unique result values in the data.
    .fractionND : float
        Fraction of data that is non-detect.
    .min : float
        Minimum value of the data.
    .max : float
        Maximum value of the data.
    .min_detect : float
        Minimum detected value of the data.
    .min_DL : float
        Minimum detection limit reported for the data.
    .mean*+^ : float
        Bootstrapped arithmetic mean of the data.
    .std*+^ : float
        Bootstrapped standard deviation of the data.
    .cov : float
        Covariance (absolute value of std/mean).
    .skew : float
        Skewness coefficient.
    .median* : float
        Median of the dataset.
    .pctl[10/25/75/90] : float
        Percentiles of the dataset.
    .pnorm : float
        Results of the Shapiro-Wilks test for normality.
    .plognorm : float
        Results of the Shapiro-Wilks test for normality on
        log-transormed data (so really a test for lognormalily).
    .lilliefors+ : list of floats
        Lilliefors statistic and p-value.
    .shapiro+ : list of floats
        Shapiro-Wilks statistic and p-value.
    .anderson+ : tuple
        Anderson-Darling statistic, critical values, significance
        levels.
    .analysis_space : string
        Based on the results of self.pnorm and self.plognorm, this is
        either "normal" or "lognormal".

    Statistial Notes
    ----------------
    * Indicates that there's an accompyaning tuple of confidence
      interval. For example, self.mean and self.mean_conf_interval
    + Indicatates that there's a equivalent stat for log-transormed
      data. For example, self.mean and self.logmean or self.lilliefors
      and self.lilliefors_log (subject to the absense of negitive
      results).
    ^ Indicatates that there's a equivalent stat in geometric space.
      For example, self.mean and self.geomean.

    Plotting Methods
    ----------------
    .verticalScatter
    .boxplot
    .probplot
    .statplot

    """

    def __init__(self, dataframe, rescol='res', qualcol='qual', ndval='ND',
                 station_type='inflow', useros=True, cencol='cen',
                 bsiter=10000, include=True):
        # plotting symbology based on location type
        self.station_type = station_type
        self.station_name = station_names[station_type]
        self.plot_marker = markers[self.station_name][0]
        self.scatter_marker = markers[self.station_name][1]
        self.color = colors[self.station_name]

        # basic stuff
        self._name = self.station_name
        self._include = include
        self._definition = {}

        # parameters of the stats analysis
        self._cache = resettable_cache()

        # properties of the dataframe and analysis
        self.bsiter = bsiter
        self.useros = useros
        self.rescol = rescol
        self.qualcol = qualcol
        self.cencol = cencol
        if numpy.isscalar(ndval):
            self.ndvals = [ndval]
        else:
            self.ndvals = ndval

        # original data and quantity
        self.raw_data = dataframe.assign(
            **{self.cencol: dataframe[qualcol].isin(self.ndvals)}
        )
        self._dataframe = None
        self._data = None

    @property
    def dataframe(self):
        if self.raw_data.shape[0] > 0 and self._dataframe is None:
            df = (
                self.raw_data
                    .assign(**{self.cencol: lambda df: df[self.qualcol].isin(self.ndvals)})
            )
            if self.useros:
                ros = ROS(df=df, result=self.rescol, censorship=self.cencol, as_array=False)
                self._dataframe = ros[['final', self.cencol]]
            else:
                self._dataframe = df[[self.rescol, self.cencol]]
        return self._dataframe

    @property
    @numpy.deprecate
    def full_data(self):
        return self.dataframe

    @property
    def data(self):
        if self.hasData:
            if self.useros:
                output = self.dataframe['final'].values
            else:
                output = self.dataframe[self.rescol].values

            return output

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def definition(self):
        return self._definition

    @definition.setter
    def definition(self, value):
        self._definition = value

    @property
    def include(self):
        return self._include

    @include.setter
    def include(self, value):
        self._include = value

    @property
    def exclude(self):
        return not self.include

    @cache_readonly
    def N(self):
        return self.data.shape[0]

    @cache_readonly
    def hasData(self):
        return self.dataframe.shape[0] > 0

    @cache_readonly
    def all_positive(self):
        if self.hasData:
            return self.min > 0

    @cache_readonly
    def ND(self):
        return self.dataframe[self.cencol].sum()

    @cache_readonly
    def NUnique(self):
        return pandas.unique(self.raw_data[self.rescol]).shape[0]

    @cache_readonly
    def fractionND(self):
        return self.ND / self.N

    @cache_readonly
    def shapiro(self):
        if self.hasData:
            return stats.shapiro(self.data)

    @cache_readonly
    def shapiro_log(self):
        if self.hasData:
            return stats.shapiro(numpy.log(self.data))

    @cache_readonly
    def lilliefors(self):
        if self.hasData:
            return sm.stats.lilliefors(self.data)

    @cache_readonly
    def lilliefors_log(self):
        if self.hasData:
            return sm.stats.lilliefors(numpy.log(self.data))

    @cache_readonly
    def anderson(self):
        if self.hasData:
            return utils.anderson_darling(self.data)

    @cache_readonly
    def anderson_log(self):
        if self.hasData:
            return utils.anderson_darling(numpy.log(self.data))

    @cache_readonly
    def analysis_space(self):
        if self.shapiro_log[1] >= self.shapiro[1] and self.shapiro_log[1] > 0.1:
            return 'lognormal'
        else:
            return 'normal'

    @cache_readonly
    def cov(self):
        if self.hasData:
            return self.data.std() / self.data.mean()

    @cache_readonly
    def min(self):
        if self.hasData:
            return self.data.min()

    @cache_readonly
    def min_detect(self):
        if self.hasData:
            return self.raw_data[self.rescol][~self.raw_data[self.qualcol].isin(self.ndvals)].min()

    @cache_readonly
    def min_DL(self):
        if self.hasData:
            return self.raw_data[self.rescol][self.raw_data[self.qualcol].isin(self.ndvals)].min()

    @cache_readonly
    def max(self):
        if self.hasData:
            return self.data.max()

    @cache_readonly
    def skew(self):
        if self.hasData:
            return stats.skew(self.data)

    @cache_readonly
    def pctl10(self):
        if self.hasData:
            return numpy.percentile(self.data, 10)

    @cache_readonly
    def pctl25(self):
        if self.hasData:
            return numpy.percentile(self.data, 25)

    @cache_readonly
    def pctl75(self):
        if self.hasData:
            return numpy.percentile(self.data, 75)

    @cache_readonly
    def pctl90(self):
        if self.hasData:
            return numpy.percentile(self.data, 90)

    # stats that we need
    @cache_readonly
    def median(self):
        if self.hasData:
            return numpy.median(self.data)

    @cache_readonly
    def median_conf_interval(self):
        if self.hasData:
            return bootstrap.BCA(self.data, numpy.median, niter=self.bsiter)

    @cache_readonly
    def mean(self):
        if self.hasData:
            return numpy.mean(self.data)

    @cache_readonly
    def mean_conf_interval(self):
        if self.hasData:
            return bootstrap.BCA(self.data, numpy.mean, niter=self.bsiter)

    @cache_readonly
    def std(self):
        if self.hasData:
            return numpy.std(self.data)

    @cache_readonly
    def logmean(self):
        if self.all_positive and self.hasData:
            return numpy.mean(numpy.log(self.data))

    @cache_readonly
    def logmean_conf_interval(self):
        if self.all_positive and self.hasData:
            def fxn(x, **kwds):
                return numpy.mean(numpy.log(x), **kwds)

            return bootstrap.BCA(self.data, fxn, niter=self.bsiter)

    @cache_readonly
    def logstd(self):
        if self.all_positive and self.hasData:
            return numpy.std(numpy.log(self.data))

    @cache_readonly
    def geomean(self):
        if self.all_positive and self.hasData:
            return numpy.exp(self.logmean)

    @cache_readonly
    def geomean_conf_interval(self):
        if self.all_positive and self.hasData:
            return numpy.exp(self.logmean_conf_interval)

    @cache_readonly
    def geostd(self):
        if self.all_positive and self.hasData:
            return numpy.exp(self.logstd)

    def boxplot_stats(self, log=True, bacteria=False):
        bxpstats = {
            'label': self.name,
            'mean': self.geomean if bacteria else self.mean,
            'med': self.median,
            'q1': self.pctl25,
            'q3': self.pctl75,
            'cilo': self.median_conf_interval[0],
            'cihi': self.median_conf_interval[1],
        }

        if log:
            wnf = viz.whiskers_and_fliers(
                numpy.log(self.data),
                numpy.log(self.pctl25),
                numpy.log(self.pctl75),
                transformout=numpy.exp
            )
        else:
            wnf = viz.whiskers_and_fliers(
                self.data,
                self.pctl25,
                self.pctl75,
                transformout=None
            )

        bxpstats.update(wnf)
        return [bxpstats]

    # plotting methods
    def boxplot(self, ax=None, pos=1, yscale='log', shownotches=True, showmean=True,
                width=0.8, bacteria=False, ylabel=None, xlabel=None,
                patch_artist=False, xlims=None):
        """ Draws a boxplot and whisker on a matplotlib figure

        Parameters
        ----------
        ax : optional matplotlib axes object or None (default)
            Axes on which the boxplot with be drawn. If None, one will
            be created.
        pos : optional int (default=1)
            Location along x-axis where boxplot will be placed.
        yscale : optional string ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        shownotches : optional bool (default=True)
            Toggles drawing of bootstrapped confidence interval around
            the median.
        showmean : optional bool (default=True)
            Toggles plotting the mean value on the boxplot as a point.
            See also the `bacteria` kwarg
        width : optional float (default=0.8)
            Width of boxplot on the axes (data units)
        bacteria : optional bool (default False)
            If True, uses the geometric mean when `showmean` is True.
            Otherwise, the arithmetic mean is used.
        ylabel : string or None (default):
            Label for y-axis
        xlabel : string or None (default):
            Label for x-axis. If None, uses self.name
        patch_artist : optional bool (default = False)
            Toggles the use of patch artist instead of a line artists
            for the boxes
        xlims : dict, optional
            Dictionary of limits for the x-axis. Keys must be either
            "left", "right", or both

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = validate.axes(ax)
        y_log = (yscale == 'log')
        bxpstats = self.boxplot_stats(log=y_log, bacteria=bacteria)
        if xlabel is not None:
            bxpstats[0]['label'] = xlabel

        bp = viz.boxplot(
            bxpstats,
            ax=ax,
            position=pos,
            width=width,
            color=self.color,
            marker=self.plot_marker,
            patch_artist=patch_artist,
            showmean=showmean,
            shownotches=shownotches,
        )

        ax.set_yscale(yscale)
        if yscale == 'log':
            ax.yaxis.set_major_formatter(viz.log_formatter(use_1x=True, threshold=6))

        if ylabel:
            ax.set_ylabel(ylabel)

        ax.set_xticks([pos])
        if self.name is not None:
            ax.set_xticklabels([self.name])
        if xlabel is not None:
            ax.set_xticklabels([xlabel])

        if xlims is not None:
            ax.set_xlim(**xlims)

        return fig

    def probplot(self, ax=None, yscale='log', axtype='prob',
                 ylabel=None, clearYLabels=False,
                 rotateticklabels=True, bestfit=False, **plotopts):
        """ Draws a probability plot on a matplotlib figure

        Parameters
        ----------
        ax : matplotlib axes object, optional or None (default).
            The Axes on which to plot. If None is provided, one will be
            created.
        axtype : string, optional (default = 'pp')
            Type of plot to be created. Options are:
                - 'prob': probabilty plot
                - 'pp': percentile plot
                - 'qq': quantile plot
        xlabel, ylabel : string, optional or None (default)
            Axis label for the plot.
        yscale : string , optional(default = 'log')
            Scale for the y-axis. Use 'log' for logarithmic (default) or
            'linear'.
        clearYLabels : bool, optional (default is False)
            If True, removed y-*tick* labels from teh Axes.
        rotateticklabels : bool, optional (default is True)
            If True, the tick labels of the probability axes will be
            rotated.
        bestfit : bool, optional (default is False)
            Specifies whether a best-fit line should be added to the
            plot.
        plotopts : keyword arguments
            Additional options passed directly to `pyplot.plot` when
            drawing the data points.

        Returns
        -------
        fig : matplotlib.Figure instance

        """

        fig, ax = validate.axes(ax)

        scatter_kws = plotopts.copy()
        scatter_kws['color'] = plotopts.get('color', self.color)
        scatter_kws['label'] = plotopts.get('label', self.name)
        scatter_kws['marker'] = plotopts.get('marker', self.plot_marker)
        scatter_kws['linestyle'] = plotopts.get('linestyle', 'none')
        fig = viz.probplot(self.data, ax=ax, axtype=axtype, yscale=yscale,
                           bestfit=bestfit, scatter_kws=scatter_kws)

        if yscale == 'log':
            pass

        if clearYLabels:
            ax.set_yticklabels([])

        if rotateticklabels:
            viz.rotateTickLabels(ax, 45, 'x', ha='right')

        if bestfit:
            utils.fit_line()

        return fig

    def statplot(self, pos=1, yscale='log', shownotches=True, showmean=True,
                 width=0.8, bacteria=False, ylabel=None, xlabel=None,
                 axtype='prob', patch_artist=False, **plotopts):
        """ Creates a two-axis figure with a boxplot & probability plot.

        Parameters
        ----------
        pos : int, optional (default=1)
            Location along x-axis where boxplot will be placed.
        yscale : string, optional ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        shownotches : bool, optional (default=True)
            Toggles drawing of bootstrapped confidence interval around
            the median.
        showmean : bool, optional (default=True)
            Toggles plotting the mean value on the boxplot as a point.
            See also the `bacteria` kwarg
        width : float, optional (default=0.8)
            Width of boxplot on the axes (data units)
        bacteria : bool, optional (default False)
            If True, uses the geometric mean when `showmean` is True.
            Otherwise, the arithmetic mean is used.
        ylabel : string, optional or None (default):
            Label for y-axis
        xlabel : string or None (default):
            Label for x-axis of boxplot. If None, uses self.name
        axtype : string, optional (default = 'pp')
            Type of probability plot to be created. Options are:
                - 'prob': probabilty plot
                - 'pp': percentile plot
                - 'qq': quantile plot
        patch_artist : bool, optional (default = False)
            Toggles the use of patch artist instead of a line artists
            for the boxes
        plotopts : keyword arguments
            Additional options passed directly to `pyplot.plot` when
            drawing the data points on the probability plots.

        Returns
        -------
        fig : matplotlib Figure

        """

        # setup the figure and axes
        fig = pyplot.figure(figsize=(6.40, 3.00), facecolor='none',
                            edgecolor='none')
        ax1 = pyplot.subplot2grid((1, 4), (0, 0))
        ax2 = pyplot.subplot2grid((1, 4), (0, 1), colspan=3)

        self.boxplot(ax=ax1, pos=pos, yscale=yscale, shownotches=shownotches,
                     showmean=showmean, width=width, bacteria=bacteria,
                     ylabel=ylabel, xlabel=xlabel, patch_artist=patch_artist,
                     xlims={'left': pos - (0.6 * width), 'right': pos + (0.6 * width)})

        self.probplot(ax=ax2, yscale=yscale, axtype=axtype,
                      ylabel=None, clearYLabels=True, **plotopts)

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        fig.subplots_adjust(wspace=0.05)
        return fig

    def verticalScatter(self, ax=None, pos=1, ylabel=None, yscale='log',
                        ignoreROS=True, markersize=6):
        """ Draws a clustered & jittered scatter plot of the data

        Parameters
        ----------
        ax : matplotlib axes object, optional or None (default)
            Axes on which the points with be drawn. If None, one will
            be created.
        pos : int, optional (default=1)
            Location along x-axis where data will be centered.
        jitter : float, optional (default=0.80)
            Width of the random x-values uniform distributed around
            `pos`
        alpha : float, optional (default=0.75)
            Opacity of the marker (1.0 -> opaque; 0.0 -> transparent)
        yscale : string, optional ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        ylabel : string, optional or None (default):
            Label for y-axis
        ignoreROS : bool, optional (default = True)
            By default, this function will plot the original, non-ROS'd
            data with a different symbol for non-detects. If `True`, the
            and `self.useros` is `True`, this function will plot the
            ROS'd data with a single marker.
        markersize : int, optional (default = 6)
            Size of data markers on the figure in points.

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = validate.axes(ax)

        if not ignoreROS and self.useros:
            rescol = 'final'
            hue_column = None
            hue_order = None
            df = self.dataframe.copy()
        else:
            rescol = self.rescol
            hue_column = 'Censored'
            hue_order = [True, False]
            df = self.raw_data.copy()

        plot = (
            df.assign(pos=pos)
            .assign(Result=lambda df: df[rescol])
            .assign(Censored=lambda df: df[self.cencol])
            .pipe(
                (seaborn.swarmplot, 'data'),
                y='Result',
                x='pos',
                hue=hue_column,
                hue_order=hue_order,
                size=markersize,
                marker=self.plot_marker,
                color=self.color,
                edgecolor='k',
                linewidth=1.25,
            )
        )

        ax.set_yscale(yscale)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        return fig


class Dataset(object):
    """ Dataset: object for comparings two Location objects

    Parameters
    ----------
    influent, effluent : wqio.Location
        Location objects that will be compared. Data from each Location
        should be joinable on the dataframe's index.
    useros : bool (default = True)
        Toggles the use of Regression On Order Statistics to
        estimate non-detect values when computing statistics.
    name : string optional
        Name for the dataset.

    Notes
    -----
    Currently moving away from this in favor of DataCollections.

    """

    # TODO: constructor should take dataframe, and build Location object,
    # not the other way around. This will allow Dataset.influent = None
    # by passing in a dataframe where df.shape[0] == 0
    def __init__(self, influent, effluent, useros=True, name=None):

        # basic attributes
        self.influent = influent
        self.effluent = effluent
        self._name = name
        self._include = None
        self.useros = useros
        self._definition = {}
        self._cache = resettable_cache()

    @cache_readonly
    def data(self):
        if self.effluent.hasData:
            effl = self.effluent.raw_data.copy()
        else:
            raise ValueError("effluent must have data")

        if self.influent.hasData:
            infl = self.influent.raw_data.copy()
        else:
            infl = pandas.DataFrame(
                index=self.effluent.raw_data.index,
                columns=self.effluent.raw_data.columns
            )

        infl = utils.add_column_level(infl, 'inflow', 'station')
        effl = utils.add_column_level(effl, 'outflow', 'station')
        return infl.join(effl, how='outer')

    @cache_readonly
    def paired_data(self):
        if self.data is not None:
            return self.data.dropna()

    @cache_readonly
    def n_pairs(self):
        if self.paired_data is not None:
            return self.paired_data.shape[0]
        else:
            return 0

    @cache_readonly
    def _non_paired_stats(self):
        return self.influent.data is not None and self.effluent.data is not None

    @cache_readonly
    def _paired_stats(self):
        return self._non_paired_stats and self.paired_data.shape[0] > 20

    def __repr__(self):
        x = "<wqio.Dataset>\n  N influent  {0}\n  N effluent = {1}".format(
            self.influent.N, self.effluent.N
        )
        if self.definition is not None:
            for k, v in self.definition.items():
                x = "{0}\n  {1} = {2}".format(x, k.title(), v)
        return x

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def definition(self):
        return self._definition

    @definition.setter
    def definition(self, value):
        self._definition = value

    @property
    def include(self):
        if self._include is None:
            self._include = self.influent.include and self.effluent.include
        return self._include

    @include.setter
    def include(self, value):
        self._include = value

    @property
    def exclude(self):
        return not self.include

    # stats describing the dataset
    @cache_readonly
    def medianCIsOverlap(self):
        overlap = True
        if self.influent.hasData and self.effluent.hasData:
            overlap = utils.checkIntervalOverlap(
                self.influent.median_conf_interval,
                self.effluent.median_conf_interval,
                oneway=False
            )
        return overlap

    @cache_readonly
    def wilcoxon_z(self):
        """The Wilcoxon Z-statistic.

        Tests the null hypothesis that the influent and effluent data are
        sampled from the same statistical distribution.

        Notes
        -----
        Operates on natural log-transormed paired data only.

        See also
        --------
        scipy.stats.wilcoxon

        """
        if self._wilcoxon_stats is not None:
            return self._wilcoxon_stats[0]

    @cache_readonly
    def wilcoxon_p(self):
        """Two-sided p-value of the Wilcoxon test

        See also
        --------
        scipy.stats.wilcoxon

        """
        if self._wilcoxon_stats is not None:
            return self._wilcoxon_stats[1]

    @cache_readonly
    def mannwhitney_u(self):
        """Mann-Whitney U-statistic.

        Performs a basic rank-sum test.

        Notes
        -----
        Operates on untransformed, non-paired data.

        See also
        --------
        scipy.stats.mannwhitneyu

        """
        if self._mannwhitney_stats is not None:
            return self._mannwhitney_stats[0]

    @cache_readonly
    def mannwhitney_p(self):
        """Two-sided p-value of the Mann-Whitney test

        Notes
        -----
        Scipy functions returns the 1-sided p-values. This corrects for that.

        See also
        --------
        scipy.stats.mannwhitneyu

        """
        if self._mannwhitney_stats is not None:
            return self._mannwhitney_stats[1]

    @cache_readonly
    def kendall_tau(self):
        """The Kendall-Tau statistic.

        Measure of the correspondence between two the rankings of influent
        and effluent data.

        Notes
        -----
        Operates on paired data only.

        See also
        --------
        scipy.stats.kendalltau

        """
        if self._kendall_stats is not None:
            return self._kendall_stats[0]

    @cache_readonly
    def kendall_p(self):
        """Two-sided p-value of the Kendall test

        See also
        --------
        scipy.stats.kendalltau

        """
        if self._kendall_stats is not None:
            return self._kendall_stats[1]

    @cache_readonly
    def spearman_rho(self):
        """The Spearman's rho statistic.

        Tests for monotonicity of the relationship between influent and.
        effluent data.

        Notes
        -----
        Operates on paired data only.

        See also
        --------
        scipy.stats.spearmanr

        """
        if self._spearman_stats is not None:
            return self._spearman_stats[0]

    @cache_readonly
    def spearman_p(self):
        """Two-sided p-value of the Spearman test

        See also
        --------
        scipy.stats.spearmanr

        """
        if self._spearman_stats is not None:
            return self._spearman_stats[1]

    @cache_readonly
    def ttest_t(self):
        return self._ttest_stats[0]

    @cache_readonly
    def ttest_p(self):
        return self._ttest_stats[1]

    @cache_readonly
    def levene_ks(self):
        return self._levene_stats[0]

    @cache_readonly
    def levene_p(self):
        return self._levene_stats[1]

    @cache_readonly
    def theil_medslope(self):
        return self._theil_stats['medslope']

    @cache_readonly
    def theil_intercept(self):
        return self._theil_stats['intercept']

    @cache_readonly
    def theil_loslope(self):
        return self._theil_stats['loslope']

    @cache_readonly
    def theil_hislope(self):
        return self._theil_stats['hislope']

    # helper objects for the stats
    @cache_readonly
    def _wilcoxon_stats(self):
        if self._paired_stats:
            return stats.wilcoxon(numpy.log(self.paired_data.inflow.res),
                                  numpy.log(self.paired_data.outflow.res))

    @cache_readonly
    def _mannwhitney_stats(self):
        if self._non_paired_stats:
            return stats.mannwhitneyu(self.influent.data, self.effluent.data,
                                      alternative='two-sided')

    @cache_readonly
    def _kendall_stats(self):
        if self._paired_stats:
            return stats.kendalltau(self.paired_data.inflow.res,
                                    self.paired_data.outflow.res)

    @cache_readonly
    def _spearman_stats(self):
        if self._paired_stats:
            return stats.spearmanr(self.paired_data.inflow.res.values,
                                   self.paired_data.outflow.res.values)

    @cache_readonly
    def _ttest_stats(self):
        if self._non_paired_stats:
            return stats.ttest_ind(self.influent.data, self.effluent.data, False)

    @cache_readonly
    def _levene_stats(self):
        if self._non_paired_stats:
            return stats.levene(self.influent.data, self.effluent.data, center='median')

    @cache_readonly
    def _theil_stats(self):
        return self.theilSlopes()

    def theilSlopes(self, log_infl=False, log_effl=False):
        output = None
        # influent data
        infl = self.paired_data.inflow.res.values
        if log_infl:
            infl = numpy.log(infl)

        # effluent data
        effl = self.paired_data.outflow.res.values
        if log_effl:
            effl = numpy.log(effl)

        if self.influent.fractionND <= 0.5 and \
           self.effluent.fractionND <= 0.5:
            # we need to make sure that the "y" values are the
            # Location with the greatest NUnique to avoid a
            # slope of zero if possible
            if self.influent.NUnique <= self.effluent.NUnique:
                inverted = False
                theilstats = stats.mstats.theilslopes(effl, x=infl)
            else:
                inverted = True
                theilstats = stats.mstats.theilslopes(infl, x=effl)

            # stuff things into a dictionary
            if not inverted:
                output = {
                    'medslope': theilstats[0],
                    'intercept': theilstats[1],
                    'loslope': theilstats[2],
                    'hislope': theilstats[3],
                    'is_inverted': inverted
                }
            else:
                output = {
                    'medslope': 1 / theilstats[0],
                    'intercept': -1 * theilstats[1] / theilstats[0],
                    'loslope': 1 / theilstats[2],
                    'hislope': 1 / theilstats[3],
                    'is_inverted': inverted
                }

            output['estimated_effluent'] = \
                _estimate_from_fit(infl, output['medslope'], output['intercept'],
                                   xlog=log_infl, ylog=log_effl)

            output['estimate_error'] = self.paired_data.outflow.res.values - output['estimated_effluent']

        return output

    # plotting methods
    def boxplot(self, ax=None, pos=1, yscale='log', shownotches=True,
                showmean=True, width=0.8, bacteria=False, ylabel=None,
                xlims=None, bothTicks=True, offset=0.5,
                patch_artist=False):
        """ Adds a boxplot to a matplotlib figure

        Parameters
        ----------
        ax : matplotlib axes object or None (default), optional
            Axes on which the boxplot with be drawn. If None, one will
            be created.
        pos : int, optional (default=1)
            Location along x-axis where boxplot will be placed.
        yscale : optional string ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        shownotches : bool, optional (default=True)
            Toggles drawing of bootstrapped confidence interval around
            the median.
        showmean : bool, optional (default is True)
            Toggles plotting the mean value on the boxplot as a point.
            See also the `bacteria` kwarg
        width : float, optional (default=0.8)
            Width of boxplot on the axes (data units)
        bacteria : bool, optional (default is False)
            If True, uses the geometric mean when `showmean` is True.
            Otherwise, the arithmetic mean is used.
        ylabel : string or None (default):
            Label for y-axis
        xlims : sequence (length=2), dict, or None (default), optional
            Custom limits of the x-axis. If None, defaults to
            [pos-1, pos+1].
        bothTicks : bool, optional (default is True)
            If True, each box gets a tick label. Otherwise, both get a
            single tick.
        offset : float, optional (default = 0.5)
            Spacing, in x-axis data coordinates, of the boxplots.
        patch_artist : bool, optional (default = False)
            Toggles the use of patch artist instead of a line artists
            for the boxes

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = validate.axes(ax)

        for loc, offset in zip([self.influent, self.effluent], [-1 * offset, offset]):
            if loc is not None:
                y_log = (yscale == 'log')
                bxpstats = loc.boxplot_stats(log=y_log, bacteria=bacteria)
                bp = viz.boxplot(
                    bxpstats,
                    ax=ax,
                    position=pos + offset,
                    width=width,
                    color=loc.color,
                    marker=loc.plot_marker,
                    patch_artist=patch_artist,
                    showmean=showmean,
                    shownotches=shownotches,
                )

        ax.set_yscale(yscale)
        if y_log:
            ax.yaxis.set_major_formatter(viz.log_formatter(use_1x=False))

        if ylabel:
            ax.set_ylabel(ylabel)

        if xlims is None:
            ax.set_xlim([pos - 1, pos + 1])
        else:
            if isinstance(xlims, dict):
                ax.set_xlim(**xlims)
            else:
                ax.set_xlim(xlims)

        if bothTicks:
            ax.set_xticks([pos - offset, pos + offset])
            ax.set_xticklabels([self.influent.name, self.effluent.name])
        else:
            ax.set_xticks([pos])
            if self.name is not None:
                ax.set_xticklabels([self.name])
            else:
                ax.set_xticklabels([''])

        return fig

    def probplot(self, ax=None, yscale='log', axtype='prob', ylabel=None,
                 clearYLabels=False, rotateticklabels=True,
                 bestfit=False):
        """ Adds probability plots to a matplotlib figure

        Parameters
        ----------
        ax : matplotlib axes object, optional or None (default).
            The Axes on which to plot. If None is provided, one will be
            created.
        yscale : string , optional(default = 'log')
            Scale for the y-axis. Use 'log' for logarithmic (default) or
            'linear'.
        axtype : string, optional (default = 'pp')
            Type of plot to be created. Options are:
                - 'prob': probabilty plot
                - 'pp': percentile plot
                - 'qq': quantile plot
        ylabel : string, optional or None (default)
            Axis label for the plot.
        clearYLabels : bool, optional (default is False)
            If True, removed y-*tick* labels from teh Axes.
        rotateticklabels : bool, optional (default is True)
            If True, the tick labels of the probability axes will be
            rotated.
        bestfit : bool, optional (default is False)
            Specifies whether a best-fit line should be added to the
            plot.

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = validate.axes(ax)

        for loc in [self.influent, self.effluent]:
            if loc.include:
                loc.probplot(ax=ax, clearYLabels=clearYLabels, axtype=axtype,
                             yscale=yscale, bestfit=bestfit,
                             rotateticklabels=rotateticklabels)

        xlabels = {
            'pp': 'Theoretical percentiles',
            'qq': 'Theoretical quantiles',
            'prob': 'Non-exceedance probability (\%)'
        }

        ax.set_xlabel(xlabels[axtype])
        ax.legend(loc='lower right', frameon=True)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if rotateticklabels:
            viz.rotateTickLabels(ax, 45, 'x')

        return fig

    def statplot(self, pos=1, yscale='log', shownotches=True, showmean=True,
                 width=0.8, bacteria=False, ylabel=None, axtype='qq',
                 patch_artist=False):
        """Creates a two-axis figure with a boxplot & probability plot.

        Parameters
        ----------
        pos : int, optional (default=1)
            Location along x-axis where boxplot will be placed.
        yscale : string, optional ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        shownotches : bool, optional (default=True)
            Toggles drawing of bootstrapped confidence interval around
            the median.
        showmean : bool, optional (default=True)
            Toggles plotting the mean value on the boxplot as a point.
            See also the `bacteria` kwarg
        width : float, optional (default=0.8)
            Width of boxplot on the axes (data units)
        bacteria : bool, optional (default False)
            If True, uses the geometric mean when `showmean` is True.
            Otherwise, the arithmetic mean is used.
        ylabel : string, optional or None (default):
            Label for y-axis
        probAxis : bool, optional (default = True)
            Toggles the display of probabilities (True) or Z-scores
            (i.e., theoretical quantiles) on the x-axis
        patch_artist : bool, optional (default = False)
            Toggles the use of patch artist instead of a line artists
            for the boxes

        Returns
        -------
        fig : matplotlib Figure

        """

        # setup the figure and axes
        fig = pyplot.figure(figsize=(6.40, 3.00), facecolor='none',
                            edgecolor='none')
        ax1 = pyplot.subplot2grid((1, 4), (0, 0))
        ax2 = pyplot.subplot2grid((1, 4), (0, 1), colspan=3)

        self.boxplot(ax=ax1, pos=pos, yscale=yscale, shownotches=shownotches,
                     showmean=showmean, width=width, bacteria=bacteria,
                     ylabel=ylabel, patch_artist=patch_artist)

        self.probplot(ax=ax2, yscale=yscale, axtype=axtype,
                      ylabel=None, clearYLabels=True,
                      rotateticklabels=True)

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        fig.subplots_adjust(wspace=0.05)
        return fig

    def jointplot(self, hist=False, kde=True, rug=True, **scatter_kws):
        """ Create a joint distribution plot for the dataset

        Parameters
        ----------
        hist : bool, optional (default is False)
            Toggles showing histograms on the distribution plots
        kde : bool, optional (default is True)
            Toggles showing KDE plots on the distribution plots
        run : bool, optional (default is True)
            Toggles showing rug plots on the distribution plots
        **scatter_kws : keyword arguments
            Optionals passed directly to Dataset.scatterplot

        Returns
        -------
        jg : seaborn.JointGrid

        See also
        --------
        seaborn.JointGrid
        seaborn.jointplot
        seaborn.distplot
        Dataset.scatterplot

        """

        showlegend = scatter_kws.pop('showlegend', True)
        _ = scatter_kws.pop('ax', None)

        if self.paired_data is not None:
            data = self.paired_data.xs('res', level='quantity', axis=1)
            jg = seaborn.JointGrid(x='inflow', y='outflow', data=data)
            self.scatterplot(ax=jg.ax_joint, showlegend=False, **scatter_kws)
            jg.plot_marginals(seaborn.distplot, hist=hist, rug=rug, kde=kde)

            jg.ax_marg_x.set_xscale(scatter_kws.pop('xscale', 'log'))
            jg.ax_marg_y.set_yscale(scatter_kws.pop('yscale', 'log'))

            if showlegend:
                jg.ax_joint.legend(loc='upper left')

        return jg

    def scatterplot(self, ax=None, xscale='log', yscale='log', showlegend=True,
                    xlabel=None, ylabel=None, one2one=False, useros=False,
                    bestfit=False, minpoints=3, eqn_pos='lower right',
                    equal_scales=True, fitopts=None, **markeropts):
        """ Creates an influent/effluent scatter plot

        Parameters
        ----------
        ax : matplotlib axes object or None (default), optional
            Axes on which the scatterplot with be drawn. If None, one
            will be created.
        xscale, yscale : string ['linear' or 'log' (default)], optional
            Scale formatting of the [x|y]-axis
        xlabel, ylabel : string or None (default), optional:
            Label for [x|y]-axis. If None, will be 'Influent'. If the
            dataset definition is available and incldues a Parameter,
            that will be included as will. For no label, use
            e.g., `xlabel = ""`.
        showlegend : bool, optional (default is True)
            Toggles including a legend on the plot.
        one2one : bool, optional (default is False), optional
            Toggles the inclusion of the 1:1 line (i.e. line of
            equality).
        useros : bool, optional (default is False)
            Toggles the use of the ROS'd results. If False, raw results
            (i.e., detection limit for NDs) are used with varying
            symbology.

        Returns
        ------
        fig : matplotlib Figure

        """

        # set up the figure/axes
        fig, ax = validate.axes(ax)

        # set the scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # common symbology
        commonopts = dict(
            linestyle='none',
            markeredgewidth=0.5,
            markersize=6,
            zorder=10,
        )

        # plot the ROSd'd result, if requested
        if useros:
            raise ValueError

        # plot the raw results, if requested
        else:
            plot_params = [
                dict(label='Detected data pairs',
                     which='neither', marker='o', alpha=0.8,
                     markerfacecolor='black', markeredgecolor='white'),
                dict(label='Influent not detected',
                     which='influent', marker='v', alpha=0.45,
                     markerfacecolor='none', markeredgecolor='black'),
                dict(label='Effluent not detected',
                     which='effluent', marker='<', alpha=0.45,
                     markerfacecolor='none', markeredgecolor='black'),
                dict(label='Both not detected',
                     which='both', marker='d', alpha=0.45,
                     markerfacecolor='none', markeredgecolor='black'),
            ]
            for pp in plot_params:
                kwargs = {**commonopts, **pp, **markeropts}
                self._plot_nds(ax, **kwargs)

        if xscale == 'linear':
            ax.set_xlim(left=0)

        if yscale == 'linear':
            ax.set_ylim(bottom=0)

        # unify the axes limits
        if xscale == yscale and equal_scales:
            ax.set_aspect('equal')
            axis_limits = [
                numpy.min([ax.get_xlim(), ax.get_ylim()]),
                numpy.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.set_ylim(axis_limits)
            ax.set_xlim(axis_limits)

        elif yscale == 'linear' or xscale == 'linear':
            axis_limits = [
                numpy.min([numpy.min(ax.get_xlim()), numpy.min(ax.get_ylim())]),
                numpy.max([ax.get_xlim(), ax.get_ylim()])
            ]

        # include the line of equality, if requested
        if one2one:
            viz.one2one(ax, linestyle='-', linewidth=1.25,
                        alpha=0.50, color='black',
                        zorder=5, label='1:1 line')

        detects = self.paired_data.loc[
            (~self.paired_data[('inflow', 'qual')].isin(self.influent.ndvals)) &
            (~self.paired_data[('outflow', 'qual')].isin(self.effluent.ndvals))
        ].xs('res', level=1, axis=1)

        if bestfit and detects.shape[0] >= minpoints:
            if xscale == 'log' and yscale == 'log':
                fitlogs = 'both'
            elif xscale == 'log':
                fitlogs = 'x'
            elif yscale == 'log':
                fitlogs = 'y'
            else:
                fitlogs = None

            x = detects['inflow']
            y = detects['outflow']

            fitopts = validate.at_least_empty_dict(fitopts, fitlogs=fitlogs)
            xhat, yhat, modelres = utils.fit_line(x, y, **fitopts)

            ax.plot(xhat, yhat, 'k--', alpha=0.75, label='Best-fit')

            if eqn_pos is not None:
                positions = {
                    'lower left': (0.05, 0.15),
                    'lower right': (0.59, 0.15),
                    'upper left': (0.05, 0.95),
                    'upper right': (0.59, 0.95)
                }
                vert_offset = 0.1
                try:
                    txt_x, txt_y = positions.get(eqn_pos.lower())
                except KeyError:
                    raise ValueError("`eqn_pos` must be on of ".format(list.positions.keys()))
                # annotate axes with stats

                ax.annotate(
                    r'$\log(y) = {} \, \log(x) + {}$'.format(
                        utils.sigFigs(modelres.params[1], n=3),
                        utils.sigFigs(modelres.params[0], n=3)
                    ),
                    (txt_x, txt_y),
                    xycoords='axes fraction'
                )

                ax.annotate(
                    'Slope p-value: {}\nIntercept p-value: {}'.format(
                        utils.process_p_vals(modelres.pvalues[1]),
                        utils.process_p_vals(modelres.pvalues[0])
                    ),
                    (txt_x, txt_y - vert_offset),
                    xycoords='axes fraction'
                )

        # setup the axes labels
        if xlabel is None:
            xlabel = 'Influent'

        if ylabel is None:
            ylabel = 'Effluent'

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # show legend, if requested
        if showlegend:
            leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.00)
            leg.get_frame().set_alpha(0.00)
            leg.get_frame().set_edgecolor('none')

        return fig

    def _plot_nds(self, ax, which='both', label='_no_legend', **markerkwargs):
        """
        Helper function for scatter plots -- plots various combinations
        of non-detect paired data
        """
        i_nondetect = self.paired_data[('inflow', 'cen')]
        o_nondetect = self.paired_data[('outflow', 'cen')]

        i_detect = ~self.paired_data[('inflow', 'cen')]
        o_detect = ~self.paired_data[('outflow', 'cen')]

        index_combos = {
            'both': i_nondetect & o_nondetect,
            'influent': i_nondetect & o_detect,
            'effluent': i_detect & o_nondetect,
            'neither': i_detect & o_detect,
        }

        try:
            index = index_combos[which]
        except KeyError:
            msg = '`which` must be "both", "influent", ' \
                  '"effluent", or "neighter"'
            raise ValueError(msg)

        x = self.paired_data.loc[index][('inflow', 'res')]
        y = self.paired_data.loc[index][('outflow', 'res')]
        return ax.plot(x, y, label=label, **markerkwargs)
