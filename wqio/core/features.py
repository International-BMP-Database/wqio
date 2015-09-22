from __future__ import division

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas
import statsmodels.api as sm
from statsmodels.tools.decorators import (resettable_cache,
                                          cache_readonly,
                                          cache_writable)

import seaborn.apionly as seaborn
from wqio import utils
from wqio import algo

# meta data mappings based on station
station_names = {
    'inflow': 'Influent',
    'outflow': 'Effluent',
    'reference': 'Reference Flow'
}

markers = {
    'Influent': ['o', 'v'],
    'Effluent': ['s', '<'],
    'Reference Flow': ['D', 'd']
}

palette = seaborn.color_palette(palette='deep', n_colors=3, desat=0.88)
colors = {
    'Influent': palette[0],
    'Effluent': palette[1],
    'Reference Flow': palette[2]
}


def sanitizeTexParam(x):
    return x


def sanitizeTexUnit(x):
    return x


def _process_p_vals(pval):
    if pval is None:
        out = 'NA'

    elif pval < 0.001:
        out = "<0.001"
    else:
        out = '%0.3f' % pval

    return out


class Parameter(object):
    def __init__(self, name, units, usingTex=False):
        """ Class representing a single analytical parameter (pollutant).

        (Input) Parameters
        ------------------
        name : string
            Name of the parameter.
        units : string
            Units of measure for the parameter.
        usingTex : bool, optional (default = False)
            If True, all returned values will be optimized for inclusion
            in LaTeX documents.

        """

        self._name = name
        self._units = units
        self._usingTex = usingTex

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = value

    @property
    def usingTex(self):
        return self._usingTex
    @usingTex.setter
    def usingTex(self, value):
        if value in (True, False):
            self._usingTex = value
        else:
            raise ValueError("`usingTex` must be of type `bool`")

    def paramunit(self, usecomma=False):
        """ Creates a string representation of the parameter and units.

        Parameters
        ----------
        usecomma : bool, optional (default = False)
            Toggles the format of the returned string attribute. If True
            the returned format is "<parameter>, <unit>". Otherwise the
            format is "<parameter> (<unit>)".
        """

        if usecomma:
            paramunit = '{0}, {1}'
        else:
            paramunit = '{0} ({1})'

        #if self.usingTex:
        #    n = sanitizeTexParam(self.name)
        #    u = sanitizeTexUnit(self.units)
        #else:
        n = self.name
        u = self.units

        return paramunit.format(n, u)

    def __repr__(self):
        return "<wqio Parameter object> ({})".format(
            self.paramunit(usecomma=False)
        )

    def __str__(self):
        return "<wqio Parameter object> ({})".format(
            self.paramunit(usecomma=False)
        )


class DrainageArea(object):
    def __init__(self, total_area=1.0, imp_area=1.0, bmp_area=0.0):
        """ A simple object representing the drainage area of a BMP.

        Units are not enforced, so keep them consistent yourself. The
        calculations available assume that the area of the BMP and the
        "total" area are mutually exclusive. In other words,
        the watershed outlet is at the BMP inlet.

        Parameters
        ----------
        total_area : float, optional (default = 1.0)
            The total geometric area of the BMP's catchment
        imp_area : float, optional (default = 1.0)
            The impervious area of the BMP's catchment
        bmp_area : float, optional (default = 0.0)
            The geometric area of the BMP itself.

        """

        self.total_area = float(total_area)
        self.imp_area = float(imp_area)
        self.bmp_area = float(bmp_area)

    def simple_method(self, storm_depth, volume_conversion=1.0, annualFactor=1.0):
        """
        Estimate runoff volume via Bob Pitt's Simple Method.

        Parameters
        ----------
        storm_depth : float
            Depth of the storm.
        volume_conversion : float, optional (default = 1.0)
            Conversion factor to go from [area units] * [depth units] to
            the desired [volume units]. If [area] = m^2, [depth] = mm,
            and [volume] = L, then `volume_conversion` = 1.
        annualFactor : float, optional (default = 1.0)
            The Simple Method's annual correction factor to account for
            small storms that do not produce runoff.

        Returns
        -------
        runoff_volume : float
            The volume of water entering the BMP immediately downstream
            of the drainage area.

        """

        # volumetric run off coneffiecient
        Rv = 0.05 + (0.9 * (self.imp_area / self.total_area))

        # run per unit storm depth
        drainage_conversion = Rv * self.total_area * volume_conversion
        bmp_conversion = self.bmp_area * volume_conversion

        # total runoff based on actual storm depth
        runoff_volume = (drainage_conversion * annualFactor + bmp_conversion) * storm_depth
        return runoff_volume


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
    bsIter : int, optional (default = 1e4)
        Number of interations to use when using a bootstrap
        algorithm to refine a statistic.
    station_type : string, optional
        Type of location being analyzed. Valid values are:
        'inflow' (default) or 'outflow'.
    useROS : bool, optional (default = True)
        Toggles the use of Regression On Order Statistics to
        estimate non-detect values when computing statistics.
    include : bool, optional (default = True)
        Toggles the inclusion of the location when programmatically
        creating many `Location` objects.

    Settable Properties
    -------------------
    .plot_marker : string
        Matplotlib string of the marker used in plotting methods.
    .color : string
        Matplotlib color for plotting.
    .bsIter : int
        Same as input.
    .useROS : bool
        Same as input.
    .filtered_data :pandas.DataFrame
        Subset of the input data. This is the final version used in all
        computations and graphics. It is set to the full dataset by
        default.
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
    .lillifors+ : list of floats
        Lillifors statistic and p-value.
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
      For example, self.mean and self.geomean_mean.

    Plotting Methods
    ----------------
    .verticalScatter
    .boxplot
    .probplot
    .statplot

    """

    def __init__(self, dataframe, rescol='res', qualcol='qual', ndval='ND',
                 bsIter=10000, station_type='inflow', useROS=True,
                 include=True):
        # plotting symbology based on location type
        self.station_type = station_type
        self.station_name = station_names[station_type]
        self._plot_marker = markers[self.station_name][0]
        self.scatter_marker = markers[self.station_name][1]
        self._color = colors[self.station_name]

        # basic stuff
        self._name = self.station_name
        self._include = include
        self._definition = {}

        # parameters of the stats analysis
        self._cache = resettable_cache()

        # properties of the dataframe and analysis
        self._bsIter = bsIter
        self._useROS = useROS
        self._rescol = rescol
        self._qualcol = qualcol
        self._ndval = ndval

        # original data and quantity
        self._raw_data = dataframe
        self._filtered_data = self._raw_data.copy()

    @property
    def color(self):
        return self._color
    @color.setter
    def color(self, value):
        self._color = value

    @property
    def plot_marker(self):
        return self._plot_marker
    @plot_marker.setter
    def plot_marker(self, value):
        self._plot_marker = value

    @property
    def bsIter(self):
        return self._bsIter
    @bsIter.setter
    def bsIter(self, value):
        self._bsIter = value
        self._cache.clear()

    @property
    def useROS(self):
        return self._useROS
    @useROS.setter
    def useROS(self, value):
        self._cache.clear()
        self._useROS = value

    @property
    def filtered_data(self):
        if self._filtered_data is None:
            return self._raw_data
        else:
            return self._filtered_data
    @filtered_data.setter
    def filtered_data(self, value):
        self._cache.clear()
        self._filtered_data = value

    @property
    def data(self):
        if self.hasData:
            if self.useROS:
               output = self.ros.data['final_data']
            else:
                output = self.filtered_data[self._rescol]
            return output.values

    @property
    def full_data(self):
        if self.hasData:
            if self.useROS:
                output = self.ros.data[['final_data', 'qual']]
                rename_dict = {'final_data': 'res'}
            else:
                output = self.filtered_data[[self._rescol, self._qualcol]]
                rename_dict = {self._qualcol: 'qual', self._rescol: 'res'}

            return output.rename(columns=rename_dict)

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
        return self.filtered_data.shape[0]

    @cache_readonly
    def hasData(self):
        if self.N == 0:
            return False
        else:
            return True

    @cache_readonly
    def all_positive(self):
        if self.hasData:
            return self.min > 0

    @cache_readonly
    def ND(self):
        return (self.filtered_data[self._qualcol] == self._ndval).sum()

    @cache_readonly
    def NUnique(self):
        return pandas.unique(self.data).shape[0]

    @cache_readonly
    def fractionND(self):
        return self.ND/self.N

    @cache_readonly
    def ros(self):
        if self.hasData:
            return algo.ros.MR(self.filtered_data, rescol=self._rescol, qualcol=self._qualcol)

    @cache_readonly
    @np.deprecate
    def pnorm(self):
        if self.hasData:
            return stats.shapiro(self.data)[1]

    @cache_readonly
    @np.deprecate
    def plognorm(self):
        if self.hasData:
            return stats.shapiro(np.log(self.data))[1]

    @cache_readonly
    def shapiro(self):
        if self.hasData:
            return  stats.shapiro(self.data)

    @cache_readonly
    def shapiro_log(self):
        if self.hasData:
            return stats.shapiro(np.log(self.data))

    @cache_readonly
    def lilliefors(self):
        if self.hasData:
            return sm.stats.lillifors(self.data)

    @cache_readonly
    def lilliefors_log(self):
        if self.hasData:
            return sm.stats.lillifors(np.log(self.data))

    @cache_readonly
    def anderson(self):
        if self.hasData:
            return  stats.anderson(self.data)

    @cache_readonly
    def anderson_log(self):
        if self.hasData:
            return stats.anderson(np.log(self.data))

    @cache_readonly
    def analysis_space(self):
        if self.plognorm >= self.pnorm and self.plognorm > 0.1:
            return 'lognormal'
        else:
            return 'normal'

    @cache_readonly
    def cov(self):
        if self.hasData:
           return self.data.std()/self.data.mean()

    @cache_readonly
    def min(self):
        if self.hasData:
            return self.data.min()

    @cache_readonly
    def min_detect(self):
        if self.hasData:
            return self._filtered_data[self._rescol][self._filtered_data[self._qualcol] != self._ndval].min()

    @cache_readonly
    def min_DL(self):
        if self.hasData:
            return self._filtered_data[self._rescol][self._filtered_data[self._qualcol] == self._ndval].min()

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
            return stats.scoreatpercentile(self.data, 10)

    @cache_readonly
    def pctl25(self):
        if self.hasData:
            return stats.scoreatpercentile(self.data, 25)

    @cache_readonly
    def pctl75(self):
        if self.hasData:
            return stats.scoreatpercentile(self.data, 75)

    @cache_readonly
    def pctl90(self):
        if self.hasData:
            return stats.scoreatpercentile(self.data, 90)

    # stats that we need
    @cache_readonly
    def median(self):
        if self.hasData:
            return self._median_boostrap[0]

    @cache_readonly
    def median_conf_interval(self):
        if self.hasData:
            return self._median_boostrap[1]

    @cache_readonly
    def mean(self):
        if self.hasData:
            return self._mean_boostrap[0]

    @cache_readonly
    def mean_conf_interval(self):
        if self.hasData:
            return self._mean_boostrap[1]

    @cache_readonly
    def std(self):
        if self.hasData:
            return np.std(self.data)

    @cache_readonly
    def logmean(self):
        if self.all_positive and self.hasData:
            return np.mean(np.log(self.data))

    @cache_readonly
    def logmean_conf_interval(self):
        if self.all_positive and self.hasData:
            return self._logmean_boostrap[1]

    @cache_readonly
    def logstd(self):
        if self.all_positive and self.hasData:
            return np.std(np.log(self.data))

    @cache_readonly
    def geomean(self):
        if self.all_positive and self.hasData:
            return np.exp(self.logmean)

    @cache_readonly
    def geomean_conf_interval(self):
        if self.all_positive and self.hasData:
            return np.exp(self.logmean_conf_interval)

    @cache_readonly
    def geostd(self):
        if self.all_positive and self.hasData:
            return np.exp(self.logstd)
        else:
            return None

    # helper bootstrap objects
    @cache_readonly
    def _median_boostrap(self):
        if self.hasData:
            return algo.bootstrap.Stat(self.data, np.median, NIter=self.bsIter).BCA()

    @cache_readonly
    def _mean_boostrap(self):
        if self.hasData:
            return algo.bootstrap.Stat(self.data, np.mean, NIter=self.bsIter).BCA()

    @cache_readonly
    def _std_boostrap(self):
        if self.hasData:
            return algo.bootstrap.Stat(self.data, np.std, NIter=self.bsIter).BCA()

    @cache_readonly
    def _logmean_boostrap(self):
        if self.all_positive and self.hasData:
            return algo.bootstrap.Stat(np.log(self.data), np.mean, NIter=self.bsIter).BCA()

    @cache_readonly
    def _logstd_boostrap(self):
        if self.all_positive and self.hasData:
            return algo.bootstrap.Stat(np.log(self.data), np.std, NIter=self.bsIter).BCA()

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
            wnf = utils.whiskers_and_fliers(
                np.log(self.data),
                np.log(self.pctl25),
                np.log(self.pctl75),
                transformout=np.exp
            )
        else:
            wnf = utils.whiskers_and_fliers(
                self.data,
                self.pctl25,
                self.pctl75,
                transformout=None
            )

        bxpstats.update(wnf)
        return [bxpstats]

    # plotting methods
    def boxplot(self, ax=None, pos=1, yscale='log', notch=True, showmean=True,
                width=0.8, bacteria=False, ylabel=None, minpoints=5,
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
        notch : optional bool (default=True)
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
        minpoints : int, optional (default = 5)
            The minimum number of data points required to draw a
            boxplot. If too few points are present, the drawing will
            fallback to a verticalScatter plot.
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

        fig, ax = utils.figutils._check_ax(ax)

        meankwargs = dict(
            marker=self.plot_marker, markersize=5,
            markerfacecolor=self.color, markeredgecolor='Black'
        )

        if self.N >= minpoints:
            bxpstats = self.boxplot_stats(log=yscale=='log', bacteria=bacteria)
            bp = ax.bxp(bxpstats, positions=[pos], widths=width,
                        showmeans=showmean, shownotches=notch, showcaps=False,
                        manage_xticks=False, patch_artist=patch_artist,
                        meanprops=meankwargs)

            utils.figutils.formatBoxplot(bp, color=self.color, marker=self.plot_marker,
                                         patch_artist=patch_artist)
        else:
            self.verticalScatter(ax=ax, pos=pos, jitter=width*0.5, alpha=0.75,
                                 ylabel=ylabel, yscale=yscale, ignoreROS=True)

        ax.set_yscale(yscale)
        label_format = mticker.FuncFormatter(utils.figutils.alt_logLabelFormatter)
        if yscale == 'log':
            ax.yaxis.set_major_formatter(label_format)
        utils.figutils.gridlines(ax, yminor=True)

        if ylabel:
            ax.set_ylabel(ylabel)

        ax.set_xticks([pos])
        if self.name is not None:
            ax.set_xticklabels([self.name])

        if xlims is not None:
            ax.set_xlim(**xlims)

        return fig

    def probplot(self, ax=None, yscale='log', axtype='prob',
                 ylabel=None, clearYLabels=False, managegrid=True,
                 rotateticklabels=True, setxlimits=True, bestfit=False,
                 **plotopts):
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
        managegrid : bool, optional (default is True)
            If True, typical gridlines will be added to the Axes.
        rotateticklabels : bool, optional (default is True)
            If True, the tick labels of the probability axes will be
            rotated.
        setxlimits : bool, optional (default is True)
            If True and `axtype` == 'prob', tehe axis limits will be
            automatically set based on the number of points in the plot.
        bestfit : bool, optional (default is False)
            Specifies whether a best-fit line should be added to the
            plot.
        plotopts : keyword arguments
            Additional options passed directly to `plt.plot` when
            drawing the data points.

        Returns
        -------
        fig : matplotlib.Figure instance

        """

        fig, ax = utils.figutils._check_ax(ax)

        scatter_kws = plotopts.copy()
        scatter_kws['color'] = plotopts.get('color', self.color)
        scatter_kws['label'] = plotopts.get('label', self.name)
        scatter_kws['marker'] = plotopts.get('marker', self.plot_marker)
        scatter_kws['linestyle'] = plotopts.get('linestyle', 'none')
        fig = utils.figutils.probplot(self.data, ax=ax, axtype=axtype, yscale=yscale,
                                      bestfit=bestfit, scatter_kws=scatter_kws)

        if yscale == 'log':
            pass
            # label_format = mticker.FuncFormatter(
            #     utils.figutils.alt_logLabelFormatter
            # )
            # ax.yaxis.set_major_formatter(label_format)

        if managegrid:
            utils.figutils.gridlines(ax, yminor=True)

        if clearYLabels:
            ax.set_yticklabels([])

        if rotateticklabels:
            utils.figutils.rotateTickLabels(ax, 45, 'x', ha='right')

        if setxlimits and axtype == 'prob':
            utils.figutils.setProbLimits(ax, self.N, 'x')

        if bestfit:
            utils.fit_line()

        return fig

    def statplot(self, pos=1, yscale='log', notch=True, showmean=True,
                 width=0.8, bacteria=False, ylabel=None, axtype='prob',
                 patch_artist=False, **plotopts):
        """ Creates a two-axis figure with a boxplot & probability plot.

        Parameters
        ----------
        pos : int, optional (default=1)
            Location along x-axis where boxplot will be placed.
        yscale : string, optional ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        notch : bool, optional (default=True)
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
        axtype : string, optional (default = 'pp')
            Type of probability plot to be created. Options are:
                - 'prob': probabilty plot
                - 'pp': percentile plot
                - 'qq': quantile plot
        patch_artist : bool, optional (default = False)
            Toggles the use of patch artist instead of a line artists
            for the boxes
        plotopts : keyword arguments
            Additional options passed directly to `plt.plot` when
            drawing the data points on the probability plots.

        Returns
        -------
        fig : matplotlib Figure

        """

        # setup the figure and axes
        fig = plt.figure(figsize=(6.40, 3.00), facecolor='none',
                         edgecolor='none')
        ax1 = plt.subplot2grid((1, 4), (0, 0))
        ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

        self.boxplot(ax=ax1, pos=pos, yscale=yscale, notch=notch,
                     showmean=showmean, width=width, bacteria=bacteria,
                     ylabel=ylabel, patch_artist=patch_artist,
                     xlims={'left': pos - (0.6*width), 'right': pos + (0.6*width)})

        self.probplot(ax=ax2, yscale=yscale, axtype=axtype,
                      ylabel=None, clearYLabels=True, **plotopts)

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        return fig

    def verticalScatter(self, ax=None, pos=1, jitter=0.4, alpha=0.75,
                        ylabel=None, yscale='log', ignoreROS=True,
                        markersize=4, xlims=None):
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
            and `self.useROS` is `True`, this function will plot the
            ROS'd data with a single marker.
        markersize : int, optional (default = 6)
            Size of data markers on the figure in points.
        xlims : dict, optional
            Dictionary of limits for the x-axis. Keys must be either
            "left", "right", or both

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = utils.figutils._check_ax(ax)


        if jitter == 0:
            xvals = [pos] * self.N
        else:
            low = pos - jitter * 0.5
            high = pos + jitter * 0.5
            xvals = np.random.uniform(low=low, high=high, size=self.N)

        if not ignoreROS and self.useROS:
            ax.plot(xvals, self.data, marker=self.plot_marker, markersize=markersize,
                    markerfacecolor=self.color, markeredgecolor='white',
                    linestyle='none', alpha=alpha)
        else:
            y_nondet = self.filtered_data[self._rescol][self.filtered_data[self._qualcol]==self._ndval]
            y_detect = self.filtered_data[self._rescol][self.filtered_data[self._qualcol]!=self._ndval]

            ax.plot(xvals[:self.ND], y_nondet, marker='v', markersize=markersize,
                    markerfacecolor='none', markeredgecolor=self.color,
                    linestyle='none', alpha=alpha, label='Non-detects')

            ax.plot(xvals[self.ND:], y_detect, marker=self.plot_marker, markersize=markersize,
                    markerfacecolor=self.color, markeredgecolor='white',
                    linestyle='none', alpha=alpha, label='Detects')

        ax.set_yscale(yscale)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if xlims is not None:
            ax.set_xlim(**xlims)

        return fig

    # other methods
    def applyFilter(self, filterfxn, **fxnkwargs):
        """ Filter the dataset and set the `include`/`exclude`
        properties based on a user-defined function.

        Warning
        -------
        This is very advanced functionality. Proceed with caution and
        ask for help.

        Parameters
        ----------
        filterfxn : callable
            A user defined function that filters the data and determines
            if the `include` attribute should be True or False. It *must*
            return a pandas.DataFrame and a bool (in that order), or an
            error will be raised. The `filterfxn` must accept the
            `Location.data` attribute as its first argument.

        **fxnkwargs : keyword arguments
            Optional named arguments pass to `filterfxn`.

        Returns
        -------
        None

        """

        newdata, include = filterfxn(self.full_data, **fxnkwargs)
        if not isinstance(newdata, pandas.DataFrame):
            raise ValueError ('first item returned by filterfxn must be a pandas.DataFrame')

        if not isinstance(include, bool):
            raise ValueError ('second item returned by filterfxn must be a bool')

        self.filtered_data = newdata
        self.include = include


class Dataset(object):
    """ Dataset: object for comparings two Location objects

    Parameters
    ----------
    influent, effluent : wqio.Location
        Location objects that will be compared. Data from each Location
        should be joinable on the dataframe's index.
    useROS : bool (default = True)
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
    def __init__(self, influent, effluent, useROS=True, name=None):

        # basic attributes
        self.influent = influent
        self.effluent = effluent
        self._name = name
        self._include = None
        self._useROS = useROS
        self._definition = {}
        self._cache = resettable_cache()

    @property
    def useROS(self):
        return self._useROS
    @useROS.setter
    def useROS(self, value):
        self._cache.clear()
        self.influent.useROS = value
        self.effluent.useROS = value
        self._useROS = value

    @cache_readonly
    def data(self):
        if self.effluent.hasData:
            effl = self.effluent._raw_data.copy()
        else:
            raise ValueError("effluent must have data")

        if self.influent.hasData:
            infl = self.influent._raw_data.copy()
        else:
            infl = pandas.DataFrame(
                index=self.effluent._raw_data.index,
                columns=self.effluent._raw_data.columns
            )

        infl = utils.addSecondColumnLevel('inflow', 'station', infl)
        effl = utils.addSecondColumnLevel('outflow', 'station', effl)
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
        x = "<openpybmp Dataset object>\n  N influent  {0}\n  N effluent = {1}".format(self.influent.N, self.effluent.N)
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
        '''The Wilcoxon Z-statistic.

        Tests the null hypothesis that the influent and effluent data are
        sampled from the same statistical distribution.

        Notes
        -----
        Operates on natural log-transormed paired data only.

        See also
        --------
        scipy.stats.wilcoxon

        '''
        if self._wilcoxon_stats is not None:
            return self._wilcoxon_stats[0]

    @cache_readonly
    def wilcoxon_p(self):
        '''Two-sided p-value of the Wilcoxon test

        See also
        --------
        scipy.stats.wilcoxon

        '''
        if self._wilcoxon_stats is not None:
            return self._wilcoxon_stats[1]

    @cache_readonly
    def mannwhitney_u(self):
        '''Mann-Whitney U-statistic.

        Performs a basic rank-sum test.

        Notes
        -----
        Operates on untransformed, non-paired data.

        See also
        --------
        scipy.stats.mannwhitneyu

        '''
        if self._mannwhitney_stats is not None:
            return self._mannwhitney_stats[0]

    @cache_readonly
    def mannwhitney_p(self):
        '''Two-sided p-value of the Mann-Whitney test

        Notes
        -----
        Scipy functions returns the 1-sided p-values. This corrects for that.

        See also
        --------
        scipy.stats.mannwhitneyu

        '''
        if self._mannwhitney_stats is not None:
            return self._mannwhitney_stats[1] * 2.0

    @cache_readonly
    def kendall_tau(self):
        '''The Kendall-Tau statistic.

        Measure of the correspondence between two the rankings of influent
        and effluent data.

        Notes
        -----
        Operates on paired data only.

        See also
        --------
        scipy.stats.kendalltau

        '''
        if self._kendall_stats is not None:
            return self._kendall_stats[0]

    @cache_readonly
    def kendall_p(self):
        '''Two-sided p-value of the Kendall test

        See also
        --------
        scipy.stats.kendalltau

        '''
        if self._kendall_stats is not None:
            return self._kendall_stats[1]

    @cache_readonly
    def spearman_rho(self):
        '''The Spearman's rho statistic.

        Tests for monotonicity of the relationship between influent and.
        effluent data.

        Notes
        -----
        Operates on paired data only.

        See also
        --------
        scipy.stats.spearmanr

        '''
        if self._spearman_stats is not None:
            return self._spearman_stats[0]

    @cache_readonly
    def spearman_p(self):
        '''Two-sided p-value of the Spearman test

        See also
        --------
        scipy.stats.spearmanr

        '''
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
            return stats.wilcoxon(np.log(self.paired_data.inflow.res),
                                  np.log(self.paired_data.outflow.res))

    @cache_readonly
    def _mannwhitney_stats(self):
        if self._non_paired_stats:
            return stats.mannwhitneyu(self.influent.data,
                                      self.effluent.data)

    @cache_readonly
    def _kendall_stats(self):
        if self._paired_stats and \
        self.influent.fractionND <= 0.5 and \
        self.effluent.fractionND <= 0.5:
            return stats.kendalltau(self.paired_data.inflow.res,
                                    self.paired_data.outflow.res)

    @cache_readonly
    def _spearman_stats(self):
        if self._paired_stats and \
        self.influent.fractionND <= 0.5 and \
        self.effluent.fractionND <= 0.5:
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
        output = None #default
        # influent data
        infl = self.paired_data.inflow.res.values
        if log_infl:
            infl = np.log(infl)

        # effluent data
        effl = self.paired_data.outflow.res.values
        if log_effl:
            effl = np.log(effl)

        if self._paired_stats and \
        self.influent.fractionND <= 0.5 and \
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
                utils.estimateFromLineParams(infl, output['medslope'], output['intercept'],
                                             xlog=log_infl, ylog=log_effl)

            output['estimate_error'] = self.paired_data.outflow.res.values - \
                                       output['estimated_effluent']

        return output

    # plotting methods
    def boxplot(self, ax=None, pos=1, yscale='log', notch=True,
                showmean=True, width=0.8, bacteria=False, ylabel=None,
                xlims=None, bothTicks=True, offset=0.5,
                patch_artist=False, minpoints=5):
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
        notch : bool, optional (default=True)
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
        minpoints : int, optional (default = 5)
            The minimum number of data points required to draw a
            boxplot. If too few points are present, the drawing will
            fallback to a verticalScatter plot.

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = utils.figutils._check_ax(ax)

        for loc, offset in zip([self.influent, self.effluent], [-1*offset, offset]):
            if loc.include:
                loc.boxplot(ax=ax, pos=pos+offset, notch=notch, showmean=showmean,
                            width=width/2, bacteria=bacteria, ylabel=None,
                            patch_artist=patch_artist, minpoints=minpoints)

        ax.set_yscale(yscale)
        label_format = mticker.FuncFormatter(utils.figutils.alt_logLabelFormatter)
        if yscale == 'log':
            ax.yaxis.set_major_formatter(label_format)
        utils.figutils.gridlines(ax, yminor=True)

        if ylabel:
            ax.set_ylabel(ylabel)

        if xlims is None:
            ax.set_xlim([pos-1, pos+1])
        else:
            if isinstance(xlims, dict):
                ax.set_xlim(**xlims)
            else:
                ax.set_xlim(xlims)

        if bothTicks:
            ax.set_xticks([pos-offset, pos+offset])
            ax.set_xticklabels([self.influent.name, self.effluent.name])
        else:
            ax.set_xticks([pos])
            if self.name is not None:
                ax.set_xticklabels([self.name])
            else:
                ax.set_xticklabels([''])

        ax.xaxis.grid(False, which='major')

        return fig

    def probplot(self, ax=None, yscale='log', axtype='prob', ylabel=None,
                 clearYLabels=False, rotateticklabels=True,
                 setxlimits=True, managegrid=True, bestfit=False):
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
        setxlimits : bool, optional (default is True)
            If True and `axtype` == 'prob', tehe axis limits will be
            automatically set based on the number of points in the plot.
        managegrid : bool, optional (default is True)
            If True, typical gridlines will be added to the Axes.
        bestfit : bool, optional (default is False)
            Specifies whether a best-fit line should be added to the
            plot.

        Returns
        -------
        fig : matplotlib Figure

        """

        fig, ax = utils.figutils._check_ax(ax)

        for loc in [self.influent, self.effluent]:
            if loc.include:
                loc.probplot(ax=ax, clearYLabels=clearYLabels, axtype=axtype,
                             yscale=yscale, managegrid=managegrid, bestfit=bestfit,
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
            utils.figutils.rotateTickLabels(ax, 45, 'x')

        if setxlimits and axtype == 'prob':
            N = np.max([self.influent.N, self.effluent.N])
            utils.figutils.setProbLimits(ax, N, 'x')

        if managegrid:
            utils.figutils.gridlines(ax, yminor=True)

        return fig

    def statplot(self, pos=1, yscale='log', notch=True, showmean=True,
                 width=0.8, bacteria=False, ylabel=None, axtype='qq',
                 patch_artist=False):
        """Creates a two-axis figure with a boxplot & probability plot.

        Parameters
        ----------
        pos : int, optional (default=1)
            Location along x-axis where boxplot will be placed.
        yscale : string, optional ['linear' or 'log' (default)]
            Scale formatting of the y-axis
        notch : bool, optional (default=True)
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
        fig = plt.figure(figsize=(6.40, 3.00), facecolor='none',
                         edgecolor='none')
        ax1 = plt.subplot2grid((1, 4), (0, 0))
        ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

        self.boxplot(ax=ax1, pos=pos, yscale=yscale, notch=notch,
                     showmean=showmean, width=width, bacteria=bacteria,
                     ylabel=ylabel, patch_artist=patch_artist)

        self.probplot(ax=ax2, yscale=yscale, axtype=axtype,
                      ylabel=None, clearYLabels=True,
                      rotateticklabels=True, setxlimits=True)

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        fig.tight_layout()
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
                    xlabel=None, ylabel=None, one2one=False, useROS=False,
                    bestfit=False, minpoints=3, eqn_pos='lower right',
                    equal_scales=True):
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
        useROS : bool, optional (default is False)
            Toggles the use of the ROS'd results. If False, raw results
            (i.e., detection limit for NDs) are used with varying
            symbology.

        Returns
        ------
        fig : matplotlib Figure

        """

        # set up the figure/axes
        fig, ax = utils.figutils._check_ax(ax)

        # set the scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # common symbology
        markerkwargs = dict(
            linestyle='none',
            markeredgewidth=0.5,
            markersize=6,
            zorder=10,
        )

        # plot the ROSd'd result, if requested
        if useROS:
            x = self.paired_data.inflow.res
            y = self.paired_data.outflow.res
            ax.plot(x, y, marker='o', label="Paired, ROS'd data",  alpha=0.3, **markerkwargs)

        # plot the raw results, if requested
        else:
            self._plot_nds(ax, which='neither', marker='o', alpha=0.8,
                           label='Detected data pairs', markerfacecolor='black',
                           markeredgecolor='white', **markerkwargs)
            self._plot_nds(ax, which='influent', marker='v', alpha=0.45,
                           label='Influent not detected',  markerfacecolor='none',
                           markeredgecolor='black', **markerkwargs)
            self._plot_nds(ax, which='effluent', marker='<', alpha=0.45,
                           label='Effluent not detected',  markerfacecolor='none',
                           markeredgecolor='black', **markerkwargs)
            self._plot_nds(ax, which='both', marker='d', alpha=0.25,
                           label='Both not detected',  markerfacecolor='none',
                           markeredgecolor='black', **markerkwargs)

        label_format = mticker.FuncFormatter(utils.figutils.alt_logLabelFormatter)
        if xscale != 'log':
            ax.set_xlim(left=0)

        if yscale != 'log':
            ax.set_ylim(bottom=0)

        # unify the axes limits
        if xscale == yscale and equal_scales:
            ax.set_aspect('equal')
            axis_limits = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.set_ylim(axis_limits)
            ax.set_xlim(axis_limits)

        elif yscale == 'linear' or xscale == 'linear':
            axis_limits = [
                np.min([np.min(ax.get_xlim()), np.min(ax.get_ylim())]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]

        # include the line of equality, if requested
        if one2one:
            ax.plot(axis_limits, axis_limits, linestyle='-', linewidth=1.25,
                    alpha=0.50, color='black', zorder=5, label='1:1 line')

        detects = self.paired_data.loc[
            (self.paired_data[('inflow', 'qual')] != self.influent._ndval) &
            (self.paired_data[('outflow', 'qual')] != self.effluent._ndval)
        ].xs('res', level='quantity', axis=1)
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

            xhat, yhat, modelres = utils.fit_line(x, y, fitlogs=fitlogs)

            ax.plot(xhat, yhat, 'k--', alpha=0.75, label='Best-fit')

            if eqn_pos is not None:
                positions = {
                    'lower left': (0.05, 0.15),
                    'lower right': (0.59, 0.15),
                    'upper left': (0.05, 0.95),
                    'upper right': (0.59, 0.95)
                }
                vert_offset = 0.05
                try:
                    txt_x, txt_y = positions.get(eqn_pos.lower())
                except KeyError:
                    raise ValueError("`eqn_pos` must be on of ".format(list.positions.keys()))
                # annotate axes with stats
                ax.annotate(
                    r'$\log(y) = m \, \log(x) + b$',
                    (txt_x, txt_y),
                    xycoords='axes fraction'
                )
                ax.annotate(
                    r'Slope, $ m = %0.3f $' % modelres.params['inflow'],
                    (txt_x, txt_y - vert_offset),
                    xycoords='axes fraction'
                )
                #ax.annotate(r'P-value, $ p = %s $' % fmt_p, (0.59, 0.05), xycoords='axes fraction')


        # setup the axes labels
        if xlabel is None:
            xlabel = 'Influent'

        if ylabel is None:
            ylabel = 'Effluent'

        # create major/minor gridlines and apply the axes labels
        utils.figutils.gridlines(ax, xlabel=xlabel, ylabel=ylabel)

        # show legend, if requested
        if showlegend:
            leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.00)
            leg.get_frame().set_alpha(0.00)
            leg.get_frame().set_edgecolor('none')

        fig.tight_layout()
        return fig

    def _plot_nds(self, ax, which='both', label='_no_legend', **markerkwargs):
        '''
        Helper function for scatter plots -- plots various combinations
        of non-detect paired data
        '''
        # store the original useROS value
        use_ros_cache = self.useROS


        if which == 'both':
            index = (self.paired_data[('inflow', 'qual')] == self.influent._ndval) & \
                    (self.paired_data[('outflow', 'qual')] == self.effluent._ndval)

        elif which == 'influent':
            index = (self.paired_data[('inflow', 'qual')] == self.influent._ndval) & \
                    (self.paired_data[('outflow', 'qual')] != self.effluent._ndval)

        elif which == 'effluent':
            index = (self.paired_data[('inflow', 'qual')] != self.influent._ndval) & \
                    (self.paired_data[('outflow', 'qual')] == self.effluent._ndval)

        elif which == 'neither':
            index = (self.paired_data[('inflow', 'qual')] != self.influent._ndval) & \
                    (self.paired_data[('outflow', 'qual')] != self.effluent._ndval)

        else:
            msg = '`which` must be "both", "influent", ' \
                  '"effluent", or "neighter"'
            raise ValueError(msg)

        x = self.paired_data.loc[index][('inflow', 'res')]
        y = self.paired_data.loc[index][('outflow', 'res')]
        ax.plot(x, y, label=label, **markerkwargs)


class DataCollection(object):
    """ WIP: object to compare an arbitrary number of Locations

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
    useROS : bool (default = True)
        Toggles the use of Regression On Order Statistics to
        estimate non-detect values when computing statistics.
    filterfxn : callable, optional
        Function that will we passes to a pandas.Groupby object that
        will filter out groups that should not be analyzed (for
        whatever reason).
    bsIter : int
        Number of iterations the bootstrapper should use when estimating
        confidence intervals around a statistic.

    """

    def __init__(self, dataframe, rescol='res', qualcol='qual',
                 stationcol='station', paramcol='parameter', ndval='ND',
                 othergroups=None, useROS=True, filterfxn=None,
                 bsIter=10000):

        self._filterfxn = filterfxn
        self._raw_rescol = rescol
        self._cache = resettable_cache()

        self.data = dataframe
        self.useROS = useROS
        self.roscol = 'ros_' + rescol
        if self.useROS:
            self.rescol = self.roscol
        else:
            self.rescol = rescol
        self.qualcol = qualcol
        self.stationcol = stationcol
        self.paramcol = paramcol
        self.ndval = ndval
        self.bsIter = bsIter

        self.groupby = [stationcol, paramcol]
        if othergroups is not None:
            if np.isscalar(othergroups):
                othergroups = [othergroups]
            self.groupby.extend(othergroups)

        self.columns = self.groupby + [self._raw_rescol, self.qualcol]

    @property
    def filterfxn(self):
        if self._filterfxn is None:
            return lambda x: True
        else:
            return self._filterfxn
    @filterfxn.setter
    def filterfxn(self, value):
        self._cache.clear()
        self._filterfxn = value

    @cache_readonly
    def tidy(self):

        if self.useROS:
            def fxn(g):
                mr = algo.ros.MR(g, rescol=self._raw_rescol,
                                 qualcol=self.qualcol,
                                 ndsymbol=self.ndval)
                return mr.data
        else:
            def fxn(g):
                g[self.roscol] = np.nan
                return g

        _tidy = (
            self.data
                .reset_index()[self.columns]
                .groupby(by=self.groupby)
                .filter(self.filterfxn)
                .groupby(by=self.groupby)
                .apply(fxn)
                .reset_index()
                .rename(columns={'final_data': self.roscol})
                .sort(columns=self.groupby)
        )

        keep_cols = self.columns + [self.roscol]
        _tidy = _tidy[keep_cols]

        return _tidy

    @cache_readonly
    def locations(self):
        _locations = []
        groups = (
            self.data
                .groupby(level=self.groupby)
                .filter(self.filterfxn)
                .groupby(level=self.groupby)
        )
        for names, data in groups:
            loc_dict = dict(zip(self.groupby, names))
            locdata = data.copy()
            locdata.index = locdata.index.droplevel(level=self.stationcol)
            loc = Location(
                locdata, station_type=loc_dict[self.stationcol].lower(),
                rescol=self._raw_rescol, qualcol=self.qualcol,
                ndval=self.ndval, bsIter=self.bsIter, useROS=self.useROS
            )

            loc.definition = loc_dict
            _locations.append(loc)

        return _locations

    @cache_readonly
    def datasets(self):
        _datasets = []
        groupcols = list(filter(lambda g: g != self.stationcol, self.groupby))

        for names, data in self.data.groupby(level=groupcols):
            ds_dict = dict(zip(groupcols, names))

            ds_dict[self.stationcol] = 'inflow'
            infl = self.selectLocations(squeeze=True, **ds_dict)

            ds_dict[self.stationcol] = 'outflow'
            effl = self.selectLocations(squeeze=True, **ds_dict)

            ds_dict.pop(self.stationcol)
            dsname = '_'.join(names).replace(', ', '')

            ds = Dataset(infl, effl, useROS=self.useROS, name=dsname)
            ds.definition = ds_dict

            _datasets.append(ds)

        return _datasets

    @cache_readonly
    def medians(self):
        return self._generic_stat(np.median, statname='median')

    @cache_readonly
    def means(self):
        return self._generic_stat(np.mean, statname='mean')

    @cache_readonly
    def std_devs(self):
        return self._generic_stat(np.std, statname='std. dev.')

    def percentiles(self, percentile):
        return self._generic_stat(lambda x: np.percentile(x, percentile),
                                  statname='pctl {}'.format(percentile),
                                  bootstrap=False)

    @cache_readonly
    def logmean(self):
        return self._generic_stat(lambda x, axis=0: np.mean(np.log(x), axis=axis), statname='Log-mean')

    @cache_readonly
    def logstd(self):
        return self._generic_stat(lambda x, axis=0: np.std(np.log(x), axis=axis), statname='Log-std. dev.')

    @cache_readonly
    def geomean(self):
        geomean = np.exp(self.logmean)
        geomean.columns.names = ['station', 'Geo-mean']
        return geomean

    @cache_readonly
    def geostd(self):
        geostd = np.exp(self.logstd)
        geostd.columns.names = ['station', 'Geo-std. dev.']
        return geostd

    @cache_readonly
    def count(self):
        return self._generic_stat(lambda x: x.count(), bootstrap=False, statname='Count')

    def _generic_stat(self, statfxn, bootstrap=True, statname=None):
        def CIs(x):
            bs = algo.bootstrap.Stat(x[self.rescol].values, statfxn=statfxn)
            stat, (lci, uci) = bs.BCA()
            statnames = ['lower', 'stat', 'upper']
            return pandas.Series([lci, stat, uci], index=statnames)

        if bootstrap:
            stat = (
                self.tidy
                    .groupby(by=self.groupby)
                    .apply(CIs)
                    .unstack(level=self.stationcol)
            )
        else:
            stat = (
                self.tidy
                    .groupby(by=self.groupby)
                    .agg({self.rescol: statfxn})
                    .unstack(level=self.stationcol)
            )

        stat.columns = stat.columns.swaplevel(0, 1)
        if statname is not None:
            stat.columns.names = ['station', statname]
        stat.sort(axis=1, inplace=True)

        return stat

    @staticmethod
    def _filter_collection(collection, squeeze, **kwargs):
        items = collection.copy()
        for key, value in kwargs.items():
            items = [r for r in filter(lambda x: x.definition[key] == value, items)]

        if squeeze:
            if len(items) == 1:
                items = items[0]
            elif len(items) == 0:
                items = None

        return items

    def selectLocations(self, squeeze=False, **kwargs):
        locations = self._filter_collection(
            self.locations.copy(), squeeze=squeeze, **kwargs
        )
        return locations

    def selectDatasets(self, squeeze=False, **kwargs):
        datasets = self._filter_collection(
            self.datasets.copy(), squeeze=squeeze, **kwargs
        )
        return datasets

    def stat_summary(self, groupcols=None, useROS=True):
        if useROS:
            col = self.roscol
        else:
            col = self.rescol

        if groupcols is None:
            groupcols = self.groupby

        ptiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        summary = (
            self.tidy
                .groupby(by=groupcols)
                .apply(lambda g: g[col].describe(percentiles=ptiles).T)
                .unstack(level='station')
        )
        return summary

    def facet_kde(self, row='category', col='parameter', hue='station',
                  log=True):
        df = self.tidy.copy()
        if log:
            plotcol = 'Log of {}'.format(self.rescol.replace('_', ' '))
            df[plotcol] = np.log(df[self.rescol])
        else:
            pltcol = self.rescol

        fgrid = seaborn.FacetGrid(df, row=row, col=col, hue=hue,
                                  sharex=True, sharey=True,
                                  margin_titles=True,
                                  legend_out=False)
        fgrid.map(seaborn.kdeplot, plotcol, shade=True)
        return fgrid
