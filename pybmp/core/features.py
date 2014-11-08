from __future__ import division

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas
from statsmodels.tools.decorators import (resettable_cache,
                                          cache_readonly,
                                          cache_writable)

import seaborn.apionly as seaborn
from .. import utils

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

palette = seaborn.color_palette(name='deep', n_colors=3, desat=0.88)
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
    def __init__(self, name=None, units=None, usingTex=False):
        '''
        Class representing a single parameter

        Input:
            name : string
                name of the parameter

            units : string
                units of measure for the parameter

        Attributes:
            name : string
                standard name of the parameter found in the DB

            unit : string
                decently formatted combo of `name` and `units`

        Methods:
            None
        '''
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
        '''
        Creates a string representation of the parameter and units

        Input:
            usecomma : optional boot (default is False)
                Toggles the format of the `paramunit` attribute...
                If True:
                    self.paramunit = <parameter>, <unit>
                If False:
                    self.paramunit = <parameter> (<unit>)
        '''
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
        return "<openpybmp Parameter object> ({})".format(
            self.paramunit(usecomma=False)
        )

    def __str__(self):
        return "<openpybmp Parameter object> ({})".format(
            self.paramunit(usecomma=False)
        )


class DrainageArea(object):
    def __init__(self, total_area=1.0, imp_area=1.0, bmp_area=0.0):
        '''
        A simple object representing the drainage area of a BMP. Units are not
        enforced, so keep them consistent yourself. The calculations available
        assume that the are of the BMP and the "total" area are mutually
        exclusive. In other words, the watershed outlet is at the BMP inlet.

        Input:
            total_area : optional float (default=1.0)
                The total geometric area of the BMP's catchment

            imp_area : optional float (default=1.0)
                The impervious area of the BMP's catchment

            bmp_area : optional float (default=0.0)
                The geometric area of the BMP itself.

        Attributes:
            See Input section.

        Methods:
            simple_method - estimates the influent volume to the BMP based on
                storm depth.
        '''
        self.total_area = float(total_area)
        self.imp_area = float(imp_area)
        self.bmp_area = float(bmp_area)

    def simple_method(self, storm_depth, volume_conversion=1, annualFactor=1):
        '''
        Estimate runoff volume via Bob Pitt's Simple Method.

        Input:
            storm_depth : float
                Depth of the storm.

            volume_conversion : float (default = 1)
                Conversion factor to go from [area units] * [depth units] to
                the desired [volume units]. If [area] = m^2, [depth] = mm, and
                [volume] = L, then `volume_conversion` = 1.

        '''
        # volumetric run off coneffiecient
        Rv = 0.05 + (0.9 * (self.imp_area/self.total_area))

        # run per unit storm depth
        drainage_conversion = Rv * self.total_area * volume_conversion
        bmp_conversion = self.bmp_area * volume_conversion

        # total runoff based on actual storm depth
        runoff = (drainage_conversion * annualFactor + bmp_conversion) * storm_depth
        return runoff


class Location(object):
    def __init__(self, dataframe, rescol='res', qualcol='qual', ndval='ND',
                 bsIter=10000, station_type='inflow', useROS=True,
                 include=True):
        '''
        Object providing convenient access to statics for data

        Input:
            dataframe : pandas.DataFrame instance
                A dataframe that contains at least two columns: one for the
                analytical values and another for the data qualfiers. Can
                contain any type of row index, but the column index must be
                simple (i.e., not a pandas.MultiIndex).

            rescol : optional string (default = 'res')
                Name of the column in `dataframe` that contains the analytical
                values

            qualcol : optional string (default = 'qual')
                Name of the column in `dataframe` containing qualifiers.

            ndval : optional string (default = 'ND')
                The *only* value in the `qualcol` of `dataframe` that indicates
                that the corresponding value in `rescol` in non-detect

            bsIter : optional int (default = 1e4)
                Number of interations to use when using a bootstrap algorithm
                to refine a statistic.

            station_type : optional string ['inflow' (default) or 'outflow']
                Type of location being analyzed

            useROS : optional bool (default = True)
                Toggles the use of Regression On Order Statistics to estimate
                non-detect values

            include : optional bool (default = True)
                Toggles the inclusion of the location in figures generated
                from `Datasets` compruised of this location

        General Attributes:
            .station_type (string) : Same as input
            .station_name (string) : 'Influent' or 'Effluent' depending on
                `station_type`
            .plot_marker (string) : matplotlib string of the marker used in
                plotting methods
            .scatter_marker (string) : matplotlib string of the marker used
                in comparitive scatter plot methods
            .color (string) : matplotlib color for plotting
            .filtered_data (pandas.DataFrame) : full dataset with qualifiers
            .ros (utils.ros.MR) : MR object of the ROS'd values
            .data (pandas.Series) : Final data for use in stats and plotting
                based on `useROS` (no qualiers)
            .full_data (pandas.DataFrame) : Representation of `self.data`
                that maintains the qualifiers associated with each result.
            .bsIter (int) : Same as input
            .useROS (bool) : Same as input
            .include (bool) : Same as input
            .exclude (bool) : Opposite of `.include`

        Statistical Attributes:
            .N (int) : total number of results
            .ND (int) : number of non-detect results
            .NUnique (int) : number of unique result values in the data
            .fractionND (float) : fraction of data that is non-detect
            .min (float) : minimum value of the data
            .max (float) : maximum value of the data
            .min_detect (float) : minimum detected value of the data
            .min_DL (float) : minimum detection limit reported for the data
            .mean*+ (float) : bootstrapped arithmetic mean of the data
            .std*+ (float) bootstrapped standard deviation of the data
            .geomean* (float) : geometric mean
            .geostd (float) : geometric standard deviation
            .cov (float) : covariance (absolute value of std/mean)
            .skew (float) : skewness coefficient
            .median* : median of the dataset
            .pctl[10/25/75/90] (float) : percentiles of the dataset
            .pnorm (float) : results of the Shapiro-Wilks test for normality
            .plognorm (float) : results of the Shapiro-Wilks test for normality
                on log-transormed data (so test for lognormalily).
            .analysis_space (string) : Based on the results of self.pnorm and
                self.plognorm, this is either "normal" or "lognormal".

        Statistial Notes:
            * indicates that there's an accompyaning tuple of confidence
              interval. For example, self.mean and self.mean_conf_interval
            + indicatates that there's a equivalent stat for log-transormed
              data. For example, self.mean and self.log_mean (subject
              to the absense of negitive results).

        Methods (see docstrings for more info):
            .boxplot
            .probplot
            .statplot
        '''
        # plotting symbology based on location type
        self.station_type = station_type
        self.station_name = station_names[station_type]
        self.plot_marker = markers[self.station_name][0]
        self.scatter_marker = markers[self.station_name][1]
        self.color = colors[self.station_name]

        # basic stuff
        self._name = self.station_name
        self._include = include

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
            output.name = 'res'
            return output

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
            return utils.ros.MR(self.filtered_data, rescol=self._rescol, qualcol=self._qualcol)

    @cache_readonly
    def pnorm(self):
        if self.hasData:
            return stats.shapiro(self.data)[1]
        else:
            return None

    @cache_readonly
    def plognorm(self):
        if self.hasData:
            return stats.shapiro(np.log(self.data))[1]
        else:
            return None

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
            return utils.bootstrap.Stat(self.data, np.median, NIter=self.bsIter).BCA()

    @cache_readonly
    def _mean_boostrap(self):
        if self.hasData:
            return utils.bootstrap.Stat(self.data, np.mean, NIter=self.bsIter).BCA()

    @cache_readonly
    def _std_boostrap(self):
        if self.hasData:
            return utils.bootstrap.Stat(self.data, np.std, NIter=self.bsIter).BCA()

    @cache_readonly
    def _logmean_boostrap(self):
        if self.all_positive and self.hasData:
            return utils.bootstrap.Stat(np.log(self.data), np.mean, NIter=self.bsIter).BCA()

    @cache_readonly
    def _logstd_boostrap(self):
        if self.all_positive and self.hasData:
            return utils.bootstrap.Stat(np.log(self.data), np.std, NIter=self.bsIter).BCA()

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
                width=0.8, bacteria=False, ylabel=None, minPoints=5,
                patch_artist=False):
        '''Adds a boxplot to a matplotlib figure

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

        patch_artist : optional bool (default = False)
            Toggles the use of patch artist instead of a line artists
            for the boxes

        Returns
        -------
        fig : matplotlib Figure

        '''

        fig, ax = utils.figutils._check_ax(ax)

        meankwargs = dict(
            marker=self.plot_marker, markersize=5,
            markerfacecolor=self.color, markeredgecolor='Black'
        )

        if self.N >= minPoints:
            bxpstats = self.boxplot_stats(log=yscale=='log', bacteria=bacteria)
            bp = ax.bxp(bxpstats, positions=[pos], widths=width,
                        showmeans=showmean, shownotches=notch, showcaps=False,
                        manage_xticks=False, patch_artist=patch_artist,
                        meanprops=meankwargs)

            utils.figutils.formatBoxplot(bp, color=self.color, marker=self.plot_marker,
                                         patch_artist=patch_artist)
        else:
            self.verticalScatter(ax=ax, pos=pos, width=width*0.5, alpha=0.75,
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

        return fig

    def probplot(self, ax=None, yscale='log', axtype='prob',
                 ylabel=None, clearYLabels=False, managegrid=True):
        '''Adds a probability plot to a matplotlib figure

        Parameters
        ----------
        ax : optional matplotlib axes object or None (default).
            The Axes on which to plot. If None is provided, one will be
            created.
        axtype : string (default = 'pp')
            Type of plot to be created. Options are:
                - 'prob': probabilty plot
                - 'pp': percentile plot
                - 'qq': quantile plot
        color : color string or three tuple (default = 'b')
            Just needs to be any valid matplotlib color representation.
        marker : string (default = 'o')
            String representing a matplotlib marker style.
        linestyle : string (default = 'none')
            String representing a matplotlib line style. No line is shown by
            default.
        [x|y]label : string or None (default)
            Axis label for the plot.
        yscale : string (default = 'log')
            Scale for the y-axis. Use 'log' for logarithmic (default) or
            'linear'.
        clearYLabels : bool (default = False)
            If True, removed y-*tick* labels from `ax`

        Returns
        -------
        fig : matplotlib.Figure instance

        '''
        fig, ax = utils.figutils._check_ax(ax)
        fig = utils.figutils.probplot(self.data, ax=ax, axtype=axtype,
                                      yscale=yscale, ylabel=ylabel,
                                      color=self.color, label=self.name,
                                      marker=self.plot_marker)

        if yscale == 'log':
            label_format = mticker.FuncFormatter(
                utils.figutils.alt_logLabelFormatter
            )
            ax.yaxis.set_major_formatter(label_format)

        if managegrid:
            utils.figutils.gridlines(ax, yminor=True)

        if clearYLabels:
            ax.set_yticklabels([])

        return fig

    def statplot(self, pos=1, yscale='log', notch=True, showmean=True,
                 width=0.8, bacteria=False, ylabel=None, axtype='prob',
                 patch_artist=False):
        '''
        Creates a two-axis figure. Left axis has a bopxplot. Right axis
        contains a probability ot quantile plot.

        Input:
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

            probAxis : bool (default = True)
                Toggles the display of probabilities (True) or Z-scores (i.e.,
                theoretical quantiles) on the x-axis

            patch_artist : optional bool (default = False)
                Toggles the use of patch artist instead of a line artists
                for the boxes

        Returns:
            fig : matplotlib Figure instance

        '''
        # setup the figure and axes
        fig = plt.figure(figsize=(6.40, 3.00), facecolor='none',
                         edgecolor='none')
        ax1 = plt.subplot2grid((1, 4), (0, 0))
        ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

        self.boxplot(ax=ax1, pos=pos, yscale=yscale, notch=notch,
                     showmean=showmean, width=width, bacteria=bacteria,
                     ylabel=ylabel, patch_artist=patch_artist)

        self.probplot(ax=ax2, yscale=yscale, axtype=axtype,
                      ylabel=None, clearYLabels=True)

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        return fig

    def verticalScatter(self, ax=None, pos=1, width=0.8, alpha=0.75,
                        ylabel=None, yscale='log', ignoreROS=True,
                        markersize=4):
        '''
        Input:
            ax : optional matplotlib axes object or None (default)
                Axes on which the points with be drawn. If None, one will
                be created.

            pos : optional int (default=1)
                Location along x-axis where data will be centered.

            width : optional float (default=0.80)
                Width of the random x-values uniform distributed around `pos`

            alpha : optional float (default=0.75)
                Opacity of the marker (1.0 -> opaque; 0.0 -> transparent)

            yscale : optional string ['linear' or 'log' (default)]
                Scale formatting of the y-axis

            ylabel : optional string or None (default):
                Label for y-axis

            ignoreROS : optional bool (default = True)
                By default, this function will plot the original, non-ROS'd
                data with a different symbol for non-detects. If `True`, the
                and `self.useROS` is `True`, this function will plot the ROS'd
                data with a single marker.

            markersize : option int (default = 6)
                Size of data markers on the figure in points.

        Returns:
            fig : matplotlib Figure instance
        '''
        fig, ax = utils.figutils._check_ax(ax)

        low = pos - width*0.5
        high = pos + width*0.5

        if not ignoreROS and self.useROS:
            xvals = np.random.uniform(size=self.N, low=low, high=high)
            ax.plot(xvals, self.data, marker=self.plot_marker, markersize=markersize,
                    markerfacecolor=self.color, markeredgecolor='white',
                    linestyle='none', alpha=alpha)
        else:
            x_nondet = np.random.uniform(size=self.ND, low=low, high=high)
            y_nondet = self.filtered_data[self._rescol][self.filtered_data[self._qualcol]==self._ndval]
            x_detect = np.random.uniform(size=self.N-self.ND, low=low, high=high)
            y_detect = self.filtered_data[self._rescol][self.filtered_data[self._qualcol]!=self._ndval]

            ax.plot(x_detect, y_detect, marker=self.plot_marker, markersize=markersize,
                    markerfacecolor=self.color, markeredgecolor='white',
                    linestyle='none', alpha=alpha, label='Detects')

            ax.plot(x_nondet, y_nondet, marker='v', markersize=markersize,
                    markerfacecolor='none', markeredgecolor=self.color,
                    linestyle='none', alpha=alpha, label='Non-detects')

        ax.set_yscale(yscale)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        return fig

    def _plot_nds(self, ax, which='both', label='_no_legend', **markerkwargs):
        '''
        Helper function for scatter plots -- plots various combinations
        of non-detect paired data
        '''
        if which == 'both':
            index = (self.paired_data[('inflow', 'qual')] == 'ND') & \
                    (self.paired_data[('outflow', 'qual')] == 'ND')

        elif which == 'influent':
            index = (self.paired_data[('inflow', 'qual')] == 'ND') & \
                    (self.paired_data[('outflow', 'qual')] != 'ND')

        elif which == 'effluent':
            index = (self.paired_data[('inflow', 'qual')] != 'ND') & \
                    (self.paired_data[('outflow', 'qual')] == 'ND')

        elif which == 'neither':
            index = (self.paired_data[('inflow', 'qual')] != 'ND') & \
                    (self.paired_data[('outflow', 'qual')] != 'ND')

        else:
            msg = '`which` must be "both", "influent", ' \
                  '"effluent", or "neighter"'
            raise ValueError(msg)

        x = self.paired_data.loc[index][('inflow', 'res')]
        y = self.paired_data.loc[index][('outflow', 'res')]
        ax.plot(x, y, label=label, **markerkwargs)

    # other methods
    def applyFilter(self, filterfxn, **fxnkwargs):
        '''
        Filter the dataset and set the `include`/`exclude` attributes based on
        a user-defined function. Warning: this is very advanced functionality.
        Proceed with caution and ask for help.

        Input:
            filterfxn : function
                A user defined function that filters the data and determines
                if the `include` attribute should be True or False. It *must*
                return a pandas.DataFrame and a bool (in that order), or an
                error will be raised. The `filterfxn` must accept the
                `Location.data` attribute as its first argument.

            **fxnkwargs : optional named arguments pass to `filterfxn`

        '''
        newdata, include = filterfxn(self.full_data, **fxnkwargs)
        if not isinstance(newdata, pandas.DataFrame):
            raise ValueError ('first item returned by filterfxn must be a pandas.DataFrame')

        if not isinstance(include, bool):
            raise ValueError ('second item returned by filterfxn must be a bool')

        self.filtered_data = newdata
        self.include = include


class Dataset(object):
    # TODO: constructor should take dataframe, and build Location object,
    # not the other way around. This will allow Dataset.influent = None
    # by passing in a dataframe where df.shape[0] == 0
    def __init__(self, influent, effluent, useROS=True, name=None, xlsDataDumpFile=None):

        ## TODO 2013-11-12: need to use useROS to set useROS attr of the locations,
        ## then use [Location].data for the stats #duh

        # basic attributes
        self.dumpFile = xlsDataDumpFile
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
            effl = self.effluent.full_data.copy()
        else:
            raise ValueError("effluent must have data")

        if self.influent.hasData:
            infl = self.influent.full_data.copy()
        else:
            infl = pandas.DataFrame(
                index=self.effluent.full_data.index,
                columns=self.effluent.full_data.columns
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
    def boxplot(self, ax=None, pos=1, yscale='log', notch=True, showmean=True,
                width=0.8, bacteria=False, ylabel=None, xlims=None, bothTicks=True,
                offset=0.35, patch_artist=False):
        '''
        Adds a boxplot to a matplotlib figure

        Input:
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

            xlims : optional sequence (length=2) or None (default)
                Custom limits of the x-axis. If None, defaults to
                [pos-1, pos+1].

            patch_artist : optional bool (default = False)
                Toggles the use of patch artist instead of a line artists
                for the boxes


        Returns:
            fig : matplotlib Figure instance
        '''
        fig, ax = utils.figutils._check_ax(ax)

        for loc, offset in zip([self.influent, self.effluent], [-1*offset, offset]):
            if loc.include:
                loc.boxplot(ax=ax, pos=pos+offset, notch=notch, showmean=showmean,
                            width=width/2, bacteria=bacteria, ylabel=None,
                            patch_artist=patch_artist)

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
            ax.set_xlim(xlims)

        if bothTicks:
            ax.set_xticks([pos-offset, pos+offset])
            ax.set_xticklabels([self.influent.name, self.effluent.name])
            if self.name is not None:
                ax.set_xlabel(self.name)
        else:
            ax.set_xticks([pos])
            if self.name is not None:
                ax.set_xticklabels([self.name])

        ax.xaxis.grid(False, which='major')

        return fig

    def probplot(self, ax=None, yscale='log', axtype='qq', ylabel=None,
                 clearYLabels=False):
        '''
        Adds probability plots to a matplotlib figure

        Input:
            ax : optional matplotlib axes object or None (default)
                Axes on which the boxplot with be drawn. If None, one will
                be created.

            yscale : optional string ['linear' or 'log' (default)]
                Scale formatting of the y-axis

            probAxis : bool (default = True)
                Toggles the display of probabilities (True) or Z-scores (i.e.,
                theoretical quantiles) on the x-axis

            ylabel : string or None (default):
                Label for y-axis

            clearYLabels : bool (default = False)
                If True, removed y-*tick* label from `ax`

        Returns:
            fig : matplotlib Figure instance
        '''
        fig, ax = utils.figutils._check_ax(ax)

        for loc in [self.influent, self.effluent]:
            if loc.include:
                loc.probplot(ax=ax, clearYLabels=False, axtype=axtype,
                             yscale=yscale)

        xlabels = {
            'pp': 'Theoretical percentiles',
            'qq': 'Theoretical quantiles',
            'prop': 'Non-exceedance probability (\%)'
        }

        ax.set_xlabel(xlabels[axtype])
        ax.legend(loc='upper left', frameon=True)

        return fig

    def statplot(self, pos=1, yscale='log', notch=True, showmean=True,
                 width=0.8, bacteria=False, ylabel=None, axtype='qq',
                 patch_artist=False):
        '''
        Creates a two-axis figure. Left axis has a bopxplot. Right axis
        contains a probability ot quantile plot.

        Input:
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

            patch_artist : optional bool (default = False)
                Toggles the use of patch artist instead of a line artists
                for the boxes

            probAxis : bool (default = True)
                Toggles the display of probabilities (True) or Z-scores (i.e.,
                theoretical quantiles) on the x-axis



        Returns:
            fig : matplotlib Figure instance
        '''
        # setup the figure and axes
        fig = plt.figure(figsize=(6.40, 3.00), facecolor='none',
                         edgecolor='none')
        ax1 = plt.subplot2grid((1, 4), (0, 0))
        ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

        self.boxplot(ax=ax1, pos=pos, yscale=yscale, notch=notch,
                     showmean=showmean, width=width, bacteria=bacteria,
                     ylabel=ylabel, patch_artist=patch_artist)

        self.probplot(ax=ax2, yscale=yscale, axtype=axtype,
                      ylabel=None, clearYLabels=True)

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        return fig

    def jointplot(self, hist=False, kde=True, rug=True, **scatter):
        showlegend = scatter.pop('showlegend', True)
        _ = scatter.pop('ax', None)

        if self.paired_data is not None:
            data = self.paired_data.xs('res', level='quantity', axis=1)
            jg = seaborn.JointGrid(x='inflow', y='outflow', data=data)
            self.scatterplot(ax=jg.ax_joint, showlegend=False, **scatter)
            jg.plot_marginals(seaborn.distplot, hist=hist, rug=rug, kde=kde)

            jg.ax_marg_x.set_xscale(scatter.pop('xscale', 'log'))
            jg.ax_marg_y.set_yscale(scatter.pop('yscale', 'log'))

            if showlegend:
                jg.ax_joint.legend(loc='upper left')

        return jg

    def scatterplot(self, ax=None, xscale='log', yscale='log', one2one=False,
                    useROS=False, xlabel=None, ylabel=None, showlegend=True):
        '''
        Adds an influent/effluent scatter plot to a matplotlibe figure

        Input:
            ax : optional matplotlib axes object or None (default)
                Axes on which the scatterplot with be drawn. If None, one will
                be created.

            [x|y]scale : optional string ['linear' or 'log' (default)]
                Scale formatting of the [x|y]-axis

            one2one : optional bool (default = False)
                Toggles the inclusion of the 1:1 line (aka line of equality)

            useROS : bool (default = True)
                Toggles the use of the ROS'd results. If False, raw results
                (i.e., detection limit for NDs) are used with varying
                symbology

            [x|y]label : string or None (default):
                Label for [x|y]-axis. If None, will be 'Influent'. If the dataset
                definition is available and incldues a Parameter, that will
                be included as will. For no label, use `[x|y]label = ""`

            showlegend : bool (default = True)
                Toggles the display of the legend

        Returns:
            fig : matplotlib Figure instance
        '''
        # set up the figure/axes
        fig, ax = utils.figutils._check_ax(ax)

        # set the scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # common symbology
        markerkwargs = dict(
            linestyle='none',
            markerfacecolor='black',
            markeredgecolor='white',
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
                           label='Detected data pairs', **markerkwargs)
            self._plot_nds(ax, which='influent', marker='v', alpha=0.6,
                           label='Influent not detected', **markerkwargs)
            self._plot_nds(ax, which='effluent', marker='<', alpha=0.6,
                           label='Effluent not detected', **markerkwargs)
            self._plot_nds(ax, which='both', marker='d', alpha=0.45,
                           label='Both not detected', **markerkwargs)

        label_format = mticker.FuncFormatter(utils.figutils.alt_logLabelFormatter)
        if xscale == 'log':
            ax.xaxis.set_major_formatter(label_format)
        else:
            ax.set_xlim(left=0)

        if yscale == 'log':
            ax.yaxis.set_major_formatter(label_format)
        else:
            ax.set_ylim(bottom=0)

        # unify the axes limits
        if xscale == yscale:
            ax.set_aspect('equal')
            axis_limits = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.set_ylim(axis_limits)
            ax.set_xlim(axis_limits)

        elif yscale == 'linear':
            axis_limits = [
                np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]

        elif xscale == 'linear':
            axis_limits = [
                np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]

        # generate plotting values for 1:1 line
        if xscale == yscale == 'linear':
            xx = np.linspace(*axis_limits)
        else:
            xx = np.logspace(*np.log10(axis_limits))

        # include the line of equality, if requested
        if one2one:
            ax.plot(xx, xx, linestyle='-', linewidth=1.25, alpha=0.50,
                    color='black', zorder=5, label='1:1 line')

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

        # set it to False to ensure that we're showing detection limits
        self.useROS = False
        if which == 'both':
            index = (self.paired_data[('inflow', 'qual')] == 'ND') & \
                    (self.paired_data[('outflow', 'qual')] == 'ND')

        elif which == 'influent':
            index = (self.paired_data[('inflow', 'qual')] == 'ND') & \
                    (self.paired_data[('outflow', 'qual')] != 'ND')

        elif which == 'effluent':
            index = (self.paired_data[('inflow', 'qual')] != 'ND') & \
                    (self.paired_data[('outflow', 'qual')] == 'ND')

        elif which == 'neither':
            index = (self.paired_data[('inflow', 'qual')] != 'ND') & \
                    (self.paired_data[('outflow', 'qual')] != 'ND')

        else:
            msg = '`which` must be "both", "influent", ' \
                  '"effluent", or "neighter"'
            raise ValueError(msg)

        x = self.paired_data.loc[index][('inflow', 'res')]
        y = self.paired_data.loc[index][('outflow', 'res')]
        ax.plot(x, y, label=label, **markerkwargs)

        # restore the original useROS values
        self.useROS = use_ros_cache


class DataCollection(object):
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
            self.groupby += list([othergroups])

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
        _tidy = (
            self.data
                .reset_index()[self.columns]
                .groupby(by=self.groupby)
                .filter(self.filterfxn)
                .groupby(by=self.groupby)
                .apply(lambda g: utils.ros.MR(g).data)
                .reset_index()
                .rename(columns={'final_data': self.roscol})
        )

        for c in _tidy.columns:
            if c not in self.columns + [self.roscol]:
                _tidy = (
                    _tidy.sort(columns=c) \
                         .reset_index() \
                         .drop([c, 'index'], axis=1)
                )

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
            loc_dict['location'] = Location(
                data, station_type=loc_dict[self.stationcol].lower(),
                rescol=self._raw_rescol, qualcol=self.qualcol,
                ndval=self.ndval, bsIter=self.bsIter,
                useROS=self.useROS
            )
            _locations.append(loc_dict)

        return _locations

    @cache_readonly
    def datasets(self):
        _datasets = []
        groupcols = [
            c for c in filter(lambda g: g != self.stationcol, self.groupby)
        ]
        for names, data in self.data.groupby(level=groupcols):
            ds_dict = dict(zip(groupcols, names))

            ds_dict[self.stationcol] = 'Inflow'
            infl = self.selectLocations(squeeze=True, **ds_dict)

            ds_dict[self.stationcol] = 'Outflow'
            effl = self.selectLocations(squeeze=True, **ds_dict)

            ds_dict.pop(self.stationcol)
            dsname = '_'.join(names).replace(', ','')
            ds_dict['dataset'] = Dataset(
                infl['location'], effl['location'], useROS=False, name=dsname
            )

            _datasets.append(ds_dict)

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
        return self._generic_stat(lambda x: np.mean(np.log(x)), statname='Log-mean')

    @cache_readonly
    def logmean(self):
        return self._generic_stat(lambda x: np.std(np.log(x)), statname='Log-std. dev.')

    @cache_readonly
    def geomean(self):
        geomean = np.exp(self.logmean)
        geomean.columns.names = ['Geo-mean']
        return geomean

    @cache_readonly
    def geostd(self):
        geostd = np.exp(self.logstd)
        geostd.columns.names = ['Geo-std. dev.']
        return geostd

    def _generic_stat(self, statfxn, bootstrap=True, statname=None):
        def CIs(x):
            bs = utils.bootstrap.Stat(x[self.rescol].values, statfxn=statfxn)
            stat, (lci, uci) = bs.BCA()
            statnames = ['lower', 'stat', 'upper']
            return pandas.Series([lci, stat, uci], index=statnames)

        stat = (
            self.tidy
                .groupby(by=self.groupby)
                .apply(CIs if bootstrap else {self.rescol: statfxn})
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
            items = [r for r in filter(lambda x: x[key] == value, items)]

        if squeeze and len(items) == 1:
            items = items[0]

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

    def facet_kde(self, row='category', col='parameter', hue='station',
                  log=True):
        df = self.tidy.copy()
        if log:
            plotcol = 'Log of {}'.format(self.rescol)
            df[plotcol] = np.log(df[self.rescol])
        else:
            pltcol = self.rescol

        fgrid = seaborn.FacetGrid(df, row=row, col=col, hue=hue,
                                  sharex=True, sharey=True,
                                  margin_titles=True,
                                  legend_out=False)
        fgrid.map(seaborn.kdeplot, plotcol, shade=True)
        return fgrid
