import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas

from ..utils import figutils

class _basic_wq_sample(object):
    def __init__(self, dataframe, starttime, samplefreq=None,
                 endtime=None, storm=None, rescol='res',
                 qualcol='qual', dlcol='DL', unitscol='units'):

        self._wqdata = dataframe

        self._startime = pandas.Timestamp(starttime)
        self._endtime = pandas.Timestamp(endtime)
        self._samplefreq = samplefreq
        self._sample_ts = None

        self.storm = storm
        self._label = None
        self._marker = None
        self._linestyle = None
        self._yfactor = None


    @property
    def wqdata(self):
        return self._wqdata
    @wqdata.setter
    def wqdata(self, value):
        self._wqdata = value

    @property
    def starttime(self):
        return self._startime
    @starttime.setter
    def starttime(self, value):
        self._startime = value

    @property
    def endtime(self):
        if self._endtime is None:
            self._endtime = self._startime
        return self._endtime
    @endtime.setter
    def endtime(self, value):
        self._endtime = value

    @property
    def samplefreq(self):
        return self._samplefreq
    @samplefreq.setter
    def samplefreq(self, value):
        self._samplefreq = value

    def plot_ts(self, ax, isFocus=True):
        if self.sample_ts is not None:
            if isFocus:
                alpha = 1.00
            else:
                alpha = 0.35

        ymax = ax.get_ylim()[-1]
        yposition = [self.yfactor * ymax] * len(self.sample_ts)

        ax.plot(self.sample_ts, yposition, marker=self.marker,
                markersize=4, linestyle=self.linestyle, color='Black',
                zorder=10, label='_nolegend', alpha=alpha)


class CompositeSample(_basic_wq_sample):
    @property
    def label(self):
        if self._label is None:
            self._label = 'Composite Sample'
        return self._label
    @label.setter
    def label(self, value):
        self._label = value

    @property
    def marker(self):
        if self._marker is None:
            self._marker = 'x'
        return self._marker
    @marker.setter
    def marker(self, value):
        self._marker = value

    @property
    def linestyle(self):
        if self._linestyle is None:
            self._linestyle = 'none'
        return self._linestyle
    @linestyle.setter
    def linestyle(self, value):
        self._linestyle = value

    @property
    def yfactor(self):
        if self._yfactor is None:
            self._yfactor = 0.35
        return self._yfactor
    @yfactor.setter
    def yfactor(self, value):
        self._yfactor = value

    @property
    def sample_ts(self):
        if self.starttime is not None and self.endtime is not None:
            if self.samplefreq is not None:
                self._sample_ts = pandas.DatetimeIndex(
                    start=self.starttime,
                    end=self.endtime,
                    freq=self.samplefreq
                )
            else:
                self._sample_ts = pandas.DatetimeIndex(data=[self.starttime, self.endtime])
        return self._sample_ts


class GrabSample(_basic_wq_sample):
    @property
    def label(self):
        if self._label is None:
            self._label = 'Grab Sample'
        return self._label
    @label.setter
    def label(self, value):
        self._label = value

    @property
    def marker(self):
        if self._marker is None:
            self._marker = '+'
        return self._marker
    @marker.setter
    def marker(self, value):
        self._marker = value

    @property
    def linestyle(self):
        if self._linestyle is None:
            self._linestyle = 'none'
        return self._linestyle
    @linestyle.setter
    def linestyle(self, value):
        self._linestyle = value

    @property
    def yfactor(self):
        if self._yfactor is None:
            self._yfactor = 0.25
        return self._yfactor
    @yfactor.setter
    def yfactor(self, value):
        self._yfactor = value

    @property
    def sample_ts(self):
        if self._sample_ts is None and self.starttime is not None:
            if self.endtime is None:
                self._sample_ts = pandas.DatetimeIndex(data=[self.starttime])
            else:
                self._sample_ts = pandas.DatetimeIndex(data=[self.starttime, self.endtime])
        return self._sample_ts


def defineStorms(hydrodata, precipcol=None, inflowcol=None, outflowcol=None,
                 minprecip=0.01, minflow=0.01, intereventperiods=36,
                 standardizeColNames=True, outputfreqMinutes=10,
                 debug=False):
    '''
    Loops through the hydrologic records and parses the data into storms.
        In this context, a storm is defined as starting whenever the
        hydrologic records shows non-zero precipitation or [in|out]flow
        from the BMP after a minimum inter-event dry period duration
        specified in the the function call. A new column (`storm`) is
        added to the DataFrame, of copy of which is returned.

    Input:
        hydrodata : pandas.DataFrame
            DataFrame of hydrologic data of the storm. Should contain
            a unique index of type pandas.DatetimeIndex.

        precipcol : optional string (default = None)
            Name of column in `hydrodata` containing precipiation data.

        inflowcol : optional string (default = None)
            Name of column in `hydrodata` containing influent flow data.

        outflowcol : optional string (default = None)
            Name of column in `hydrodata` containing effluent flow data.

        minprecip : optional float (default = 0.01)
            The minimum incremental precipiation depth required to be
            considered part of a storm.

        minflow : optional float (default = 0.01)
            The minimum incremental volumetric flowrate required to be
            considered part of a storm.

        intereventperiods : optional int (default = 36)
            The number of dry records (no flow or rain) required to end
            a storm.

        standardizeColNames : optional bool (default = True)
            Toggles renaming columns to standard names in the returned
            DataFrame.

        outputfreqMinutes : optional int (default = 10)
            The default frequency (minutes) to which all data will be
            resampled. Precipitation data will be summed up across '
            multiple timesteps during resampling, while flow will be
            averaged.

        debug : bool (default = False)
            If True, diagnostic columns will not be dropped prior to
            returning the dataframe of parsed_storms.

    Writes:
        None

    Returns:
        parsed_storms : pandas.DataFrame
            Copy of the origin `hydrodata` DataFrame, but resmapled to
            a fixed frequency, columns possibly renamed, and a `storm`
            column added to denote the storm to which each record
            belongs. Records where `storm` == 0 are not a part of any
            storm.

    '''

    # validate input
    if precipcol is None and inflowcol is None and outflowcol is None:
        msg = '`hydrodata` must have at least a precip or in/outflow column'
        raise ValueError(msg)

    # pull out the rain and flow data
    if precipcol is None:
        precipcol = 'precip'
        hydrodata.loc[:, precipcol] = np.nan

    if inflowcol is None:
        inflowcol = 'inflow'
        hydrodata.loc[:, inflowcol] = np.nan

    if outflowcol is None:
        outflowcol = 'outflow'
        hydrodata.loc[:, outflowcol] = np.nan

    # bool column where True means there's rain or flow of some kind
    hydrodata.loc[:, 'wet'] = hydrodata.apply(
        lambda r: (r[precipcol] >= minprecip or
                   r[inflowcol] >= minflow or
                   r[outflowcol] >= minflow),
        axis=1
    )

    # copy the bool column into its own df and add a bunch
    # shifted columns so each row looks backwards and forwards
    hydrodata.loc[:, 'windiff'] = pandas.rolling_apply(
        hydrodata['wet'],
        intereventperiods,
        lambda x: x.any(),
        min_periods=1
    ).diff()

    firstrow = hydrodata.iloc[0]
    if firstrow['wet']:
        hydrodata.loc[firstrow.name, 'windiff'] = 1

    hydrodata.loc[:, 'event_start'] = False
    hydrodata.loc[:, 'event_end'] = False

    starts = hydrodata['windiff'] == 1
    hydrodata.loc[starts, 'event_start'] = True

    stops = hydrodata['windiff'].shift(-1 * intereventperiods) == -1
    hydrodata.loc[stops, 'event_end'] = True

    # initialize the new column as zeros
    hydrodata.loc[:, 'storm'] = 0

    # each time a storm starts, incriment the storm number + 1
    hydrodata.loc[:, 'storm'] = hydrodata['event_start'].cumsum()

    # periods between storms are where the cumulative number
    # of storms that have ended are equal to the cumulative
    # number of storms that have started.
    # Stack Overflow: http://tinyurl.com/lsjkr9x
    nostorm = hydrodata['storm'] == hydrodata['event_end'].shift(2).cumsum()
    hydrodata.loc[nostorm, 'storm'] = 0

    if standardizeColNames:
        coldict = {
            precipcol: 'precip',
            inflowcol: 'inflow',
            outflowcol: 'outflow'
        }
        hydrodata.rename(columns=coldict, inplace=True)

    #hydrodata['storm'] = iswet['storm']
    if not debug:
        cols_to_drop = ['wet', 'windiff', 'event_end', 'event_start']
        hydrodata = hydrodata.drop(cols_to_drop, axis=1)

    return hydrodata


class Storm(object):
    def __init__(self, dataframe, stormnumber, precipcol='precip',
                 inflowcol='inflow', outflowcol='outflow',
                 tempcol='temp', stormcol='storm', freqMinutes=5):

        self.inflowcol = inflowcol
        self.outflowcol = outflowcol
        self.precipcol = precipcol

        # basic data
        self.full_record = dataframe.copy()
        self.data = dataframe[dataframe[stormcol] == stormnumber].copy()

        self.freqMinutes = freqMinutes
        self.hydrofreq_label = '{0} min'.format(self.freqMinutes)

        # tease out start/stop info
        self.storm_start = self.data.index[0]
        self.storm_end = self.data.index[-1]

        # storm duration (hours)
        duration = self.storm_end - self.storm_start
        sec_per_hr = 60.0 * 60.0
        self.duration_hours = duration.total_seconds() / sec_per_hr

        # antecedent dry period (hours)
        prev_storm_selector = self.full_record[stormcol] == stormnumber - 1
        previous_storm_end = self.full_record[prev_storm_selector].index[-1]
        antecedent_timedelta = self.storm_start - previous_storm_end
        sec_per_day = 60.0 * 60.0 * 24.0
        self.antecedent_period_days = \
            antecedent_timedelta.total_seconds() / sec_per_day

        self._precip_start = None
        self._precip_end = None
        self._inflow_start = None
        self._inflow_end = None
        self._outflow_start = None
        self._outflow_end = None

        # peaks
        self._peak_precip_intensity = None
        self._peak_inflow = None
        self._peak_outflow = None

        # times of peaks
        self._peak_precip_intensity_time = None
        self._peak_inflow_time = None
        self._peak_outflow_time = None
        self._peak_lag_hours = None

        # centroids
        self._centroid_precip_time = None
        self._centroid_inflow_time = None
        self._centroid_outflow_time = None
        self._centroid_lag_hours = None

        # totals
        self._total_precip_depth = None
        self._total_inflow_volume = None
        self._total_outflow_volume = None

        self.precip =  self.data[self.data[precipcol] > 0]
        self.inflow =  self.data[self.data[inflowcol] > 0]
        self.outflow =  self.data[self.data[outflowcol] > 0]

        self.has_precip = self.precip.shape[0] > 0
        self.has_inflow = self.inflow.shape[0] > 0
        self.has_outflow = self.outflow.shape[0] > 0

        self.meta = {
            'outflow': {
                'name': 'Flow (calculated, L/s)',
                'ylabel': 'Effluent flow (L/s)',
                'color': 'CornFlowerBlue',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': 0
            },
            'inflow': {
                'name': 'Inflow (estimated, L/s)',
                'ylabel': 'Estimated influent flow (L/s)',
                'color': 'Maroon',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': 0
            },
            'precip': {
                'name': 'Precip (mm)',
                'ylabel': '%s Precip.\nDepth (mm)' % self.hydrofreq_label,
                'color': 'DarkGreen',
                'linewidth': 1.5,
                'alpha': 0.4,
                'ymin': 0
            },
            'water level': {
                'name': 'Level (m)',
                'ylabel': 'Water level in BMP (m)',
                'color': 'Black',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': 0
            },
            'temp': {
                'name': 'Air Temp (deg C)',
                'ylabel': 'Air Temperature (deg. C)',
                'color': 'DarkGoldenRod',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': None
            }
        }

        # peaks
        self.peak_precip_intensity = self.data[precipcol].max()
        self.peak_inflow = self.data[inflowcol].max()
        self.peak_outflow = self.data[outflowcol].max()

    # starts adn stops
    @property
    def precip_start(self):
        if self._precip_start is None:
            self._precip_start = self._get_event_time(self.precipcol, 'start')
        return self._precip_start

    @property
    def precip_end(self):
        if self._precip_end is None:
            self._precip_end = self._get_event_time(self.precipcol, 'end')
        return self._precip_end

    @property
    def inflow_start(self):
        if self._inflow_start is None:
            self._inflow_start = self._get_event_time(self.inflowcol, 'start')
        return self._inflow_start

    @property
    def inflow_end(self):
        if self._inflow_end is None:
            self._inflow_end = self._get_event_time(self.inflowcol, 'end')
        return self._inflow_end

    @property
    def outflow_start(self):
        if self._outflow_start is None:
            self._outflow_start = self._get_event_time(self.outflowcol, 'start')
        return self._outflow_start

    @property
    def outflow_end(self):
        if self._outflow_end is None:
            self._outflow_end = self._get_event_time(self.outflowcol, 'end')
        return self._outflow_end

    # totals
    @property
    def total_precip_depth(self):
        if self._total_precip_depth is None:
            self._total_precip_depth = self.data[self.precipcol].sum()
        return self._total_precip_depth

    @property
    def total_inflow_volume(self):
        if self._total_inflow_volume is None:
            self._total_inflow_volume = self.data[self.inflowcol].sum() * 60 * self.freqMinutes
        return self._total_inflow_volume

    @property
    def total_outflow_volume(self):
        if self._total_outflow_volume is None:
            self._total_outflow_volume = self.data[self.outflowcol].sum() * 60 * self.freqMinutes
        return self._total_outflow_volume

    # centroids
    @property
    def centroid_precip_time(self):
        if self._centroid_precip_time is None:
            self._centroid_precip_time = self._compute_centroid(self.precipcol)
        return self._centroid_precip_time

    @property
    def centroid_inflow_time(self):
        if self._centroid_inflow_time is None:
            self._centroid_inflow_time = self._compute_centroid(self.inflowcol)
        return self._centroid_inflow_time

    @property
    def centroid_outflow_time(self):
        if self._centroid_outflow_time is None:
            self._centroid_outflow_time = self._compute_centroid(self.outflowcol)
        return self._centroid_outflow_time

    @property
    def centroid_lag_hours(self):
        if self._centroid_lag_hours is None:
            self._centroid_lag_hours = (
                self.centroid_outflow_time - self.centroid_inflow_time
            ).total_seconds() / 60. / 60.
        return self._centroid_lag_hours

    #times
    @property
    def peak_precip_intensity_time(self):
        if self._peak_precip_intensity_time is None:
            PI_selector = self.data[self.precipcol] == self.peak_precip_intensity
            self._peak_precip_intensity_time = self.data[PI_selector].index[0]
        return self._peak_precip_intensity_time

    @property
    def peak_inflow_time(self):
        if self._peak_inflow_time is None:
            PInf_selector = self.data[self.inflowcol] == self.peak_inflow
            self._peak_inflow_time = self.data[PInf_selector].index[0]
        return self._peak_inflow_time

    @property
    def peak_outflow_time(self):
        if self._peak_outflow_time is None:
            PEff_selector = self.data[self.outflowcol] == self.peak_outflow
            self._peak_outflow_time = self.data[PEff_selector].index[0]
        return self._peak_outflow_time

    @property
    def peak_lag_hours(self):
        if self._peak_lag_hours is None:
            self._peak_lag_hours = (self.peak_outflow_time - self.peak_inflow_time).total_seconds() / 60. / 60.
        return self._peak_lag_hours

    def _get_event_time(self, column, bound):
        index_map = {'start': 0, 'end': -1}
        quantity = self.data[self.data[column] > 0]
        if quantity.shape[0] == 0:
            warnings.warn("Storm has no {}".format(column), UserWarning)
        else:
            return quantity.index[index_map[bound]]

    def _get_max_quantity(self, column):
        return self.data[column].max()

    def _compute_centroid(self, column):
        # ordinal time index of storm
        time_idx = [
            mdates.date2num(idx.to_datetime()) for idx in self.data.index.tolist()
        ]

        centroid = np.sum(self.data[column] * time_idx) / np.sum(self.data[column])

        if np.isnan(centroid):
            return None
        else:
            return pandas.Timestamp(mdates.num2date(centroid)).tz_convert(None)

    def _plot_centroids(self, ax, yfactor=0.5):

        artists = []
        labels = []

        y_val = yfactor*ax.get_ylim()[1]
        if self.centroid_precip is not None:
            ax.plot([self.centroid_precip], [y_val], color='DarkGreen', marker='o',
                    linestyle='none', zorder=20, markersize=6)
            artists.append(mlines.Line2D([0], [0], marker='.', markersize=6,
                           linestyle='none', color='DarkGreen'))
            labels.append('Precip. centroid')

        if self.centroid_flow is not None:
            ax.plot([self.centroid_flow], [y_val], color='CornflowerBlue',
                    marker='s', linestyle='none', zorder=20, markersize=6)
            artists.append(mlines.Line2D([0], [0], marker='s',
                           markersize=6, linestyle='none', color='CornflowerBlue'))
            labels.append('Effluent centroid')

        if self.centroid_precip is not None and self.centroid_flow is not None:
            pass
            ax.annotate('', (self.centroid_flow, y_val),
                        arrowprops=dict(arrowstyle="-|>"),
                        xytext=(self.centroid_precip, y_val))
            #ax.plot([self.centroid_flow, self.centroid_precip], [y_val]*2,
            #            color='k', linestyle='-', zorder=20)

        return artists, labels

    def _plot_hydroquantity(self, quantity, ax=None, inverty=False,
                            rotation=90, cumulative=False):
        '''
        Adds an area of a quantity in the hydrologic data to an matplotlib
            axes.

        Input:
            quantity (string) : column name of the quantity you want to plot
            ax (matplotlib axes object, default None) : the axes on which the
                data will be plotted. If None, a new one will be created.
            interty (bool, default False) : whether or not to invert the y-axis

        Writes:
            None

        Returns:
            fig (matplotlib figure object) : the figure on which `ax` lives
        '''
        # setup the figure
        if ax is None:
            fig, ax = plt.subplots()
        #else:
        #    fig = ax.figure

        # plot properties based on the column
        col = quantity.lower()


        # select the plot props based on the column
        try:
            meta = self.meta[col]
        except KeyError:
            raise KeyError('%s not available. Try: %s' % (quantity, colmap.keys()))

        # selct data and fill in missing records with 0
        data = self.data[col]
        data.fillna(value=0, inplace=True)

        # plot the data
        ax.fill_between(data.index.to_pydatetime(), data, color=meta['color'],
                        linewidth=0, alpha=meta['alpha'], zorder=5)

        # y-label
        if quantity.lower() in ['outflow', 'inflow']:
            ax.set_ylabel('Flow (L/s)', color='Black', rotation=rotation,
                          verticalalignment='bottom')
        else:
            ax.set_ylabel(meta['ylabel'], color=meta['color'], rotation=rotation,
                          verticalalignment='bottom')

        # y-axis minuimum
        if meta['ymin'] is not None:
            ax.set_ylim(ymin=meta['ymin'])

        # y-axis maximum (the `if` handles no-data situations)
        ymax = ax.get_ylim()[1]
        if ymax < 0.25:
            ax.set_ylim(ymax=0.5)

        # y-axis inversion (precip only, typically)
        if inverty:
            ax.invert_yaxis()

        proxy = mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=meta['color'], linewidth=0, alpha=meta['alpha']
        )

        return proxy

    def summaryPlot(self, axratio=2, filename=None, **figkwargs):
        '''
        Creates a figure showing the hydrlogic record (flow and
            precipitation) of the storm

        Input:
            axratio : optional float or int (default = 2)
                Relative height of the flow axis compared to the
                precipiation axis.

            filename : optional string (default = None)
                Filename to which the figure will be saved.

            **figwargs will be passed on to `plt.Figure`

        Writes:
            Figure of flow and precipitation for a storm

        Returns:
            None
        '''
        fig = plt.figure(**figkwargs)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, axratio], hspace=0)
        rainax = fig.add_subplot(gs[0])
        flowax = fig.add_subplot(gs[1])

        # create the legend proxy artists
        artists = []
        labels = []

        legcols = 0
        if not np.all(np.isnan(self.data['inflow'])):
            inflow_proxy = self._plot_hydroquantity('inflow', ax=flowax, inverty=False)
            artists.append(inflow_proxy)
            labels.append('Influent flow')
            legcols += 1

        if not np.all(np.isnan(self.data['outflow'])):
            outflow_proxy = self._plot_hydroquantity('outflow', ax=flowax, inverty=False)
            artists.append(outflow_proxy)
            labels.append('Measured effluent')
            legcols += 1

        if not np.all(np.isnan(self.data['precip'])):
            precip_proxy = self._plot_hydroquantity('precip', ax=rainax, inverty=True, rotation=270)
            artists.append(precip_proxy)
            labels.append('Precipitation')
            legcols += 1

        leg = rainax.legend(artists, labels, loc='upper left', fontsize=7,
                            markerscale=0.75, ncol=legcols, frameon=False,
                            bbox_to_anchor=(0.00, 1.35))
        leg.get_frame().set_zorder(25)

        storm_length = 0.2 + flowax.get_xlim()[1] - flowax.get_xlim()[0]
        if storm_length <= 0.75:
            hourinterval = 1
            day_array = np.arange(0, 24, 4)
        elif 0.75 < storm_length <= 1.5:
            hourinterval = 2
            day_array = np.arange(0, 24, 6)
        elif 1.5 < storm_length <= 3:
            hourinterval = 4
            day_array = np.arange(0, 24, 12)
        elif 4 < storm_length <= 6:
            hourinterval = 8
            day_array = np.array([0])
        else:
            hourinterval = 12
            day_array = np.array([0])

        # hour tick marks and label format
        hour_array = np.arange(hourinterval, 24, hourinterval)

        for dhour in day_array[1:]:
            index = np.nonzero(hour_array == dhour)[0][0]
            hour_array = np.delete(hour_array, index)

        hours = mdates.HourLocator(byhour=hour_array)
        hourfmt = mdates.DateFormatter('%H:%M')

        # day tick marks and label formats
        days = mdates.HourLocator(byhour=day_array)
        dayfmt = mdates.DateFormatter('%H:%M\n%d-%b\n%Y')

        # x major ticks
        flowax.xaxis.set_major_locator(days)
        flowax.xaxis.set_major_formatter(dayfmt)

        # x minor ticks
        flowax.xaxis.set_minor_locator(hours)
        flowax.xaxis.set_minor_formatter(hourfmt)

        # tick tick formats
        flowax.tick_params(axis='both', which='which',
                                labelsize=7, length=4, pad=4)
        rainax.tick_params(axis='both', which='which',
                                labelsize=7, length=4, pad=4)

        rainax.xaxis.tick_top()
        rainax.yaxis.tick_right()
        rainax.yaxis.set_label_position('right')
        rainax.set_ylabel('Precipitation\n(mm)', rotation=270, verticalalignment='bottom')
        rainax.set_xticklabels([])

        flowax.yaxis.tick_left()
        flowax.xaxis.tick_bottom()

        rainax.spines['bottom'].set_visible(False)
        flowax.spines['top'].set_visible(False)

        # grid lines and axis background color and layout
        #figutils.grayAxes(flowax)
        figutils.gridlines(flowax)
        figutils.gridlines(rainax)
        #rainax.yaxis.grid(False, which='both')
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')

        return fig
