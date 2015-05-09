import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn.apionly as seaborn
import pandas

from ..utils import figutils

SEC_PER_MINUTE = 60.
MIN_PER_HOUR = 60.
HOUR_PER_DAY = 24.
SEC_PER_HOUR = SEC_PER_MINUTE * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOUR_PER_DAY


class _basic_wq_sample(object):
    def __init__(self, dataframe, starttime, samplefreq=None,
                 endtime=None, storm=None, rescol='res',
                 qualcol='qual', dlcol='DL', unitscol='units'):

        self._wqdata = dataframe
        self._startime = pandas.Timestamp(starttime)
        self._endtime = pandas.Timestamp(endtime)
        self._samplefreq = samplefreq
        self._sample_ts = None
        self._label = None
        self._marker = None
        self._linestyle = None
        self._yfactor = None
        self._season = getSeason(self.starttime)
        self.storm = storm

    @property
    def season(self):
        return self._season
    @season.setter
    def season(self, value):
        self._season = value

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

    def plot_ts(self, ax, isFocus=True, asrug=False):
        if self.sample_ts is not None:
            if isFocus:
                alpha = 0.75
            else:
                alpha = 0.35

        ymax = ax.get_ylim()[-1]
        yposition = [self.yfactor * ymax] * len(self.sample_ts)

        if asrug:
            seaborn.rugplot(self.sample_ts, ax=ax, color='black', alpha=alpha)

        else:
            ax.plot(self.sample_ts, yposition, marker=self.marker,
                    markersize=4, linestyle=self.linestyle, color='Black',
                    zorder=10, label='_nolegend', alpha=alpha)

        return plt.Line2D([0, 0], [0, 0], marker='|', mew=0.75,
                          color='black', alpha=alpha, linestyle='none')


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
    def sample_ts(self):
        if self._sample_ts is None and self.starttime is not None:
            if self.endtime is None:
                self._sample_ts = pandas.DatetimeIndex(data=[self.starttime])
            else:
                self._sample_ts = pandas.DatetimeIndex(data=[self.starttime, self.endtime])
        return self._sample_ts


class Storm(object):
    def __init__(self, dataframe, stormnumber, precipcol='precip',
                 inflowcol='inflow', outflowcol='outflow',
                 tempcol='temp', stormcol='storm', freqMinutes=5,
                 volume_conversion=1):

        self.inflowcol = inflowcol
        self.outflowcol = outflowcol
        self.precipcol = precipcol
        self.tempcol = tempcol
        self.stormnumber = stormnumber
        self.freqMinutes = freqMinutes
        self.volume_conversion = volume_conversion * SEC_PER_MINUTE * self.freqMinutes

        # basic data
        self.full_record = dataframe.copy()
        self.data = dataframe[dataframe[stormcol] == self.stormnumber].copy()
        self.hydrofreq_label = '{0} min'.format(self.freqMinutes)

        # tease out start/stop info
        self.storm_start = self.data.index[0]
        self.storm_end = self.data.index[-1]
        self._season = getSeason(self.storm_start)

        # storm duration (hours)
        duration = self.storm_end - self.storm_start
        self.duration_hours = duration.total_seconds() / SEC_PER_HOUR

        # antecedent dry period (hours)
        prev_storm_mask = self.full_record[stormcol] == self.stormnumber - 1
        previous_storm_end = self.full_record[prev_storm_mask].index[-1]
        antecedent_timedelta = self.storm_start - previous_storm_end
        self.antecedent_period_days = antecedent_timedelta.total_seconds() / SEC_PER_DAY

        # quantities
        self._precip = None
        self._inflow = None
        self._outflow = None

        # starts and stop
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

        self.meta = {
            self.outflowcol: {
                'name': 'Flow (calculated, L/s)',
                'ylabel': 'Effluent flow (L/s)',
                'color': 'CornFlowerBlue',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': 0
            },
            self.inflowcol: {
                'name': 'Inflow (estimated, L/s)',
                'ylabel': 'Estimated influent flow (L/s)',
                'color': 'Maroon',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': 0
            },
            self.precipcol: {
                'name': 'Precip (mm)',
                'ylabel': '%s Precip.\nDepth (mm)' % self.hydrofreq_label,
                'color': 'DarkGreen',
                'linewidth': 1.5,
                'alpha': 0.4,
                'ymin': 0
            },
            # self.waterlevelcol: {
            #     'name': 'Level (m)',
            #     'ylabel': 'Water level in BMP (m)',
            #     'color': 'Black',
            #     'linewidth': 1.5,
            #     'alpha': 0.5,
            #     'ymin': 0
            # },
            self.tempcol: {
                'name': 'Air Temp (deg C)',
                'ylabel': 'Air Temperature (deg. C)',
                'color': 'DarkGoldenRod',
                'linewidth': 1.5,
                'alpha': 0.5,
                'ymin': None
            }
        }

        # summaries
        self._summary_dict = None

    # quantities
    @property
    def precip(self):
        if self._precip is None:
            if self.precipcol is not None:
                self._precip = self.data[self.data[self.precipcol] > 0][self.precipcol]
            else:
                self._precip = np.array([])
        return self._precip

    @property
    def inflow(self):
        if self._inflow is None:
            if self.inflowcol is not None:
                self._inflow = self.data[self.data[self.inflowcol] > 0][self.inflowcol]
            else:
                self._inflow = np.array([])
        return self._inflow

    @property
    def outflow(self):
        if self._outflow is None:
            if self.outflowcol is not None:
                self._outflow = self.data[self.data[self.outflowcol] > 0][self.outflowcol]
            else:
                self._outflow = np.array([])
        return self._outflow

    @property
    def has_precip(self):
        return self.precip.shape[0] > 0

    @property
    def has_inflow(self):
        return self.inflow.shape[0] > 0

    @property
    def has_outflow(self):
        return self.outflow.shape[0] > 0

    @property
    def season(self):
        return self._season
    @season.setter
    def season(self, value):
        self._season = value

    # starts and stops
    @property
    def precip_start(self):
        if self._precip_start is None and self.has_precip:
            self._precip_start = self._get_event_time(self.precipcol, 'start')
        return self._precip_start

    @property
    def precip_end(self):
        if self._precip_end is None and self.has_precip:
            self._precip_end = self._get_event_time(self.precipcol, 'end')
        return self._precip_end

    @property
    def inflow_start(self):
        if self._inflow_start is None and self.has_inflow:
            self._inflow_start = self._get_event_time(self.inflowcol, 'start')
        return self._inflow_start

    @property
    def inflow_end(self):
        if self._inflow_end is None and self.has_inflow:
            self._inflow_end = self._get_event_time(self.inflowcol, 'end')
        return self._inflow_end

    @property
    def outflow_start(self):
        if self._outflow_start is None and self.has_outflow:
            self._outflow_start = self._get_event_time(self.outflowcol, 'start')
        return self._outflow_start

    @property
    def outflow_end(self):
        if self._outflow_end is None and self.has_outflow:
            self._outflow_end = self._get_event_time(self.outflowcol, 'end')
        return self._outflow_end

    # peaks
    @property
    def peak_precip_intensity(self):
        if self._peak_precip_intensity is None and self.has_precip:
            self._peak_precip_intensity = self.precip.max()
        return self._peak_precip_intensity

    @property
    def peak_inflow(self):
        if self._peak_inflow is None and self.has_inflow:
            self._peak_inflow = self.inflow.max()
        return self._peak_inflow

    @property
    def peak_outflow(self):
        if self._peak_outflow is None and self.has_outflow:
            self._peak_outflow = self.outflow.max()
        return self._peak_outflow

    # totals
    @property
    def total_precip_depth(self):
        if self._total_precip_depth is None and self.has_precip:
            self._total_precip_depth = self.data[self.precipcol].sum()
        return self._total_precip_depth

    @property
    def total_inflow_volume(self):
        if self._total_inflow_volume is None and self.has_inflow:
            self._total_inflow_volume = self.data[self.inflowcol].sum() * self.volume_conversion
        return self._total_inflow_volume

    @property
    def total_outflow_volume(self):
        if self._total_outflow_volume is None and self.has_outflow:
            self._total_outflow_volume = self.data[self.outflowcol].sum() * self.volume_conversion
        return self._total_outflow_volume

    # centroids
    @property
    def centroid_precip_time(self):
        if self._centroid_precip_time is None and self.has_precip:
            self._centroid_precip_time = self._compute_centroid(self.precipcol)
        return self._centroid_precip_time

    @property
    def centroid_inflow_time(self):
        if self._centroid_inflow_time is None and self.has_inflow:
            self._centroid_inflow_time = self._compute_centroid(self.inflowcol)
        return self._centroid_inflow_time

    @property
    def centroid_outflow_time(self):
        if self._centroid_outflow_time is None and self.has_outflow:
            self._centroid_outflow_time = self._compute_centroid(self.outflowcol)
        return self._centroid_outflow_time

    @property
    def centroid_lag_hours(self):
        if (self._centroid_lag_hours is None and
                self.centroid_outflow_time is not None and
                self.centroid_inflow_time is not None):
            self._centroid_lag_hours = (
                self.centroid_outflow_time - self.centroid_inflow_time
            ).total_seconds() / SEC_PER_HOUR
        return self._centroid_lag_hours

    #times
    @property
    def peak_precip_intensity_time(self):
        if self._peak_precip_intensity_time is None and self.has_precip:
            PI_selector = self.data[self.precipcol] == self.peak_precip_intensity
            self._peak_precip_intensity_time = self.data[PI_selector].index[0]
        return self._peak_precip_intensity_time

    @property
    def peak_inflow_time(self):
        if self._peak_inflow_time is None and self.has_inflow:
            PInf_selector = self.data[self.inflowcol] == self.peak_inflow
            self._peak_inflow_time = self.data[PInf_selector].index[0]
        return self._peak_inflow_time

    @property
    def peak_outflow_time(self):
        if self._peak_outflow_time is None and self.has_outflow:
            PEff_selector = self.data[self.outflowcol] == self.peak_outflow
            if PEff_selector.sum() > 0:
                self._peak_outflow_time = self.data[PEff_selector].index[0]
        return self._peak_outflow_time

    @property
    def peak_lag_hours(self):
        if (self._peak_lag_hours is None and
                self.peak_outflow_time is not None and
                self.peak_inflow_time is not None):

            time_delta = self.peak_outflow_time - self.peak_inflow_time
            self._peak_lag_hours = time_delta.total_seconds() / SEC_PER_HOUR
        return self._peak_lag_hours

    @property
    def summary_dict(self):
        if self._summary_dict is None:
            self._summary_dict = {
                'Storm Number': self.stormnumber,
                'Antecedent Days': self.antecedent_period_days,
                'Start Date': self.storm_start,
                'End Date': self.storm_end,
                'Duration Hours': self.duration_hours,
                'Peak Precip Intensity': self.peak_precip_intensity,
                'Total Precip Depth': self.total_precip_depth,
                'Total Inflow Volume': self.total_inflow_volume,
                'Peak Inflow': self.peak_inflow,
                'Total Outflow Volume': self.total_outflow_volume,
                'Peak Outflow': self.peak_outflow,
                'Peak Lag Hours': self.peak_lag_hours,
                'Season': self.season
            }

        return self._summary_dict

    def is_small(self, minprecip=0.0, mininflow=0.0, minoutflow=0.0):
        storm_is_small = (
            (self.total_precip_depth is not None and self.total_precip_depth < minprecip) or
            (self.total_inflow_volume is not None and self.total_inflow_volume < mininflow) or
            (self.total_outflow_volume is not None and self.total_outflow_volume < minoutflow)
        )
        return storm_is_small

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
            ax.annotate('', (self.centroid_flow, y_val),
                        arrowprops=dict(arrowstyle="-|>"),
                        xytext=(self.centroid_precip, y_val))

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
        data = self.data[col].copy()
        data.fillna(value=0, inplace=True)

        # plot the data
        ax.fill_between(data.index, data, color=meta['color'],
                        alpha=meta['alpha'], zorder=5,
                        linewidth=0, )

        # y-label
        if quantity.lower() in ['outflow', 'inflow']:
            ax.set_ylabel('Flow (L/s)', color='Black')
        else:
            ax.set_ylabel(meta['ylabel'], color=meta['color'])

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

    def summaryPlot(self, axratio=2, filename=None, showLegend=True,
                    precip=True, inflow=True, outflow=True, **figkwargs):
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
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, axratio],
                               hspace=0.08)
        rainax = fig.add_subplot(gs[0])
        flowax = fig.add_subplot(gs[1], sharex=rainax)

        # create the legend proxy artists
        artists = []
        labels = []

        legcols = 0
        if inflow:
            inflow_proxy = self._plot_hydroquantity(
                self.inflowcol, ax=flowax, inverty=False
            )
            artists.append(inflow_proxy)
            labels.append('Influent flow')
            legcols += 1

        if outflow:
            outflow_proxy = self._plot_hydroquantity(
                self.outflowcol, ax=flowax, inverty=False
            )
            artists.append(outflow_proxy)
            labels.append('Measured effluent')
            legcols += 1

        if precip:
            precip_proxy = self._plot_hydroquantity(
                self.precipcol, ax=rainax, inverty=True
            )
            artists.append(precip_proxy)
            labels.append('Precipitation')
            legcols += 1

        if showLegend:
            leg = rainax.legend(artists, labels, loc='upper left', fontsize=7,
                                markerscale=0.75, ncol=legcols, frameon=False,
                                bbox_to_anchor=(0.00, 1.35))
            leg.get_frame().set_zorder(25)

        storm_length = 0.2 + flowax.get_xlim()[1] - flowax.get_xlim()[0]
        if storm_length <= 0.75:
            hourinterval = 1
            day_array = np.arange(0, HOUR_PER_DAY, 4)
        elif 0.75 < storm_length <= 1.5:
            hourinterval = 2
            day_array = np.arange(0, HOUR_PER_DAY, 6)
        elif 1.5 < storm_length <= 3:
            hourinterval = 4
            day_array = np.arange(0, HOUR_PER_DAY, 12)
        elif 4 < storm_length <= 6:
            hourinterval = 8
            day_array = np.array([0])
        else:
            hourinterval = 12
            day_array = np.array([0])

        # hour tick marks and label format
        hour_array = np.arange(hourinterval, HOUR_PER_DAY, hourinterval)

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

        rainax.set_ylabel('Precipitation\n(mm)')
        rainax.set_xticklabels([])

        # grid lines and axis background color and layout
        #fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')

        return fig, artists, labels


class HydroRecord(object):
    '''
    Parameters
    ----------
    hydrodata : pandas.DataFrame
        DataFrame of hydrologic data of the storm. Should contain
        a unique index of type pandas.DatetimeIndex.

    precipcol : optional string (default = None)
        Name of column in `hydrodata` containing precipiation data.

    inflowcol : optional string (default = None)
        Name of column in `hydrodata` containing influent flow data.

    outflowcol : optional string (default = None)
        Name of column in `hydrodata` containing effluent flow data.

    intereventPeriods : optional int (default = 36)
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
    '''

    def __init__(self, hydrodata, precipcol=None, inflowcol=None,
                 outflowcol=None, tempcol=None, stormcol='storm',
                 minprecip=0.0, mininflow=0.0, minoutflow=0.0,
                 outputfreqMinutes=10, intereventPeriods=36,
                 volume_conversion=1, stormclass=None):

        # validate input
        if precipcol is None and inflowcol is None and outflowcol is None:
            msg = '`hydrodata` must have at least a precip or in/outflow column'
            raise ValueError(msg)

        if stormclass is None:
            self.stormclass = Storm
        else:
            self.stormclass = stormclass

        # static input
        self._raw_data = hydrodata
        self.precipcol = precipcol
        self.inflowcol = inflowcol
        self.outflowcol = outflowcol
        self.stormcol = stormcol
        self.tempcol = tempcol
        self.outputfreq = pandas.offsets.Minute(outputfreqMinutes)
        self.intereventPeriods = intereventPeriods
        self.minprecip = minprecip
        self.mininflow = mininflow
        self.minoutflow = minoutflow
        self.volume_conversion = volume_conversion

        # properties
        self._data = None
        self._all_storms = None
        self._storms = None
        self._storm_stats = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._define_storms()
        return self._data

    @property
    def all_storms(self):
        if self._all_storms is None:
            self._all_storms = {}
            for storm_number in self.data[self.stormcol].unique():
                if storm_number > 0:
                    this_storm = self.stormclass(
                        self.data, storm_number, precipcol=self.precipcol,
                        inflowcol=self.inflowcol, outflowcol=self.outflowcol,
                        tempcol=self.tempcol, stormcol=self.stormcol,
                        volume_conversion=self.volume_conversion,
                        freqMinutes=self.outputfreq.n,
                    )
                    self._all_storms[storm_number] = this_storm

        return self._all_storms

    @property
    def storms(self):
        if self._storms is None:
            self._storms = {
                snum: storm for snum, storm in self.all_storms.items()
                    if not storm.is_small(
                        minprecip=self.minprecip,
                        mininflow=self.mininflow,
                        minoutflow=self.minoutflow
                    )
            }
        return self._storms

    @property
    def storm_stats(self):
        if self._storm_stats is None:
            storm_stats = pandas.DataFrame([
                self.storms[sn].summary_dict for sn in self.storms
            ])

            col_order = [
                'Storm Number',
                'Antecedent Days',
                'Start Date',
                'End Date',
                'Duration Hours',
                'Peak Precip Intensity',
                'Total Precip Depth',
                'Total Inflow Volume',
                'Peak Inflow',
                'Total Outflow Volume',
                'Peak Outflow',
                'Peak Lag Hours'
            ]
            self._storm_stats = storm_stats[col_order]

        return self._storm_stats

    def _define_storms(self, debug=False):
        '''
        Loops through the hydrologic records and parses the data into
        storms. In this context, a storm is defined as starting whenever
        the hydrologic records shows non-zero precipitation or
        [in|out]flow from the BMP after a minimum inter-event dry period
        duration specified in the the function call.

        Parameters
        ----------
        debug : bool (default = False)
            If True, diagnostic columns will not be dropped prior to
            returning the dataframe of parsed_storms.

        Writes
        ------
        None

        Returns
        -------
        parsed_storms : pandas.DataFrame
            Copy of the origin `hydrodata` DataFrame, but resmapled to
            a fixed frequency, columns possibly renamed, and a `storm`
            column added to denote the storm to which each record
            belongs. Records where `storm` == 0 are not a part of any
            storm.

        '''

        data = self._raw_data.copy()

        # pull out the rain and flow data
        if self.precipcol is None:
            precipcol = 'precip'
            data.loc[:, precipcol] = np.nan
        else:
            precipcol = self.precipcol

        if self.inflowcol is None:
            inflowcol = 'inflow'
            data.loc[:, inflowcol] = np.nan
        else:
            inflowcol = self.inflowcol

        if self.outflowcol is None:
            outflowcol = 'outflow'
            data.loc[:, outflowcol] = np.nan
        else:
            outflowcol = self.outflowcol

        # bool column where True means there's rain or flow of some kind
        water_columns = [precipcol, inflowcol, outflowcol]
        data.loc[:, 'wet'] = np.any(data[water_columns] > 0, axis=1)

        # copy the bool column into its own df and add a bunch
        # shifted columns so each row looks backwards and forwards
        data.loc[:, 'windiff'] = pandas.rolling_apply(
            data['wet'],
            self.intereventPeriods,
            lambda x: x.any(),
            min_periods=1
        ).diff()

        firstrow = data.iloc[0]
        if firstrow['wet']:
            data.loc[firstrow.name, 'windiff'] = 1

        data.loc[:, 'event_start'] = False
        data.loc[:, 'event_end'] = False

        starts = data['windiff'] == 1
        data.loc[starts, 'event_start'] = True

        stops = data['windiff'].shift(-1 * self.intereventPeriods) == -1
        data.loc[stops, 'event_end'] = True

        # initialize the new column as zeros
        data.loc[:, self.stormcol] = 0

        # each time a storm starts, incriment the storm number + 1
        data.loc[:, self.stormcol] = data['event_start'].cumsum()

        # periods between storms are where the cumulative number
        # of storms that have ended are equal to the cumulative
        # number of storms that have started.
        # Stack Overflow: http://tinyurl.com/lsjkr9x
        nostorm = data[self.stormcol] == data['event_end'].shift(2).cumsum()
        data.loc[nostorm, self.stormcol] = 0

        if not debug:
            cols_to_drop = ['wet', 'windiff', 'event_end', 'event_start']
            data = data.drop(cols_to_drop, axis=1)

        return data

    def getStormFromTimestamp(self, timestamp, lookback_hours=0, smallstorms=False):
        '''Get the storm associdated with a give (sample) date

        Parameters
        ----------
        timestamp : pandas.Timestamp
            The date/time for which to search within the hydrologic
            record.
        lookback_hours : positive int or float, optional (default = 0)
            If no storm is actively occuring at the provided timestamp,
            we can optionally look backwards in the hydrologic record a
            fixed amount of time (specified in hours). Negative values
            are ignored.
        smallstorms : bool, optional (default = False)
            If True, small storms will be included in the search.

        Returns
        -------
        storm_number : int
        storm : wqio.Storm

        '''

        # santize date input
        if not isinstance(timestamp, pandas.Timestamp):
            try:
                timestamp = pandas.Timestamp(timestamp)
            except:
                raise ValueError('{} could not be coerced into a pandas.Timestamp')

        # check lookback hours
        if lookback_hours < 0:
            raise ValueError('`lookback_hours` must be greater than 0')

        # initial search for the storm
        storm_number = int(self.data.loc[:timestamp, self.stormcol].iloc[-1])

        # look backwards if we have too
        if (storm_number == 0 or pandas.isnull(storm_number)) and lookback_hours != 0:
            lookback_time = timestamp - pandas.offsets.Hour(lookback_hours)
            storms = self.data.loc[lookback_time:timestamp, [self.stormcol]]
            storms = storms[storms > 0].dropna()

            if storms.shape[0] == 0:
                # no storm
                storm_number = None
            else:
                # storm w/i the lookback period
                storm_number = int(storms.iloc[-1])

        # return storm_number and storms
        if smallstorms:
            return storm_number, self.all_storms.get(storm_number, None)
        else:
            return storm_number, self.storms.get(storm_number, None)


def getSeason(date):
    '''Defines the season from a given date.

    Parameters
    ----------
    date : datetime.datetime object or similar
        Any object that represents a date and has `.month` and `.day`
        attributes

    Returns
    -------
    season : str

    Notes
    -----
    Assumes that all seasons changed on the 22nd (e.g., all winters
    start on Decemeber 22). This isn't strictly true, but it's good
    enough for now.

    '''

    if (date.month == 12 and date.day >= 22) or \
            (date.month in [1, 2]) or \
            (date.month == 3 and date.day < 22):
        return 'winter'
    elif (date.month == 3 and date.day >= 22) or \
            (date.month in [4, 5]) or \
            (date.month == 6 and date.day < 22):
        return 'spring'
    elif (date.month == 6 and date.day >= 22) or \
            (date.month in [7, 8]) or \
            (date.month == 9 and date.day < 22):
        return 'summer'
    elif (date.month == 9 and date.day >= 22) or \
            (date.month in [10, 11]) or \
            (date.month == 12 and date.day < 22):
        return 'autumn'
    else: # pragma: no cover
        raise ValueError('could not assign season to  {}'.format(date))



