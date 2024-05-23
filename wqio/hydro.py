import warnings

import numpy
import pandas
import seaborn
from matplotlib import dates, gridspec, pyplot
from pandas.plotting import register_matplotlib_converters

from wqio import utils, validate, viz

register_matplotlib_converters()

SEC_PER_MINUTE = 60.0
MIN_PER_HOUR = 60.0
HOUR_PER_DAY = 24.0
SEC_PER_HOUR = SEC_PER_MINUTE * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOUR_PER_DAY


def _wet_first_row(df, wetcol, diffcol):
    # make sure that if the first record is associated with the first
    # storm if it's wet
    firstrow = df.iloc[0]
    if firstrow[wetcol]:
        df.loc[firstrow.name, diffcol] = 1

    return df


def _wet_window_diff(is_wet, ie_periods):
    return (
        is_wet.rolling(int(ie_periods), min_periods=1)
        .apply(lambda window: window.any(), raw=False)
        .diff()
    )


def parse_storm_events(
    data,
    intereventHours,
    outputfreqMinutes,
    precipcol=None,
    inflowcol=None,
    outflowcol=None,
    baseflowcol=None,
    stormcol="storm",
    debug=False,
):
    """Parses the hydrologic data into distinct storms.

    In this context, a storm is defined as starting whenever the
    hydrologic records shows non-zero precipitation or [in|out]flow
    from the BMP after a minimum inter-event dry period duration
    specified in the the function call. The storms ends the observation
    *after* the last non-zero precipitation or flow value.

    Parameters
    ----------
    data : pandas.DataFrame
    intereventHours : float
        The Inter-Event dry duration (in hours) that classifies the
        next hydrlogic activity as a new event.
    precipcol : string, optional (default = None)
        Name of column in `hydrodata` containing precipiation data.
    inflowcol : string, optional (default = None)
        Name of column in `hydrodata` containing influent flow data.
    outflowcol : string, optional (default = None)
        Name of column in `hydrodata` containing effluent flow data.
    baseflowcol : string, optional (default = None)
        Name of column in `hydrodata` containing boolean indicating
        which records are considered baseflow.
    stormcol : string (default = 'storm')
        Name of column in `hydrodata` indentifying distinct storms.
    debug : bool (default = False)
        If True, diagnostic columns will not be dropped prior to
        returning the dataframe of parsed_storms.

    Writes
    ------
    None

    Returns
    -------
    parsed_storms : pandas.DataFrame
        Copy of the origin `hydrodata` DataFrame, but resampled to a
        fixed frequency, columns possibly renamed, and a `storm` column
        added to denote the storm to which each record belongs. Records
        where `storm` == 0 are not a part of any storm.

    """

    # pull out the rain and flow data
    if precipcol is None:
        precipcol = "precip"
        data.loc[:, precipcol] = numpy.nan

    if inflowcol is None:
        inflowcol = "inflow"
        data.loc[:, inflowcol] = numpy.nan

    if outflowcol is None:
        outflowcol = "outflow"
        data.loc[:, outflowcol] = numpy.nan

    if baseflowcol is None:
        baseflowcol = "baseflow"
        data.loc[:, baseflowcol] = False

    # bool column where True means there's rain or flow of some kind
    water_columns = [inflowcol, outflowcol, precipcol]
    cols_to_use = water_columns + [baseflowcol]

    agg_dict = {
        precipcol: numpy.sum,
        inflowcol: numpy.mean,
        outflowcol: numpy.mean,
        baseflowcol: numpy.any,
    }

    freq = pandas.offsets.Minute(outputfreqMinutes)
    ie_periods = int(MIN_PER_HOUR / freq.n * intereventHours)

    # periods between storms are where the cumulative number
    # of storms that have ended are equal to the cumulative
    # number of storms that have started.
    # Stack Overflow: http://tinyurl.com/lsjkr9x
    res = (
        data.resample(freq)
        .agg(agg_dict)
        .loc[:, lambda df: df.columns.isin(cols_to_use)]
        .assign(__wet=lambda df: numpy.any(df[water_columns] > 0, axis=1) & ~df[baseflowcol])
        .assign(__windiff=lambda df: _wet_window_diff(df["__wet"], ie_periods))
        .pipe(_wet_first_row, "__wet", "__windiff")
        .assign(__event_start=lambda df: df["__windiff"] == 1)
        .assign(__event_end=lambda df: df["__windiff"].shift(-1 * ie_periods) == -1)
        .assign(__storm=lambda df: df["__event_start"].cumsum())
        .assign(
            storm=lambda df: numpy.where(
                df["__storm"] == df["__event_end"].shift(2).cumsum(),
                0,  # inter-event periods marked as zero
                df["__storm"],  # actual events keep their number
            )
        )
    )

    if not debug:
        res = res.loc[:, res.columns.map(lambda c: not c.startswith("__"))]

    return res


class Storm:
    """Object representing a storm event

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A datetime-indexed Dataframe containing all of the hydrologic
        data and am interger column indentifying distinct storms.
    stormnumber : int
        The storm we care about.
    precipcol, inflowcol, outflow, tempcol, stormcol : string, optional
        Names for columns representing each hydrologic quantity.
    freqMinutes : float (default = 5)
        The time period, in minutes, between observations.
    volume_conversion : float, optional (default = 1)
        Conversion factor to go from flow to volume for a single
        observation.

    """

    # TODO: rename freqMinutes to periodMinutes
    def __init__(
        self,
        dataframe,
        stormnumber,
        precipcol="precip",
        inflowcol="inflow",
        outflowcol="outflow",
        tempcol="temp",
        stormcol="storm",
        freqMinutes=5,
        volume_conversion=1,
    ):
        self.inflowcol = inflowcol
        self.outflowcol = outflowcol
        self.precipcol = precipcol
        self.tempcol = tempcol
        self.stormnumber = stormnumber
        self.freqMinutes = freqMinutes
        self.volume_conversion = volume_conversion * SEC_PER_MINUTE * self.freqMinutes

        # basic data
        self.data = dataframe[dataframe[stormcol] == self.stormnumber].copy()
        self.hydrofreq_label = f"{self.freqMinutes} min"

        # tease out start/stop info
        self.start = self.data.index[0]
        self.end = self.data.index[-1]
        self._season = utils.get_season(self.start)

        # storm duration (hours)
        duration = self.end - self.start
        self.duration_hours = duration.total_seconds() / SEC_PER_HOUR

        # antecedent dry period (hours)
        if self.stormnumber > 1:
            prev_storm_mask = dataframe[stormcol] == self.stormnumber - 1
            previous_end = dataframe[prev_storm_mask].index[-1]
            antecedent_timedelta = self.start - previous_end
            self.antecedent_period_days = antecedent_timedelta.total_seconds() / SEC_PER_DAY
        else:
            self.antecedent_period_days = numpy.nan

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
                "name": "Flow (calculated, L/s)",
                "ylabel": "Effluent flow (L/s)",
                "color": "CornFlowerBlue",
                "linewidth": 1.5,
                "alpha": 0.5,
                "ymin": 0,
            },
            self.inflowcol: {
                "name": "Inflow (estimated, L/s)",
                "ylabel": "Estimated influent flow (L/s)",
                "color": "Maroon",
                "linewidth": 1.5,
                "alpha": 0.5,
                "ymin": 0,
            },
            self.precipcol: {
                "name": "Precip (mm)",
                "ylabel": f"{self.hydrofreq_label} Precip.\nDepth (mm)",
                "color": "DarkGreen",
                "linewidth": 1.5,
                "alpha": 0.4,
                "ymin": 0,
            },
            self.tempcol: {
                "name": "Air Temp (deg C)",
                "ylabel": "Air Temperature (deg. C)",
                "color": "DarkGoldenRod",
                "linewidth": 1.5,
                "alpha": 0.5,
                "ymin": None,
            },
        }

        self._summary_dict = None

    @property
    def precip(self):
        if self._precip is None:
            if self.precipcol is not None:
                self._precip = self.data[self.data[self.precipcol] > 0][self.precipcol]
            else:
                self._precip = numpy.array([])
        return self._precip

    @property
    def inflow(self):
        if self._inflow is None:
            if self.inflowcol is not None:
                self._inflow = self.data[self.data[self.inflowcol] > 0][self.inflowcol]
            else:
                self._inflow = numpy.array([])
        return self._inflow

    @property
    def outflow(self):
        if self._outflow is None:
            if self.outflowcol is not None:
                self._outflow = self.data[self.data[self.outflowcol] > 0][self.outflowcol]
            else:
                self._outflow = numpy.array([])
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
            self._precip_start = self._get_event_time(self.precipcol, "start")
        return self._precip_start

    @property
    def precip_end(self):
        if self._precip_end is None and self.has_precip:
            self._precip_end = self._get_event_time(self.precipcol, "end")
        return self._precip_end

    @property
    def inflow_start(self):
        if self._inflow_start is None and self.has_inflow:
            self._inflow_start = self._get_event_time(self.inflowcol, "start")
        return self._inflow_start

    @property
    def inflow_end(self):
        if self._inflow_end is None and self.has_inflow:
            self._inflow_end = self._get_event_time(self.inflowcol, "end")
        return self._inflow_end

    @property
    def outflow_start(self):
        if self._outflow_start is None and self.has_outflow:
            self._outflow_start = self._get_event_time(self.outflowcol, "start")
        return self._outflow_start

    @property
    def outflow_end(self):
        if self._outflow_end is None and self.has_outflow:
            self._outflow_end = self._get_event_time(self.outflowcol, "end")
        return self._outflow_end

    @property
    def _peak_depth(self):
        if self.has_precip:
            return self.precip.max()

    @property
    def peak_precip_intensity(self):
        if self._peak_precip_intensity is None and self.has_precip:
            self._peak_precip_intensity = self._peak_depth * MIN_PER_HOUR / self.freqMinutes
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
        if (
            self._centroid_lag_hours is None
            and self.centroid_outflow_time is not None
            and self.centroid_inflow_time is not None
        ):
            self._centroid_lag_hours = (
                self.centroid_outflow_time - self.centroid_inflow_time
            ).total_seconds() / SEC_PER_HOUR
        return self._centroid_lag_hours

    @property
    def peak_precip_intensity_time(self):
        if self._peak_precip_intensity_time is None and self.has_precip:
            PI_selector = self.data[self.precipcol] == self._peak_depth
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
        if (
            self._peak_lag_hours is None
            and self.peak_outflow_time is not None
            and self.peak_inflow_time is not None
        ):
            time_delta = self.peak_outflow_time - self.peak_inflow_time
            self._peak_lag_hours = time_delta.total_seconds() / SEC_PER_HOUR
        return self._peak_lag_hours

    @property
    def summary_dict(self):
        if self._summary_dict is None:
            self._summary_dict = {
                "Storm Number": self.stormnumber,
                "Antecedent Days": self.antecedent_period_days,
                "Start Date": self.start,
                "End Date": self.end,
                "Duration Hours": self.duration_hours,
                "Peak Precip Intensity": self.peak_precip_intensity,
                "Total Precip Depth": self.total_precip_depth,
                "Total Inflow Volume": self.total_inflow_volume,
                "Peak Inflow": self.peak_inflow,
                "Total Outflow Volume": self.total_outflow_volume,
                "Peak Outflow": self.peak_outflow,
                "Peak Lag Hours": self.peak_lag_hours,
                "Centroid Lag Hours": self.centroid_lag_hours,
                "Season": self.season,
            }

        return self._summary_dict

    def is_small(self, minprecip=0.0, mininflow=0.0, minoutflow=0.0):
        """Determines whether a storm can be considered "small".

        Parameters
        ----------
        minprecip, mininflow, minoutflow : float, optional (default = 0)
            The minimum amount of each hydrologic quantity below which a
            storm can be considered "small".

        Returns
        -------
        storm_is_small : bool
            True if the storm is considered small.

        """

        storm_is_small = (
            (self.total_precip_depth is not None and self.total_precip_depth < minprecip)
            or (self.total_inflow_volume is not None and self.total_inflow_volume < mininflow)
            or (self.total_outflow_volume is not None and self.total_outflow_volume < minoutflow)
        )
        return storm_is_small

    def _get_event_time(self, column, bound):
        index_map = {"start": 0, "end": -1}
        quantity = self.data[self.data[column] > 0]
        if quantity.shape[0] == 0:
            warnings.warn(f"Storm has no {column}", UserWarning)
        else:
            return quantity.index[index_map[bound]]

    def _get_max_quantity(self, column):
        return self.data[column].max()

    def _compute_centroid(self, column):
        # ordinal time index of storm
        time_idx = [dates.date2num(idx.to_pydatetime()) for idx in self.data.index.tolist()]

        centroid = numpy.sum(self.data[column] * time_idx) / numpy.sum(self.data[column])

        if numpy.isnan(centroid):
            return None
        else:
            return pandas.Timestamp(dates.num2date(centroid)).tz_convert(None)

    def _plot_centroids(self, ax, yfactor=0.5):
        artists = []
        labels = []
        y_val = yfactor * ax.get_ylim()[1]

        if self.centroid_precip is not None:
            ax.plot(
                [self.centroid_precip],
                [y_val],
                color="DarkGreen",
                marker="o",
                linestyle="none",
                zorder=20,
                markersize=6,
            )
            artists.append(
                pyplot.Line2D(
                    [0],
                    [0],
                    marker=".",
                    markersize=6,
                    linestyle="none",
                    color="DarkGreen",
                )
            )
            labels.append("Precip. centroid")

        if self.centroid_flow is not None:
            ax.plot(
                [self.centroid_flow],
                [y_val],
                color="CornflowerBlue",
                marker="s",
                linestyle="none",
                zorder=20,
                markersize=6,
            )
            artists.append(
                pyplot.Line2D(
                    [0],
                    [0],
                    marker="s",
                    markersize=6,
                    linestyle="none",
                    color="CornflowerBlue",
                )
            )
            labels.append("Effluent centroid")

        if self.centroid_precip is not None and self.centroid_flow is not None:
            ax.annotate(
                "",
                (self.centroid_flow, y_val),
                arrowprops=dict(arrowstyle="-|>"),
                xytext=(self.centroid_precip, y_val),
            )

        return artists, labels

    def plot_hydroquantity(self, quantity, ax=None, label=None, otherlabels=None, artists=None):
        """Draws a hydrologic quantity to a matplotlib axes.

        Parameters
        ----------
        quantity : string
            Column name of the quantity you want to plot.
        ax : matplotlib axes object, optional
            The axes on which the data will be plotted. If None, a new
            one will be created.
        label : string, optional
            How the series should be labeled in the figure legend.
        otherlabels : list of strings, optional
            A list of other legend labels that have already been plotted
            to ``ax``. If provided, ``label`` will be appended. If not
            provided, and new list will be created.
        artists : list of matplotlib artists, optional
            A list of other legend items that have already been plotted
            to ``ax``. If provided, the artist created will be appended.
            If not provided, and new list will be created.

        Returns
        -------
        fig : matplotlib.Figure
            The figure containing the plot.
        labels : list of strings
            Labels to be included in a legend for the figure.
        artists : list of matplotlib artists
            Symbology for the figure legend.

        """

        # setup the figure
        fig, ax = validate.axes(ax)

        if label is None:
            label = quantity

        # select the plot props based on the column
        try:
            meta = self.meta[quantity]
        except KeyError:
            raise KeyError(f"{quantity} not available")

        # plot the data
        self.data[quantity].fillna(0).plot(
            ax=ax, kind="area", color=meta["color"], alpha=meta["alpha"], zorder=5
        )

        if artists is not None:
            proxy = pyplot.Rectangle(
                (0, 0), 1, 1, facecolor=meta["color"], linewidth=0, alpha=meta["alpha"]
            )
            artists.append(proxy)
        if otherlabels is not None:
            otherlabels.append(label)

        return fig, otherlabels, artists

    def summaryPlot(
        self,
        axratio=2,
        filename=None,
        showLegend=True,
        precip=True,
        inflow=True,
        outflow=True,
        figopts={},
        serieslabels={},
    ):
        """
        Creates a figure showing the hydrlogic record (flow and
            precipitation) of the storm

        Input:
            axratio : optional float or int (default = 2)
                Relative height of the flow axis compared to the
                precipiation axis.

            filename : optional string (default = None)
                Filename to which the figure will be saved.

            **figwargs will be passed on to `pyplot.Figure`

        Writes:
            Figure of flow and precipitation for a storm

        Returns:
            None
        """
        fig = pyplot.figure(**figopts)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, axratio], hspace=0.12)
        rainax = fig.add_subplot(gs[0])
        rainax.yaxis.set_major_locator(pyplot.MaxNLocator(5))
        flowax = fig.add_subplot(gs[1], sharex=rainax)

        # create the legend proxy artists
        artists = []
        labels = []

        # in the label assignment: `serieslabels.pop(item, item)` might
        # seem odd. What it does is looks for a label (value) in the
        # dictionary with the key equal to `item`. If there is no valur
        # for that key in the dictionary the `item` itself is returned.
        # so if there's nothing called "test" in mydict,
        # `mydict.pop("test", "test")` returns `"test"`.
        if inflow:
            fig, labels, artists = self.plot_hydroquantity(
                self.inflowcol,
                ax=flowax,
                label=serieslabels.pop(self.inflowcol, self.inflowcol),
                otherlabels=labels,
                artists=artists,
            )

        if outflow:
            fig, labels, arti = self.plot_hydroquantity(
                self.outflowcol,
                ax=flowax,
                label=serieslabels.pop(self.outflowcol, self.outflowcol),
                otherlabels=labels,
                artists=artists,
            )

        if precip:
            fig, labels, arti = self.plot_hydroquantity(
                self.precipcol,
                ax=rainax,
                label=serieslabels.pop(self.precipcol, self.precipcol),
                otherlabels=labels,
                artists=artists,
            )
            rainax.invert_yaxis()

        if showLegend:
            leg = rainax.legend(
                artists,
                labels,
                fontsize=7,
                ncol=1,
                markerscale=0.75,
                frameon=False,
                loc="lower right",
            )
            leg.get_frame().set_zorder(25)
            _leg = [leg]
        else:
            _leg = None

        seaborn.despine(ax=rainax, bottom=True, top=False)
        seaborn.despine(ax=flowax)
        flowax.set_xlabel("")
        rainax.set_xlabel("")

        if filename is not None:
            fig.savefig(
                filename,
                dpi=300,
                transparent=True,
                bbox_inches="tight",
                bbox_extra_artists=_leg,
            )

        return fig, artists, labels


class HydroRecord:
    """Class representing an entire hydrologic record.

    Parameters
    ----------
    hydrodata : pandas.DataFrame
        DataFrame of hydrologic data of the storm. Should contain
        a unique index of type pandas.DatetimeIndex.
    precipcol : string, optional (default = None)
        Name of column in `hydrodata` containing precipiation data.
    inflowcol : string, optional (default = None)
        Name of column in `hydrodata` containing influent flow data.
    outflowcol : string, optional (default = None)
        Name of column in `hydrodata` containing effluent flow data.
    baseflowcol : string, optional (default = None)
        Name of column in `hydrodata` containing boolean indicating
        which records are considered baseflow.
    stormcol : string (default = 'storm')
        Name of column in `hydrodata` indentifying distinct storms.
    minprecip, mininflow, minoutflow : float, optional (default = 0)
        The minimum amount of each hydrologic quantity below which a
        storm can be considered "small".
    outputfreqMinutes : int, optional (default = 10)
        The default frequency (minutes) to which all data will be
        resampled. Precipitation data will be summed up across '
        multiple timesteps during resampling, while flow will be
        averaged.
    intereventHours : int, optional (default = 6)
        The dry duration (no flow or rain) required to signal the end of
        a storm.
    volume_conversion : float, optional (default = 1)
        Conversion factor to go from flow to volume for a single
        observation.
    stormclass : object, optional
        Defaults to wqio.hydro.Storm. Can be a subclass of that in cases
        where custom functionality is needed.
    lowmem : bool (default = False)
        If True, all dry observations are removed from the dataframe.

    """

    # TODO: rename `outputfreqMinutes` to `outputPeriodMinutes`
    def __init__(
        self,
        hydrodata,
        precipcol=None,
        inflowcol=None,
        outflowcol=None,
        baseflowcol=None,
        tempcol=None,
        stormcol="storm",
        minprecip=0.0,
        mininflow=0.0,
        minoutflow=0.0,
        outputfreqMinutes=10,
        intereventHours=6,
        volume_conversion=1,
        stormclass=None,
        lowmem=False,
    ):
        # validate input
        if precipcol is None and inflowcol is None and outflowcol is None:
            msg = "`hydrodata` must have at least a precip or in/outflow column"
            raise ValueError(msg)

        self.stormclass = stormclass or Storm

        # static input
        self._raw_data = hydrodata
        self.precipcol = precipcol
        self.inflowcol = inflowcol
        self.outflowcol = outflowcol
        self.baseflowcol = baseflowcol
        self.stormcol = stormcol
        self.tempcol = tempcol
        self.outputfreq = pandas.offsets.Minute(outputfreqMinutes)
        self.intereventHours = intereventHours
        self.intereventPeriods = MIN_PER_HOUR / self.outputfreq.n * self.intereventHours
        self.minprecip = minprecip
        self.mininflow = mininflow
        self.minoutflow = minoutflow
        self.volume_conversion = volume_conversion
        self.lowmem = lowmem

        # properties
        self._data = None
        self._all_storms = None
        self._storms = None
        self._storm_stats = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._define_storms()
            if self.lowmem:
                self._data = self._data[self._data[self.stormcol] != 0]

        return self._data

    @property
    def all_storms(self):
        if self._all_storms is None:
            self._all_storms = {}
            for storm_number in self.data[self.stormcol].unique():
                if storm_number > 0:
                    this_storm = self.stormclass(
                        self.data,
                        storm_number,
                        precipcol=self.precipcol,
                        inflowcol=self.inflowcol,
                        outflowcol=self.outflowcol,
                        tempcol=self.tempcol,
                        stormcol=self.stormcol,
                        volume_conversion=self.volume_conversion,
                        freqMinutes=self.outputfreq.n,
                    )
                    self._all_storms[storm_number] = this_storm

        return self._all_storms

    @property
    def storms(self):
        if self._storms is None:
            self._storms = {}
            for snum, storm in self.all_storms.items():
                is_small = storm.is_small(
                    minprecip=self.minprecip,
                    mininflow=self.mininflow,
                    minoutflow=self.minoutflow,
                )

                if not is_small:
                    self._storms[snum] = storm

        return self._storms

    @property
    def storm_stats(self):
        col_order = [
            "Storm Number",
            "Antecedent Days",
            "Season",
            "Start Date",
            "End Date",
            "Duration Hours",
            "Peak Precip Intensity",
            "Total Precip Depth",
            "Total Inflow Volume",
            "Peak Inflow",
            "Total Outflow Volume",
            "Peak Outflow",
            "Peak Lag Hours",
            "Centroid Lag Hours",
        ]
        if self._storm_stats is None:
            storm_stats = pandas.DataFrame([self.storms[sn].summary_dict for sn in self.storms])

            self._storm_stats = storm_stats[col_order]

        return self._storm_stats.sort_values(by=["Storm Number"]).reset_index(drop=True)

    def _define_storms(self, debug=False):
        parsed = parse_storm_events(
            self._raw_data,
            self.intereventHours,
            self.outputfreq.n,
            precipcol=self.precipcol,
            inflowcol=self.inflowcol,
            outflowcol=self.outflowcol,
            baseflowcol=self.baseflowcol,
            stormcol="storm",
            debug=debug,
        )
        return parsed

    def getStormFromTimestamp(self, timestamp, lookback_hours=0, smallstorms=False):
        """Get the storm associdated with a give (sample) date

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

        """

        # santize date input
        timestamp = validate.timestamp(timestamp)

        # check lookback hours
        if lookback_hours < 0:
            raise ValueError("`lookback_hours` must be greater than 0")

        # initial search for the storm
        storm_number = int(self.data.loc[:timestamp, self.stormcol].iloc[-1])

        # look backwards if we have too
        if (storm_number == 0 or pandas.isnull(storm_number)) and lookback_hours != 0:
            lookback_time = timestamp - pandas.offsets.Hour(lookback_hours)
            storms = self.data.loc[lookback_time:timestamp, [self.stormcol]]
            storms = storms[storms > 0].dropna()

            storm_number = None if storms.shape[0] == 0 else int(storms.iloc[-1])

        # return storm_number and storms
        if smallstorms:
            return storm_number, self.all_storms.get(storm_number, None)
        else:
            return storm_number, self.storms.get(storm_number, None)

    def histogram(self, valuecol, bins, **factoropts):
        """Plot a faceted, categorical histogram of storms.

        Parameters
        ----------
        valuecol : str, optional
            The name of the column that should be categorized and plotted.
        bins : array-like, optional
            The right-edges of the histogram bins.
        factoropts : keyword arguments, optional
            Options passed directly to seaborn.factorplot

        Returns
        -------
        fig : seaborn.FacetGrid

        See also
        --------
        viz.categorical_histogram
        seaborn.factorplot

        """

        fg = viz.categorical_histogram(self.storm_stats, valuecol, bins, **factoropts)
        fg.figure.tight_layout()
        return fg


class DrainageArea:
    def __init__(self, total_area=1.0, imp_area=1.0, bmp_area=0.0):
        """A simple object representing the drainage area of a BMP.

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

    def simple_method(self, storm_depth, volume_conversion=1.0, annual_factor=1.0):
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
        annual_factor : float, optional (default = 1.0)
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
        runoff_volume = (drainage_conversion * annual_factor + bmp_conversion) * storm_depth
        return runoff_volume
