import pandas
import seaborn
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters

from wqio import utils

register_matplotlib_converters()


class Parameter:
    def __init__(self, name, units, usingTex=False):
        """Class representing a single analytical parameter (pollutant).

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
        """Creates a string representation of the parameter and units.

        Parameters
        ----------
        usecomma : bool, optional (default = False)
            Toggles the format of the returned string attribute. If True
            the returned format is "<parameter>, <unit>". Otherwise the
            format is "<parameter> (<unit>)".
        """

        paramunit = "{0}, {1}" if usecomma else "{0} ({1})"

        n = self.name
        u = self.units

        return paramunit.format(n, u)

    def __repr__(self):
        return f"<wqio Parameter object> ({self.paramunit(usecomma=False)})"

    def __str__(self):
        return f"<wqio Parameter object> ({self.paramunit(usecomma=False)})"


class SampleMixin:
    def __init__(
        self,
        dataframe,
        starttime,
        samplefreq=None,
        endtime=None,
        storm=None,
        rescol="res",
        qualcol="qual",
        dlcol="DL",
        unitscol="units",
    ):
        self._wqdata = dataframe
        self._startime = pandas.Timestamp(starttime)
        self._endtime = pandas.Timestamp(endtime)
        self._samplefreq = samplefreq
        self._sample_ts = None
        self._label = None
        self._marker = None
        self._markersize = None
        self._linestyle = None
        self._yfactor = None
        self._season = utils.get_season(self.starttime)
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
            self._linestyle = "none"
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        self._linestyle = value

    @property
    def markersize(self):
        if self._markersize is None:
            self._markersize = 4
        return self._markersize

    @markersize.setter
    def markersize(self, value):
        self._markersize = value

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
            alpha = 0.75 if isFocus else 0.35

        ymax = ax.get_ylim()[-1]
        yposition = [self.yfactor * ymax] * len(self.sample_ts)

        timeseries = pandas.Series(yposition, index=self.sample_ts)

        if asrug:
            seaborn.rugplot(self.sample_ts, ax=ax, color="black", alpha=alpha, mew=0.75)
            line = pyplot.Line2D(
                [0, 0],
                [0, 0],
                marker="|",
                mew=0.75,
                color="black",
                alpha=alpha,
                linestyle="none",
            )

        else:
            timeseries.plot(
                ax=ax,
                marker=self.marker,
                markersize=4,
                linestyle=self.linestyle,
                color="Black",
                zorder=10,
                label="_nolegend",
                alpha=alpha,
                mew=0.75,
            )
            line = pyplot.Line2D(
                [0, 0],
                [0, 0],
                marker=self.marker,
                mew=0.75,
                color="black",
                alpha=alpha,
                linestyle="none",
            )

        return line


class CompositeSample(SampleMixin):
    """Class for composite samples"""

    @property
    def label(self):
        if self._label is None:
            self._label = "Composite Sample"
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def marker(self):
        if self._marker is None:
            self._marker = "x"
        return self._marker

    @marker.setter
    def marker(self, value):
        self._marker = value

    @property
    def sample_ts(self):
        if self.starttime is not None and self.endtime is not None:
            _sampfreq = self.samplefreq or self.endtime - self.starttime
            self._sample_ts = pandas.date_range(
                start=self.starttime, end=self.endtime, freq=_sampfreq
            )
        return self._sample_ts


class GrabSample(SampleMixin):
    """Class for grab (discrete) samples"""

    @property
    def label(self):
        if self._label is None:
            self._label = "Grab Sample"
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def marker(self):
        if self._marker is None:
            self._marker = "+"
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
                self._sample_ts = pandas.date_range(
                    start=self.starttime,
                    end=self.endtime,
                    freq=self.endtime - self.starttime,
                )
        return self._sample_ts


_basic_doc = """ {} water quality sample

Container to hold water quality results from many different pollutants
collected at a single point in time.

Parameters
----------
dataframe : pandas.DataFrame
    The water quality data.
starttime : datetime-like
    The date and time at which the sample collection started.
samplefreq : string, optional
    A valid pandas timeoffset string specifying the frequency with which
    sample aliquots are collected.
endtime : datetime-like, optional
    The date and time at which sample collection ended.
storm : wqio.Storm object, optional
    A Storm object (or a subclass or Storm) that triggered sample
    collection.
rescol, qualcol, dlcol, unitscol : string, optional
    Strings that define the column labels for the results, qualifiers,
    detection limits, and units if measure, respectively.

"""


SampleMixin.__doc__ = _basic_doc.format("Basic")
CompositeSample.__doc__ = _basic_doc.format("Composite")
GrabSample.__doc__ = _basic_doc.format("Grab")
