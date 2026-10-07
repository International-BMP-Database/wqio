from datetime import datetime
from typing import Any

import pandas
import seaborn
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas.plotting import register_matplotlib_converters

from wqio import utils

register_matplotlib_converters()


class Parameter:
    def __init__(self, name: str, units: str, usingTex: bool = False) -> None:
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
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def units(self) -> str:
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        self._units = value

    @property
    def usingTex(self) -> bool:
        return self._usingTex

    @usingTex.setter
    def usingTex(self, value: bool) -> None:
        if value in (True, False):
            self._usingTex = value
        else:
            raise ValueError("`usingTex` must be of type `bool`")

    def paramunit(self, usecomma: bool = False) -> str:
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

    def __repr__(self) -> str:
        return f"<wqio Parameter object> ({self.paramunit(usecomma=False)})"

    def __str__(self) -> str:
        return f"<wqio Parameter object> ({self.paramunit(usecomma=False)})"


class SampleMixin:
    # provided by the subclasses
    sample_ts: pandas.DatetimeIndex | None
    marker: str

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        starttime: datetime | str,
        samplefreq: str | pandas.Timedelta | None = None,
        endtime: datetime | str | None = None,
        storm: Any = None,
        rescol: str = "res",
        qualcol: str = "qual",
        dlcol: str = "DL",
        unitscol: str = "units",
    ) -> None:
        self._wqdata = dataframe
        self._startime: Any = pandas.Timestamp(starttime)
        self._endtime: Any = pandas.Timestamp(endtime)  # ty: ignore[invalid-argument-type]
        self._samplefreq = samplefreq
        self._sample_ts: pandas.DatetimeIndex | None = None
        self._label: str | None = None
        self._marker: str | None = None
        self._markersize: int | float | None = None
        self._linestyle: str | None = None
        self._yfactor: float | None = None
        self._season = utils.get_season(self.starttime)
        self.storm = storm

    @property
    def season(self) -> str:
        return self._season

    @season.setter
    def season(self, value: str) -> None:
        self._season = value

    @property
    def wqdata(self) -> pandas.DataFrame:
        return self._wqdata

    @wqdata.setter
    def wqdata(self, value: pandas.DataFrame) -> None:
        self._wqdata = value

    @property
    def starttime(self) -> pandas.Timestamp:
        return self._startime

    @starttime.setter
    def starttime(self, value: pandas.Timestamp) -> None:
        self._startime = value

    @property
    def endtime(self) -> pandas.Timestamp:
        if self._endtime is None:
            self._endtime = self._startime
        return self._endtime

    @endtime.setter
    def endtime(self, value: pandas.Timestamp | None) -> None:
        self._endtime = value

    @property
    def samplefreq(self) -> str | pandas.Timedelta | None:
        return self._samplefreq

    @samplefreq.setter
    def samplefreq(self, value: str | pandas.Timedelta | None) -> None:
        self._samplefreq = value

    @property
    def linestyle(self) -> str:
        if self._linestyle is None:
            self._linestyle = "none"
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value: str | None) -> None:
        self._linestyle = value

    @property
    def markersize(self) -> int | float:
        if self._markersize is None:
            self._markersize = 4
        return self._markersize

    @markersize.setter
    def markersize(self, value: int | float | None) -> None:
        self._markersize = value

    @property
    def yfactor(self) -> float:
        if self._yfactor is None:
            self._yfactor = 0.25
        return self._yfactor

    @yfactor.setter
    def yfactor(self, value: float | None) -> None:
        self._yfactor = value

    def plot_ts(self, ax: Axes, isFocus: bool = True, asrug: bool = False) -> Line2D:
        if self.sample_ts is not None:
            alpha = 0.75 if isFocus else 0.35

        ymax = ax.get_ylim()[-1]
        yposition = [self.yfactor * ymax] * len(self.sample_ts)  # ty: ignore[invalid-argument-type]

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
    def label(self) -> str:
        if self._label is None:
            self._label = "Composite Sample"
        return self._label

    @label.setter
    def label(self, value: str | None) -> None:
        self._label = value

    @property
    def marker(self) -> str:
        if self._marker is None:
            self._marker = "x"
        return self._marker

    @marker.setter
    def marker(self, value: str | None) -> None:
        self._marker = value

    @property
    def sample_ts(self) -> pandas.DatetimeIndex | None:
        if self.starttime is not None and self.endtime is not None:
            _sampfreq = self.samplefreq or self.endtime - self.starttime
            self._sample_ts = pandas.date_range(
                start=self.starttime, end=self.endtime, freq=_sampfreq
            )
        return self._sample_ts


class GrabSample(SampleMixin):
    """Class for grab (discrete) samples"""

    @property
    def label(self) -> str:
        if self._label is None:
            self._label = "Grab Sample"
        return self._label

    @label.setter
    def label(self, value: str | None) -> None:
        self._label = value

    @property
    def marker(self) -> str:
        if self._marker is None:
            self._marker = "+"
        return self._marker

    @marker.setter
    def marker(self, value: str | None) -> None:
        self._marker = value

    @property
    def sample_ts(self) -> pandas.DatetimeIndex | None:
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
