import sys

import matplotlib.pyplot as plt
import seaborn.apionly as seaborn
import pandas

from wqio import utils


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
        self._markersize = None
        self._linestyle = None
        self._yfactor = None
        self._season = utils.getSeason(self.starttime)
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
            if isFocus:
                alpha = 0.75
            else:
                alpha = 0.35

        ymax = ax.get_ylim()[-1]
        yposition = [self.yfactor * ymax] * len(self.sample_ts)

        timeseries = pandas.Series(yposition, index=self.sample_ts)

        if asrug:
            seaborn.rugplot(self.sample_ts, ax=ax, color='black', alpha=alpha, mew=0.75)
            line = plt.Line2D([0, 0], [0, 0], marker='|', mew=0.75,
                              color='black', alpha=alpha, linestyle='none')

        else:
            timeseries.plot(ax=ax, marker=self.marker, markersize=4,
                            linestyle=self.linestyle, color='Black',
                            zorder=10, label='_nolegend', alpha=alpha)
            line = plt.Line2D([0, 0], [0, 0], marker=self.marker, mew=0.75,
                              color='black', alpha=alpha, linestyle='none')

        return line


class CompositeSample(_basic_wq_sample):
    """ Class for composite samples """
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
    """ Class for grab (discrete) samples """
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

if sys.version_info.major >= 3:
    _basic_wq_sample.__doc__  = _basic_doc.format("Basic")
    CompositeSample.__doc__ = _basic_doc.format("Composite")
    GrabSample.__doc__ = _basic_doc.format("Grab")
