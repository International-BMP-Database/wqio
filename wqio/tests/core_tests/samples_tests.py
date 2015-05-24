import os
import sys
import datetime

import numpy as np
import pandas

import nose.tools as nt
import numpy.testing as nptest
import pandas.util.testing as pdtest

from wqio import testing
from wqio.testing.testutils import assert_timestamp_equal, setup_prefix
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = False
from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt
import seaborn

from wqio.core import samples
from wqio import utils


class base_wqsampleMixin(object):
    @nt.nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nt.nottest
    def basic_setup(self):
        self.prefix = setup_prefix('core_tests')
        self.fig, self.ax = plt.subplots()
        self.known_wqdata_type = pandas.DataFrame
        self.known_starttime_type = pandas.Timestamp
        self.known_endtime_type = pandas.Timestamp
        self.known_sample_ts_type = pandas.DatetimeIndex
        self.known_linestyle = 'none'
        datafile = os.path.join(sys.prefix, 'wqio_data', 'testing', 'test_wqsample_data.csv')
        self.rawdata = pandas.read_csv(datafile, index_col=[0,1,2,3,4,5,6,11,12])

    @nt.nottest
    def basic_teardown(self):
        plt.close('all')

    def test_wqdata(self):
        nt.assert_true(hasattr(self.wqs, 'wqdata'))
        nt.assert_true(isinstance(self.wqs.wqdata, self.known_wqdata_type))

    def test_starttime(self):
        nt.assert_true(hasattr(self.wqs, 'starttime'))
        nt.assert_true(isinstance(self.wqs.starttime, self.known_starttime_type))

    def test_endtime(self):
        nt.assert_true(hasattr(self.wqs, 'endtime'))
        nt.assert_true(isinstance(self.wqs.endtime, self.known_endtime_type))

    def test_season(self):
        nt.assert_true(hasattr(self.wqs, 'season'))
        nt.assert_equal(self.wqs.season, self.known_season)

    def test_season_setter(self):
        self.wqs.season = 'spring'
        nt.assert_equal(self.wqs.season, 'spring')

    def test_samplefreq(self):
        nt.assert_true(hasattr(self.wqs, 'samplefreq'))
        nt.assert_true(isinstance(self.wqs.samplefreq, self.known_samplefreq_type))
        nt.assert_equal(self.wqs.samplefreq, self.known_samplefreq)

    def test_sample_ts_form(self):
        nt.assert_true(hasattr(self.wqs, 'sample_ts'))
        nt.assert_true(isinstance(self.wqs.sample_ts, self.known_sample_ts_type))

    def test_sample_ts_start(self):
        nt.assert_equal(self.wqs.sample_ts[0], pandas.Timestamp(self.known_starttime))

    def test_sample_ts_end(self):
        nt.assert_equal(self.wqs.sample_ts[-1], pandas.Timestamp(self.known_endtime))

    def test_sample_ts_freq(self):
        nt.assert_equal(self.wqs.sample_ts.freq, self.known_samplefreq)

    def test_sample_ts_len(self):
        nt.assert_equal(len(self.wqs.sample_ts), self.known_sample_ts_len)

    def test_label(self):
        nt.assert_true(hasattr(self.wqs, 'label'))
        nt.assert_equal(self.wqs.label, self.known_label)

    def test_marker(self):
        nt.assert_true(hasattr(self.wqs, 'marker'))
        nt.assert_equal(self.wqs.marker, self.known_marker)

    def test_linestyle(self):
        nt.assert_true(hasattr(self.wqs, 'linestyle'))
        nt.assert_equal(self.wqs.linestyle, self.known_linestyle)

    def test_yfactor(self):
        nt.assert_true(hasattr(self.wqs, 'yfactor'))
        nt.assert_equal(self.wqs.yfactor, self.known_yfactor)


class base_wqsample_NoStorm(base_wqsampleMixin):
    def test_storm(self):
        nt.assert_true(hasattr(self.wqs, 'storm'))
        nt.assert_true(self.wqs.storm is None)


class base_wqsample_WithStorm(base_wqsampleMixin):
    def test_storm(self):
        nt.assert_true(hasattr(self.wqs, 'storm'))
        nt.assert_true(isinstance(self.wqs.storm, samples.Storm))


class test_GrabSample_NoStorm(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'GrabNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:54'
        self.known_endtime = '2013-02-24 16:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 2
        self.known_samplefreq = None
        self.known_samplefreq_type = type(None)

        self.known_marker = '+'
        self.known_label = 'Grab Sample'
        self.wqs = samples.GrabSample(self.rawdata, self.known_starttime,
                                     endtime=self.known_endtime, storm=None)

    def teardown(self):
        self.basic_teardown()


class test_CompositeSample_NoStorm(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'CompositeNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 31
        self.known_samplefreq = pandas.tseries.offsets.Minute(20)
        self.known_samplefreq_type = pandas.tseries.offsets.Minute
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.wqs = samples.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)

    def teardown(self):
        self.basic_teardown()
        pass


class test_CompositeSample_NoStormNoFreq(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'CompositeNoStormnoFreq'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 2
        self.known_samplefreq = None
        self.known_samplefreq_type = type(None)
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.wqs = samples.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)

    def teardown(self):
        self.basic_teardown()


@nt.nottest
def setup_sample(sampletype, with_storm=False):
    datafile = os.path.join(sys.prefix, 'wqio_data', 'testing', 'test_wqsample_data.csv')
    rawdata = pandas.read_csv(datafile, index_col=[0,1,2,3,4,5,6,11,12])
    if sampletype.lower() =='grab':
        st = samples.GrabSample
        starttime = '2013-02-24 16:54'
        endtime = '2013-02-24 16:59'
        freq = None
    elif sampletype.lower() =='composite':
        st = samples.CompositeSample
        starttime = '2013-02-24 16:59'
        endtime = '2013-02-25 02:59'
        freq = pandas.offsets.Minute(20)

    if with_storm:
        storm = setup_storm()
    else:
        storm = None

    wqs = st(rawdata, starttime, endtime=endtime, samplefreq=freq, storm=storm)
    wqs.marker = 'D'
    wqs.markersize = 8

    return wqs


@image_comparison(
    baseline_images=['test_Grab_without_storm', 'test_Grab_without_storm_rug'],
    extensions=['png']
)
def test_plot_grabsample_no_storm_not_focus():
    wqs = setup_sample('grab', with_storm=False)
    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=False)

    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=False, asrug=True)


@image_comparison(
    baseline_images=['test_Grab_without_storm_focus', 'test_Grab_without_storm_focus_rug'],
    extensions=['png']
)
def test_plot_grabsample_no_storm_focus():
    wqs = setup_sample('grab', with_storm=False)
    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=True)

    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=True, asrug=True)


@image_comparison(
    baseline_images=['test_Comp_without_storm', 'test_Comp_without_storm_rug'],
    extensions=['png']
)
def test_plot_compsample_no_storm_not_focus():
    wqs = setup_sample('composite', with_storm=False)
    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=False)

    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=False, asrug=True)


@image_comparison(
    baseline_images=['test_Comp_without_storm_focus', 'test_Comp_without_storm_focus_rug'],
    extensions=['png']
)
def test_plot_compsample_no_storm_focus():
    wqs = setup_sample('composite', with_storm=False)
    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=True)

    fig, ax = plt.subplots()
    wqs.plot_ts(ax, isFocus=True, asrug=True)
