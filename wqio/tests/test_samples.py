import pytest
from wqio.tests import helpers

import pandas

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

from wqio import samples


BASELINE_IMAGES = '_baseline_images/samples_tests'


class base_wqsampleMixin(object):
    def basic_setup(self):
        self.known_wqdata_type = pandas.DataFrame
        self.known_starttime_type = pandas.Timestamp
        self.known_endtime_type = pandas.Timestamp
        self.known_sample_ts_type = pandas.DatetimeIndex
        self.known_linestyle = 'none'
        datafile = helpers.test_data_path('test_wqsample_data.csv')
        self.rawdata = pandas.read_csv(datafile, index_col=[0,1,2,3,4,5,6,11,12])

    def basic_teardown(self):
        pyplot.close('all')

    def test_wqdata(self):
        assert hasattr(self.wqs, 'wqdata')
        assert isinstance(self.wqs.wqdata, self.known_wqdata_type)

    def test_starttime(self):
        assert hasattr(self.wqs, 'starttime')
        assert isinstance(self.wqs.starttime, self.known_starttime_type)

    def test_endtime(self):
        assert hasattr(self.wqs, 'endtime')
        assert isinstance(self.wqs.endtime, self.known_endtime_type)

    def test_season(self):
        assert hasattr(self.wqs, 'season')
        assert self.wqs.season == self.known_season

    def test_season_setter(self):
        self.wqs.season = 'spring'
        assert self.wqs.season == 'spring'

    def test_samplefreq(self):
        assert hasattr(self.wqs, 'samplefreq')
        assert self.wqs.samplefreq == self.known_samplefreq

    def test_sample_ts_form(self):
        assert hasattr(self.wqs, 'sample_ts')
        assert isinstance(self.wqs.sample_ts, self.known_sample_ts_type)

    def test_sample_ts_start(self):
        assert self.wqs.sample_ts[0] == pandas.Timestamp(self.known_starttime)

    def test_sample_ts_end(self):
        assert self.wqs.sample_ts[-1] == pandas.Timestamp(self.known_endtime)

    def test_sample_ts_freq(self):
        assert self.wqs.sample_ts.freq == self.known_ts_samplefreq

    def test_sample_ts_len(self):
        assert len(self.wqs.sample_ts) == self.known_sample_ts_len

    def test_label(self):
        assert hasattr(self.wqs, 'label')
        assert self.wqs.label == self.known_label

    def test_marker(self):
        assert hasattr(self.wqs, 'marker')
        assert self.wqs.marker == self.known_marker

    def test_linestyle(self):
        assert hasattr(self.wqs, 'linestyle')
        assert self.wqs.linestyle == self.known_linestyle

    def test_yfactor(self):
        assert hasattr(self.wqs, 'yfactor')
        assert self.wqs.yfactor == self.known_yfactor


class base_wqsample_NoStorm(base_wqsampleMixin):
    def test_storm(self):
        assert hasattr(self.wqs, 'storm')
        assert self.wqs.storm is None


class base_wqsample_WithStorm(base_wqsampleMixin):
    def test_storm(self):
        assert hasattr(self.wqs, 'storm')
        assert isinstance(self.wqs.storm, samples.Storm)


class Test_GrabSample_NoStorm(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'GrabNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:54'
        self.known_endtime = '2013-02-24 16:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 2
        self.known_ts_samplefreq = pandas.offsets.Minute(5)
        self.known_samplefreq = None

        self.known_marker = '+'
        self.known_label = 'Grab Sample'
        self.wqs = samples.GrabSample(self.rawdata, self.known_starttime,
                                     endtime=self.known_endtime, storm=None)

    def teardown(self):
        self.basic_teardown()


class Test_CompositeSample_NoStorm(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'CompositeNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 31
        self.known_samplefreq = pandas.offsets.Minute(20)
        self.known_ts_samplefreq = pandas.offsets.Minute(20)
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.wqs = samples.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)

    def teardown(self):
        self.basic_teardown()


class Test_CompositeSample_NoStormNoFreq(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'CompositeNoStormnoFreq'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 2
        self.known_samplefreq = pandas.offsets.Hour(10)
        self.known_ts_samplefreq = pandas.offsets.Hour(10)
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.wqs = samples.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)

    def teardown(self):
        self.basic_teardown()


def setup_sample(sampletype, with_storm=False):
    datafile = helpers.test_data_path('test_wqsample_data.csv')
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
    xlims = (pandas.Timestamp('2013-02-24 16:45'), pandas.Timestamp('2013-02-25 03:30'))

    return wqs, xlims


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_grabsample_noStorm_notFocus():
    wqs, xlims = setup_sample('grab', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=False)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_grabsample_noStorm_notFocus_asRug():
    wqs, xlims = setup_sample('grab', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=False, asrug=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_grabsample_noStorm_asFocus():
    wqs, xlims = setup_sample('grab', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_grabsample_noStorm_Focus_asRug():
    wqs, xlims = setup_sample('grab', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=True, asrug=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_compsample_noStorm_notFocus():
    wqs, xlims = setup_sample('composite', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=False)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_compsample_noStorm_notFocus_asRug():
    wqs, xlims = setup_sample('composite', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=False, asrug=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_compsample_noStorm_Focus():
    wqs, xlims = setup_sample('composite', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES)
def test_plot_compsample_noStorm_Focus_asRug():
    wqs, xlims = setup_sample('composite', with_storm=False)
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    wqs.plot_ts(ax, isFocus=True, asrug=True)
    return fig
