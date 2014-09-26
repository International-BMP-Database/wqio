import os

from nose.tools import *
import numpy as np
import numpy.testing as nptest

import testing
from testing.testutils import assert_timestamp_equal
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex
import matplotlib.pyplot as plt
import pandas

from core import events
import utils

class base_wqsampleMixin(object):
    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nottest
    def basic_setup(self):
        if os.path.split(os.getcwd())[-1] == 'src':
            self.prefix = os.path.join('.', 'core', 'tests', 'result_images')
        else:
            self.prefix = os.path.join('..', 'core', 'tests', 'result_images')
        self.fig, self.ax = plt.subplots()
        self.known_wqdata_type = pandas.DataFrame
        self.known_starttime_type = pandas.Timestamp
        self.known_endtime_type = pandas.Timestamp
        self.known_sample_ts_type = pandas.DatetimeIndex
        self.known_linestyle = 'none'
        datafile = os.path.join('testing', 'data', 'test_wqsample_data.csv')
        self.rawdata = pandas.read_csv(datafile, index_col=[0,1,2,3,4,5,6,11,12])

    @nottest
    def basic_teardown(self):
        plt.close('all')

    def test_wqdata(self):
        assert_true(hasattr(self.wqs, 'wqdata'))
        assert_true(isinstance(self.wqs.wqdata, self.known_wqdata_type))

    def test_starttime(self):
        assert_true(hasattr(self.wqs, 'starttime'))
        assert_true(isinstance(self.wqs.starttime, self.known_starttime_type))

    def test_endtime(self):
        assert_true(hasattr(self.wqs, 'endtime'))
        assert_true(isinstance(self.wqs.endtime, self.known_endtime_type))

    def test_samplefreq(self):
        assert_true(hasattr(self.wqs, 'samplefreq'))
        assert_true(isinstance(self.wqs.samplefreq, self.known_samplefreq_type))
        assert_equal(self.wqs.samplefreq, self.known_samplefreq)

    def test_sample_ts_form(self):
        assert_true(hasattr(self.wqs, 'sample_ts'))
        assert_true(isinstance(self.wqs.sample_ts, self.known_sample_ts_type))

    def test_sample_ts_start(self):
        assert_equal(self.wqs.sample_ts[0], pandas.Timestamp(self.known_starttime))

    def test_sample_ts_end(self):
        assert_equal(self.wqs.sample_ts[-1], pandas.Timestamp(self.known_endtime))

    def test_sample_ts_freq(self):
        assert_equal(self.wqs.sample_ts.freq, self.known_samplefreq)

    def test_sample_ts_len(self):
        assert_equal(len(self.wqs.sample_ts), self.known_sample_ts_len)

    def test_label(self):
        assert_true(hasattr(self.wqs, 'label'))
        assert_equal(self.wqs.label, self.known_label)

    def test_marker(self):
        assert_true(hasattr(self.wqs, 'marker'))
        assert_equal(self.wqs.marker, self.known_marker)

    def test_linestyle(self):
        assert_true(hasattr(self.wqs, 'linestyle'))
        assert_equal(self.wqs.linestyle, self.known_linestyle)

    def test_yfactor(self):
        assert_true(hasattr(self.wqs, 'yfactor'))
        assert_equal(self.wqs.yfactor, self.known_yfactor)

    def test_plot_ts_isFocus(self):
        self.wqs.plot_ts(self.ax, isFocus=True)
        self.fig.savefig(self.makePath('{0}_{1}_wqsample_isFocus.png'.format(self.wqs.label, self.test_name)))

    def test_plot_ts_isNotFocus(self):
        self.wqs.plot_ts(self.ax, isFocus=False)
        self.fig.savefig(self.makePath('{0}_{1}_wqsample_isNotFocus.png'.format(self.wqs.label, self.test_name)))


class base_wqsample_NoStorm(base_wqsampleMixin):
    def test_storm(self):
        assert_true(hasattr(self.wqs, 'storm'))
        assert_true(self.wqs.storm is None)


class base_wqsample_WithStorm(base_wqsampleMixin):
    def test_storm(self):
        assert_true(hasattr(self.wqs, 'storm'))
        assert_true(isinstance(self.wqs.storm, events.Storm))


class test_GrabSample_NoStorm(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'GrabNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:54'
        self.known_endtime = '2013-02-24 16:59'
        self.known_sample_ts_len = 2
        self.known_samplefreq = None
        self.known_samplefreq_type = type(None)

        self.known_marker = '+'
        self.known_label = 'Grab Sample'
        self.wqs = events.GrabSample(self.rawdata, self.known_starttime,
                                     endtime=self.known_endtime, storm=None)

    def teardown(self):
        self.basic_teardown()
        pass


class test_CompositeSample_NoStorm(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'CompositeNoStorm'
        self.known_yfactor = 0.35
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_sample_ts_len = 31
        self.known_samplefreq = pandas.tseries.offsets.Minute(20)
        self.known_samplefreq_type = pandas.tseries.offsets.Minute
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.wqs = events.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)
        pass

    def teardown(self):
        self.basic_teardown()
        pass


class test_CompositeSample_NoStormNoFreq(base_wqsample_NoStorm):
    def setup(self):
        self.basic_setup()
        self.test_name = 'CompositeNoStormnoFreq'
        self.known_yfactor = 0.35
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_sample_ts_len = 2
        self.known_samplefreq = None
        self.known_samplefreq_type = type(None)
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.wqs = events.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)
        pass

    def teardown(self):
        self.basic_teardown()
        pass


class base_defineStormsMixin(object):
    def teardown(self):
        pass

    def test_check_type_and_columns(self):
        assert_true(isinstance(self.parsed_record, pandas.DataFrame))
        for col in self.known_std_columns:
            assert_true(col in self.parsed_record.columns.tolist())

    @raises(ValueError)
    def test_defineStorm_no_columns(self):
        data = events.defineStorms(self.orig_record)

    def test_check_nan_col(self):
        assert_true(np.all(np.isnan(self.parsed_record['outflow'])))

    def test_number_storms(self):
        last_storm = self.parsed_record['storm'].max()
        assert_equal(last_storm, self.known_number_of_storms)

    def test_first_storm_start(self):
        storm1 = self.parsed_record[self.parsed_record['storm'] == 1]
        assert_timestamp_equal(storm1.index[0], self.known_storm1_start)

    def test_first_storm_end(self):
        storm1 = self.parsed_record[self.parsed_record['storm'] == 1]
        assert_timestamp_equal(storm1.index[-1], self.known_storm1_end)

    def test_second_storm_start(self):
        storm2 = self.parsed_record[self.parsed_record['storm'] == 2]
        assert_timestamp_equal(storm2.index[0], self.known_storm2_start)

    def test_second_storm_end(self):
        storm2 = self.parsed_record[self.parsed_record['storm'] == 2]
        assert_timestamp_equal(storm2.index[-1], self.known_storm2_end)

    def test_index_type(self):
        assert_true(isinstance(self.parsed_record.index, pandas.DatetimeIndex))


class test_defineStorms_Simple(base_defineStormsMixin):
    def setup(self):
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = os.path.join(
            '.', 'testing', 'data', 'teststorm_simple.csv'
        )

        self.known_std_columns = ['precip', 'inflow', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.parsed_record = events.defineStorms(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5
        )


class test_defineStorms_Singular(base_defineStormsMixin):
    def setup(self):
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 07:00')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 07:05')
        self.storm_file = os.path.join(
            '.', 'testing', 'data', 'teststorm_singular.csv'
        )
        self.known_std_columns = ['precip', 'inflow', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.parsed_record = events.defineStorms(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5
        )


class test_defineStorms_FirstObservation(base_defineStormsMixin):
    def setup(self):
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 12:05')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = os.path.join(
            '.', 'testing', 'data', 'teststorm_firstobs.csv'
        )
        self.known_std_columns = ['precip', 'inflow', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.parsed_record = events.defineStorms(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5
        )


class test_storm:
    def setup(self):
        # path stuff
        if os.path.split(os.getcwd())[-1] == 'src':
            self.prefix = os.path.join('.', 'core', 'tests', 'result_images')
        else:
            self.prefix = os.path.join('..', 'core', 'tests', 'result_images')


        self.storm_file = os.path.join('.', 'testing', 'data', 'teststorm_simple.csv')
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.parsed_record = events.defineStorms(self.orig_record,
                                                precipcol='rain',
                                                inflowcol='influent',
                                                outflowcol='effluent',
                                                outputfreqMinutes=5,
                                                intereventperiods=24)
        self.storm = events.Storm(self.parsed_record, 2)

        self.known_columns = ['precip', 'inflow', 'outflow', 'storm']
        self.known_index_type = pandas.DatetimeIndex
        self.known_storm_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm_end = pandas.Timestamp('2013-05-19 11:55')
        self.known_precip_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_precip_end = pandas.Timestamp('2013-05-19 09:10')
        self.known_inflow_start = pandas.Timestamp('2013-05-19 06:25')
        self.known_inflow_end = pandas.Timestamp('2013-05-19 11:30')
        self.known_outflow_start = pandas.Timestamp('2013-05-19 06:50')
        self.known_outflow_end = pandas.Timestamp('2013-05-19 11:50')
        self.known_duration_hours = 5.75
        self.known_antecedent_period_days = 0.17708333
        self.known_peak_precip_intensity = 0.11
        self.known_peak_inflow = 123.0
        self.known_peak_outflow = 41.0
        self.known_peak_precip_intensity_time = pandas.Timestamp('2013-05-19 08:00')
        self.known_peak_inflow_time = pandas.Timestamp('2013-05-19 08:35')
        self.known_peak_outflow_time = pandas.Timestamp('2013-05-19 08:45')
        self.known_centroid_precip = pandas.Timestamp('2013-05-19 07:41:58.705023')
        self.known_centroid_inflow = pandas.Timestamp('2013-05-19 08:44:31.252553')
        self.known_centroid_outflow = pandas.Timestamp('2013-05-19 08:59:51.675231')
        self.known_total_precip_depth = 1.39
        self.known_total_inflow_volume = 876600
        self.known_total_outflow_volume = 291900
        self.known_peak_lag_time = 0.1666666667
        self.known_centroid_lag_time = 0.255672966

        self.fig, self.ax = plt.subplots()

    def teardown(self):
        plt.close(self.fig)
        pass

    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    def test_columns(self):
        assert_list_equal(self.known_columns, self.storm.data.columns.tolist())

    def test_index_type(self):
        assert_true(isinstance(self.storm.data.index, pandas.DatetimeIndex))

    def test_storm_start(self):
        assert_true(hasattr(self.storm, 'storm_start'))
        assert_timestamp_equal(self.storm.storm_start, self.known_storm_start)

    def test_storm_end(self):
        assert_true(hasattr(self.storm, 'storm_end'))
        assert_timestamp_equal(self.storm.storm_end, self.known_storm_end)

    def test_precip_start(self):
        assert_true(hasattr(self.storm, 'precip_start'))
        assert_timestamp_equal(self.storm.precip_start, self.known_precip_start)

    def test_precip_end(self):
        assert_true(hasattr(self.storm, 'precip_end'))
        assert_timestamp_equal(self.storm.precip_end, self.known_precip_end)

    def test_inflow_start(self):
        assert_true(hasattr(self.storm, 'inflow_start'))
        assert_timestamp_equal(self.storm.inflow_start, self.known_inflow_start)

    def test_inflow_end(self):
        assert_true(hasattr(self.storm, 'inflow_end'))
        assert_timestamp_equal(self.storm.inflow_end, self.known_inflow_end)

    def test_outflow_start(self):
        assert_true(hasattr(self.storm, 'outflow_start'))
        assert_timestamp_equal(self.storm.outflow_start, self.known_outflow_start)

    def test_outflow_end(self):
        assert_true(hasattr(self.storm, 'outflow_end'))
        assert_timestamp_equal(self.storm.outflow_end, self.known_outflow_end)

    def test_duration_hours(self):
        assert_true(hasattr(self.storm, 'duration_hours'))
        nptest.assert_almost_equal(self.storm.duration_hours, self.known_duration_hours)

    def test_antecedent_period_days(self):
        assert_true(hasattr(self.storm, 'antecedent_period_days'))
        nptest.assert_almost_equal(self.storm.antecedent_period_days, self.known_antecedent_period_days)

    def test_peak_precip_intensity(self):
        assert_true(hasattr(self.storm, 'peak_precip_intensity'))
        nptest.assert_almost_equal(self.storm.peak_precip_intensity, self.known_peak_precip_intensity)

    def test_peak_inflow(self):
        assert_true(hasattr(self.storm, 'peak_inflow'))
        nptest.assert_almost_equal(self.storm.peak_inflow, self.known_peak_inflow)

    def test_peak_outflow(self):
        assert_true(hasattr(self.storm, 'peak_outflow'))
        nptest.assert_almost_equal(self.storm.peak_outflow, self.known_peak_outflow)

    def test_peak_precip_intensity_time(self):
        assert_true(hasattr(self.storm, 'peak_precip_intensity_time'))
        assert_timestamp_equal(self.storm.peak_precip_intensity_time, self.known_peak_precip_intensity_time)

    def test_peak_inflow_time(self):
        assert_true(hasattr(self.storm, 'peak_inflow_time'))
        assert_timestamp_equal(self.storm.peak_inflow_time, self.known_peak_inflow_time)

    def test_peak_outflow_time(self):
        assert_true(hasattr(self.storm, 'peak_outflow_time'))
        assert_timestamp_equal(self.storm.peak_outflow_time, self.known_peak_outflow_time)

    def test_centroid_precip(self):
        assert_true(hasattr(self.storm, 'centroid_precip_time'))
        assert_timestamp_equal(self.storm.centroid_precip_time, self.known_centroid_precip)

    def test_centroid_inflow(self):
        assert_true(hasattr(self.storm, 'centroid_inflow_time'))
        assert_timestamp_equal(self.storm.centroid_inflow_time, self.known_centroid_inflow)

    def test_centroid_outflow(self):
        assert_true(hasattr(self.storm, 'centroid_outflow_time'))
        assert_timestamp_equal(self.storm.centroid_outflow_time, self.known_centroid_outflow)

    def test_total_precip_depth(self):
        assert_true(hasattr(self.storm, 'total_precip_depth'))
        nptest.assert_almost_equal(self.storm.total_precip_depth, self.known_total_precip_depth)

    def test_total_inflow_volume(self):
        assert_true(hasattr(self.storm, 'total_inflow_volume'))
        nptest.assert_almost_equal(self.storm.total_inflow_volume, self.known_total_inflow_volume)

    def test_total_outflow_volume(self):
        assert_true(hasattr(self.storm, 'total_outflow_volume'))
        nptest.assert_almost_equal(self.storm.total_outflow_volume, self.known_total_outflow_volume)

    def test_peak_lag_hours(self):
        assert_true(hasattr(self.storm, 'peak_lag_hours'))
        nptest.assert_almost_equal(self.storm.peak_lag_hours, self.known_peak_lag_time)

    def test_centroid_lag_hours(self):
        assert_true(hasattr(self.storm, 'centroid_lag_hours'))
        nptest.assert_almost_equal(self.storm.centroid_lag_hours, self.known_centroid_lag_time)

    def test_summaryPlot(self):
        assert_true(hasattr(self.storm, 'summaryPlot'))
        output = self.makePath('test_basicstorm.png')
        self.storm.summaryPlot(filename=output)


