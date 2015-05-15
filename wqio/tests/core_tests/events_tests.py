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
matplotlib.rcParams['text.usetex'] = usetex
import matplotlib.pyplot as plt


from wqio.core import events
from wqio import utils


class fakeStormSublcass(events.Storm):
    is_subclassed = True


class base_wqsampleMixin(object):
    @nt.nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nt.nottest
    def basic_setup(self):
        self.prefix = setup_prefix('core.events')
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

    def test_plot_ts_isFocus(self):
        self.wqs.plot_ts(self.ax, isFocus=True)
        self.fig.savefig(self.makePath('{0}_{1}_wqsample_isFocus.png'.format(self.wqs.label, self.test_name)))

    def test_plot_ts_isNotFocus(self):
        self.wqs.plot_ts(self.ax, isFocus=False)
        self.fig.savefig(self.makePath('{0}_{1}_wqsample_isNotFocus.png'.format(self.wqs.label, self.test_name)))


class base_wqsample_NoStorm(base_wqsampleMixin):
    def test_storm(self):
        nt.assert_true(hasattr(self.wqs, 'storm'))
        nt.assert_true(self.wqs.storm is None)


class base_wqsample_WithStorm(base_wqsampleMixin):
    def test_storm(self):
        nt.assert_true(hasattr(self.wqs, 'storm'))
        nt.assert_true(isinstance(self.wqs.storm, events.Storm))


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
        self.wqs = events.GrabSample(self.rawdata, self.known_starttime,
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
        self.wqs = events.CompositeSample(self.rawdata, self.known_starttime,
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
        self.wqs = events.CompositeSample(self.rawdata, self.known_starttime,
                                          endtime=self.known_endtime,
                                          samplefreq=self.known_samplefreq,
                                          storm=None)

    def teardown(self):
        self.basic_teardown()


class base_HydroRecordMixin(object):
    def teardown(self):
        plt.close('all')

    def test_attributes(self):
        nt.assert_true(hasattr(self.hr, '_raw_data'))
        nt.assert_true(isinstance(self.hr._raw_data, pandas.DataFrame))


        nt.assert_true(hasattr(self.hr, 'data'))
        nt.assert_true(isinstance(self.hr.data, pandas.DataFrame))
        nt.assert_true(isinstance(self.hr.data.index, pandas.DatetimeIndex))
        for col in self.known_std_columns:
            nt.assert_true(col in self.hr.data.columns.tolist())

        nt.assert_true(hasattr(self.hr, 'all_storms'))
        nt.assert_true(isinstance(self.hr.all_storms, dict))

        nt.assert_true(hasattr(self.hr, 'storms'))
        nt.assert_true(isinstance(self.hr.storms, dict))

        nt.assert_true(hasattr(self.hr, 'storm_stats'))
        nt.assert_true(isinstance(self.hr.storm_stats, pandas.DataFrame))

    def test_check_type_and_columns(self):
        nt.assert_true(isinstance(self.hr.data, pandas.DataFrame))

    def test_check_nan_col(self):
        nt.assert_true(np.all(np.isnan(self.hr.data['outflow'])))

    def test_number_storms(self):
        last_storm = self.hr.data['storm'].max()
        nt.assert_equal(last_storm, self.known_number_of_storms)

    def test_first_storm_start(self):
        storm1 = self.hr.data[self.hr.data['storm'] == 1]
        assert_timestamp_equal(storm1.index[0], self.known_storm1_start)

    def test_first_storm_end(self):
        storm1 = self.hr.data[self.hr.data['storm'] == 1]
        assert_timestamp_equal(storm1.index[-1], self.known_storm1_end)

    def test_second_storm_start(self):
        storm2 = self.hr.data[self.hr.data['storm'] == 2]
        assert_timestamp_equal(storm2.index[0], self.known_storm2_start)

    def test_second_storm_end(self):
        storm2 = self.hr.data[self.hr.data['storm'] == 2]
        assert_timestamp_equal(storm2.index[-1], self.known_storm2_end)

    def test_interevent_stuff(self):
        nt.assert_true(hasattr(self.hr, 'intereventPeriods'))
        nt.assert_true(hasattr(self.hr, 'intereventHours'))

        nt.assert_equal(self.hr.intereventPeriods, self.known_ie_periods)
        nt.assert_equal(self.hr.intereventHours, self.known_ie_hours)


class test_HydroRecord_Simple(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_number_of_bigstorms = 2
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = os.path.join(
            sys.prefix, 'wqio_data', 'testing', 'teststorm_simple.csv'
        )

        self.known_std_columns = ['rain', 'influent', 'effluent', 'outflow', 'storm']
        self.known_stats_subset = pandas.DataFrame({
            'Storm Number': [1, 5],
            'Antecedent Days': [-2.409722, 0.708333],
            'Peak Precip Intensity': [1.200, 1.200],
            'Total Precip Depth': [2.76, 4.14]
        })

        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = events.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5, minprecip=1.5,
            intereventHours=self.known_ie_hours
        )

        self.storm_date = pandas.Timestamp('2013-05-19 11:11')
        self.gap_date = pandas.Timestamp('2013-05-19 14:42')
        self.known_storm = 2

    def test_storm_attr_lengths(self):
        nt.assert_equal(len(self.hr.storms), self.known_number_of_bigstorms)
        nt.assert_equal(len(self.hr.all_storms), self.known_number_of_storms)

    def test_storm_stats(self):

        cols = [
            'Storm Number',
            'Antecedent Days',
            'Peak Precip Intensity',
            'Total Precip Depth'
        ]

        pdtest.assert_frame_equal(self.hr.storm_stats[cols], self.known_stats_subset[cols])

    def test_getStormFromTimestamp_basic(self):
        sn, storm = self.hr.getStormFromTimestamp(self.storm_date)
        nt.assert_equal(sn, self.known_storm)
        nt.assert_true(storm is None)

    def test_getStormFromTimestamp_string(self):
        datestring = '{}'.format(self.storm_date)
        sn, storm = self.hr.getStormFromTimestamp(datestring)
        nt.assert_equal(sn, self.known_storm)
        nt.assert_true(storm is None)

    def test_getStormFromTimestamp_datetime(self):
        pydt = self.storm_date.to_pydatetime()
        sn, storm = self.hr.getStormFromTimestamp(pydt)
        nt.assert_equal(sn, self.known_storm)
        nt.assert_true(storm is None)

    def test_getStormFromTimestamp_small(self):
        sn, storm = self.hr.getStormFromTimestamp(self.storm_date, smallstorms=True)
        nt.assert_equal(sn, self.known_storm)
        nt.assert_true(isinstance(storm, events.Storm))


    def test_getStormFromTimestame_big(self):
        sn, storm = self.hr.getStormFromTimestamp(self.known_storm1_start)
        nt.assert_equal(sn, 1)
        nt.assert_true(isinstance(storm, events.Storm))

    def test_getStormFromTimestamp_lookback_6(self):
        sn, storm = self.hr.getStormFromTimestamp(self.gap_date, lookback_hours=6)
        nt.assert_equal(sn, self.known_storm)

    def test_getStormFromTimestamp_lookback_2(self):
        sn, storm = self.hr.getStormFromTimestamp(self.gap_date, lookback_hours=2)
        nt.assert_equal(sn, None)

    @nt.raises(ValueError)
    def test_getStormFromTimestamp_bad_date(self):
        self.hr.getStormFromTimestamp('junk')

    @nt.raises(ValueError)
    def test_getStormFromTimestamp_bad_lookback(self):
        sn = self.hr.getStormFromTimestamp(self.gap_date, lookback_hours=-3)


class test_HydroRecord_Singular(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 07:00')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 07:05')
        self.storm_file = os.path.join(
            sys.prefix, 'wqio_data', 'testing', 'teststorm_singular.csv'
        )
        self.known_std_columns = ['rain', 'influent', 'effluent', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = events.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5,
            intereventHours=self.known_ie_hours
        )


class test_HydroRecord_FirstObservation(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 12:05')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = os.path.join(
            sys.prefix, 'wqio_data', 'testing', 'teststorm_firstobs.csv'
        )
        self.known_std_columns = ['rain', 'influent', 'effluent', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = events.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5,
            intereventHours=self.known_ie_hours
        )


class testHydroRecord_diffStormClass(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = os.path.join(
            sys.prefix, 'wqio_data', 'testing', 'teststorm_simple.csv'
        )

        self.known_std_columns = ['rain', 'influent', 'effluent', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = events.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5, minprecip=1.5,
            stormclass=fakeStormSublcass,
            intereventHours=self.known_ie_hours
        )

        self.storm_date = pandas.Timestamp('2013-05-19 11:11')
        self.gap_date = pandas.Timestamp('2013-05-19 14:42')
        self.known_storm = 2
        self.known_stats_subset = pandas.DataFrame({
            'Storm Number': [1, 5],
            'Antecedent Days': [-2.409722, 0.708333],
            'Peak Precip Intensity': [1.200, 1.200],
            'Total Precip Depth': [2.76, 4.14]
        })

    def test_subclassed_storms(self):
        for sn in self.hr.all_storms:
            nt.assert_true(self.hr.all_storms[sn].is_subclassed)


class test_Storm(object):
    def setup(self):
        # path stuff
        self.prefix = setup_prefix('core.events')


        self.storm_file = os.path.join(sys.prefix, 'wqio_data', 'testing', 'teststorm_simple.csv')
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = events.HydroRecord(self.orig_record,
                                     precipcol='rain',
                                     inflowcol='influent',
                                     outflowcol='effluent',
                                     outputfreqMinutes=5,
                                     intereventHours=2)
        self.storm = events.Storm(self.hr.data, 2,
                                  precipcol=self.hr.precipcol,
                                  inflowcol=self.hr.inflowcol,
                                  outflowcol=self.hr.outflowcol,
                                  freqMinutes=self.hr.outputfreq.n)

        self.known_columns = ['rain', 'influent', 'effluent', 'storm']
        self.known_index_type = pandas.DatetimeIndex
        self.known_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_end = pandas.Timestamp('2013-05-19 11:55')
        self.known_season = 'spring'
        self.known_precip_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_precip_end = pandas.Timestamp('2013-05-19 09:10')
        self.known_inflow_start = pandas.Timestamp('2013-05-19 06:25')
        self.known_inflow_end = pandas.Timestamp('2013-05-19 11:30')
        self.known_outflow_start = pandas.Timestamp('2013-05-19 06:50')
        self.known_outflow_end = pandas.Timestamp('2013-05-19 11:50')
        self.known_duration_hours = 5.75
        self.known_antecedent_period_days = 0.17708333
        self.known_peak_precip_intensity = 1.32
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
        plt.close('all')
        pass

    @nt.nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    def test_columns(self):
        nt.assert_list_equal(self.known_columns, self.storm.data.columns.tolist())

    def test_index_type(self):
        nt.assert_true(isinstance(self.storm.data.index, pandas.DatetimeIndex))

    def test_start(self):
        nt.assert_true(hasattr(self.storm, 'start'))
        assert_timestamp_equal(self.storm.start, self.known_start)

    def test_end(self):
        nt.assert_true(hasattr(self.storm, 'end'))
        assert_timestamp_equal(self.storm.end, self.known_end)

    def test_season(self):
        nt.assert_true(hasattr(self.storm, 'season'))
        nt.assert_equal(self.storm.season, self.known_season)

    def test_season_setter(self):
        self.storm.season = 'spring'
        nt.assert_equal(self.storm.season, 'spring')

    def test_precip_start(self):
        nt.assert_true(hasattr(self.storm, 'precip_start'))
        assert_timestamp_equal(self.storm.precip_start, self.known_precip_start)

    def test_precip_end(self):
        nt.assert_true(hasattr(self.storm, 'precip_end'))
        assert_timestamp_equal(self.storm.precip_end, self.known_precip_end)

    def test_inflow_start(self):
        nt.assert_true(hasattr(self.storm, 'inflow_start'))
        assert_timestamp_equal(self.storm.inflow_start, self.known_inflow_start)

    def test_inflow_end(self):
        nt.assert_true(hasattr(self.storm, 'inflow_end'))
        assert_timestamp_equal(self.storm.inflow_end, self.known_inflow_end)

    def test_outflow_start(self):
        nt.assert_true(hasattr(self.storm, 'outflow_start'))
        assert_timestamp_equal(self.storm.outflow_start, self.known_outflow_start)

    def test_outflow_end(self):
        nt.assert_true(hasattr(self.storm, 'outflow_end'))
        assert_timestamp_equal(self.storm.outflow_end, self.known_outflow_end)

    def test_duration_hours(self):
        nt.assert_true(hasattr(self.storm, 'duration_hours'))
        nptest.assert_almost_equal(self.storm.duration_hours, self.known_duration_hours)

    def test_antecedent_period_days(self):
        nt.assert_true(hasattr(self.storm, 'antecedent_period_days'))
        nptest.assert_almost_equal(self.storm.antecedent_period_days, self.known_antecedent_period_days)

    def test_peak_precip_intensity(self):
        nt.assert_true(hasattr(self.storm, 'peak_precip_intensity'))
        nptest.assert_almost_equal(self.storm.peak_precip_intensity, self.known_peak_precip_intensity)

    def test_peak_inflow(self):
        nt.assert_true(hasattr(self.storm, 'peak_inflow'))
        nptest.assert_almost_equal(self.storm.peak_inflow, self.known_peak_inflow)

    def test_peak_outflow(self):
        nt.assert_true(hasattr(self.storm, 'peak_outflow'))
        nptest.assert_almost_equal(self.storm.peak_outflow, self.known_peak_outflow)

    def test_peak_precip_intensity_time(self):
        nt.assert_true(hasattr(self.storm, 'peak_precip_intensity_time'))
        assert_timestamp_equal(self.storm.peak_precip_intensity_time, self.known_peak_precip_intensity_time)

    def test_peak_inflow_time(self):
        nt.assert_true(hasattr(self.storm, 'peak_inflow_time'))
        assert_timestamp_equal(self.storm.peak_inflow_time, self.known_peak_inflow_time)

    def test_peak_outflow_time(self):
        nt.assert_true(hasattr(self.storm, 'peak_outflow_time'))
        assert_timestamp_equal(self.storm.peak_outflow_time, self.known_peak_outflow_time)

    def test_centroid_precip(self):
        nt.assert_true(hasattr(self.storm, 'centroid_precip_time'))
        assert_timestamp_equal(self.storm.centroid_precip_time, self.known_centroid_precip)

    def test_centroid_inflow(self):
        nt.assert_true(hasattr(self.storm, 'centroid_inflow_time'))
        assert_timestamp_equal(self.storm.centroid_inflow_time, self.known_centroid_inflow)

    def test_centroid_outflow(self):
        nt.assert_true(hasattr(self.storm, 'centroid_outflow_time'))
        assert_timestamp_equal(self.storm.centroid_outflow_time, self.known_centroid_outflow)

    def test_total_precip_depth(self):
        nt.assert_true(hasattr(self.storm, 'total_precip_depth'))
        nptest.assert_almost_equal(self.storm.total_precip_depth, self.known_total_precip_depth)

    def test_total_inflow_volume(self):
        nt.assert_true(hasattr(self.storm, 'total_inflow_volume'))
        nptest.assert_almost_equal(self.storm.total_inflow_volume, self.known_total_inflow_volume)

    def test_total_outflow_volume(self):
        nt.assert_true(hasattr(self.storm, 'total_outflow_volume'))
        nptest.assert_almost_equal(self.storm.total_outflow_volume, self.known_total_outflow_volume)

    def test_peak_lag_hours(self):
        nt.assert_true(hasattr(self.storm, 'peak_lag_hours'))
        nptest.assert_almost_equal(self.storm.peak_lag_hours, self.known_peak_lag_time)

    def test_centroid_lag_hours(self):
        nt.assert_true(hasattr(self.storm, 'centroid_lag_hours'))
        nptest.assert_almost_equal(self.storm.centroid_lag_hours, self.known_centroid_lag_time)

    def test_summary_dict(self):
        nt.assert_true(hasattr(self.storm, 'summary_dict'))
        nt.assert_true(isinstance(self.storm.summary_dict, dict))
        known_keys = [
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
            'Peak Lag Hours',
            'Season'
        ]
        keys = list(self.storm.summary_dict.keys())
        nt.assert_list_equal(sorted(keys), sorted(known_keys))

    def test_summaryPlot(self):
        nt.assert_true(hasattr(self.storm, 'summaryPlot'))
        output = self.makePath('test_basicstorm.png')
        self.storm.summaryPlot(filename=output)

    def test_is_small(self):
        nt.assert_true(self.storm.is_small(minprecip=5.0))
        nt.assert_false(self.storm.is_small(minprecip=1.0))


class _base_getSeason(object):
    def setup(self):
        self.winter = self.makeDate('1998-12-25')
        self.spring = self.makeDate('2015-03-22')
        self.summer = self.makeDate('1965-07-04')
        self.autumn = self.makeDate('1982-11-24')

    def test_winter(self):
        nt.assert_equal(events.getSeason(self.winter), 'winter')

    def test_spring(self):
        nt.assert_equal(events.getSeason(self.spring), 'spring')

    def test_summer(self):
        nt.assert_equal(events.getSeason(self.summer), 'summer')

    def test_autumn(self):
        nt.assert_equal(events.getSeason(self.autumn), 'autumn')


class test_getSeason_Datetime(_base_getSeason):
    @nt.nottest
    def makeDate(self, date_string):
        return datetime.datetime.strptime(date_string, '%Y-%m-%d')


class test_getSeason_Timestamp(_base_getSeason):
    @nt.nottest
    def makeDate(self, date_string):
        return pandas.Timestamp(date_string)
