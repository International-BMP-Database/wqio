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

from wqio.core import hydro
from wqio import utils


class fakeStormSublcass(hydro.Storm):
    is_subclassed = True


class base_HydroRecordMixin(object):
    known_storm_stats_columns = [
        'Storm Number', 'Antecedent Days', 'Season', 'Start Date', 'End Date',
        'Duration Hours', 'Peak Precip Intensity', 'Total Precip Depth',
        'Total Inflow Volume', 'Peak Inflow', 'Total Outflow Volume',
        'Peak Outflow', 'Peak Lag Hours', 'Centroid Lag Hours'
    ]
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

    def test_storm_stat_columns(self):
        nt.assert_list_equal(
            self.known_storm_stats_columns,
            self.hr.storm_stats.columns.tolist()
        )


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
            'Antecedent Days': [np.nan, 0.708333],
            'Peak Precip Intensity': [1.200, 1.200],
            'Total Precip Depth': [2.76, 4.14],
        })

        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = hydro.HydroRecord(
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
            'Total Precip Depth',
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
        nt.assert_true(isinstance(storm, hydro.Storm))


    def test_getStormFromTimestame_big(self):
        sn, storm = self.hr.getStormFromTimestamp(self.known_storm1_start)
        nt.assert_equal(sn, 1)
        nt.assert_true(isinstance(storm, hydro.Storm))

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
        self.hr = hydro.HydroRecord(
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
        self.hr = hydro.HydroRecord(
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
        self.hr = hydro.HydroRecord(
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
            'Total Precip Depth': [2.76, 4.14],
        })

    def test_subclassed_storms(self):
        for sn in self.hr.all_storms:
            nt.assert_true(self.hr.all_storms[sn].is_subclassed)


class test_Storm(object):
    def setup(self):
        # path stuff

        self.storm_file = os.path.join(sys.prefix, 'wqio_data', 'testing', 'teststorm_simple.csv')
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = hydro.HydroRecord(self.orig_record,
                                     precipcol='rain',
                                     inflowcol='influent',
                                     outflowcol='effluent',
                                     outputfreqMinutes=5,
                                     intereventHours=2)
        self.storm = hydro.Storm(self.hr.data, 2,
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
            'Centroid Lag Hours',
            'Season'
        ]
        keys = list(self.storm.summary_dict.keys())
        nt.assert_list_equal(sorted(keys), sorted(known_keys))

    def test_is_small(self):
        nt.assert_true(self.storm.is_small(minprecip=5.0))
        nt.assert_false(self.storm.is_small(minprecip=1.0))


@nt.nottest
def setup_storm():
    plt.rcdefaults()
    storm_file = os.path.join(sys.prefix, 'wqio_data', 'testing', 'teststorm_simple.csv')
    orig_record = (
        pandas.read_csv(storm_file, index_col='date', parse_dates=True )
            .resample('5T')
            .fillna(0)
    )

    hr = hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol=None, outputfreqMinutes=5, minprecip=0,
        intereventHours=3
    )

    return hr.storms[2]

@image_comparison(baseline_images=['test_stormplot'], extensions=['png'])
def test_plot_storm_summary():
    storm = setup_storm()
    def doplot(storm):
        labels = {
            storm.inflowcol: 'Effluent (l/s)',
            storm.precipcol: 'Precip Depth (mm)'

        }
        fig, artists, labels = storm.summaryPlot(outflow=False, serieslabels=labels)
        return fig

    fig = doplot(storm)
