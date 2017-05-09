import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest
from wqio.tests import helpers

import numpy
import pandas
from matplotlib import pyplot

from wqio import hydro


BASELINE_IMAGES = '_baseline_images/hydro_tests'


class fakeStormSublcass(hydro.Storm):
    is_subclassed = True


def setup_storms(filename, baseflow=None):
    storm_file = helpers.test_data_path(filename)
    orig_record = pandas.read_csv(
        storm_file, index_col='date', parse_dates=True
    ).resample('5T').asfreq().fillna(0)

    baseflowcol = None
    if baseflow is not None:
        orig_record = orig_record.assign(baseflow=orig_record['influent'] < baseflow)
        baseflowcol = 'baseflow'

    parsed = hydro.parse_storm_events(orig_record, 3, 5,
                                      precipcol='rain',
                                      inflowcol='influent',
                                      baseflowcol=baseflowcol,
                                      debug=True)
    return parsed


@pytest.fixture
def storms_simple():
    return setup_storms('teststorm_simple.csv')


@pytest.fixture
def storms_firstobs():
    return setup_storms('teststorm_firstobs.csv')


@pytest.fixture
def storms_singular():
    return setup_storms('teststorm_singular.csv')


@pytest.fixture
def storms_with_baseflowcol():
    return setup_storms('teststorm_simple.csv', baseflow=2)


@pytest.mark.parametrize(('wetcol', 'expected'), [('A', 1), ('B', 0)])
def test_wet_first_row(wetcol, expected):
    df = pandas.DataFrame({
        'A': [True, True, False],
        'B': [False, False, True],
        'Z': [0, 0, 0],
    }).pipe(hydro._wet_first_row, wetcol, 'Z')

    assert df.iloc[0].loc['Z'] == expected


def test_wet_window_diff():
    is_wet = pandas.Series([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    result = hydro._wet_window_diff(is_wet, 3)
    expected = pandas.Series([numpy.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    pdtest.assert_series_equal(result, expected)


@pytest.mark.parametrize('parsed', [
    storms_simple(),
    storms_firstobs(),
    storms_singular(),
    storms_with_baseflowcol()
])
def test_number_of_storms(parsed):
    last_storm = parsed['storm'].max()
    assert last_storm == 5


@pytest.mark.parametrize(('parsed', 'sn', 'idx', 'known'), [
    (storms_simple(), 1, 0, '2013-05-18 13:40:00'),
    (storms_simple(), 1, -1, '2013-05-19 01:45:00'),
    (storms_simple(), 2, 0, '2013-05-19 06:10'),
    (storms_simple(), 2, -1, '2013-05-19 11:35'),
    (storms_with_baseflowcol(), 1, 0, '2013-05-18 14:05:00'),
    (storms_with_baseflowcol(), 1, -1, '2013-05-19 01:40:00'),
    (storms_with_baseflowcol(), 2, 0, '2013-05-19 06:35'),
    (storms_with_baseflowcol(), 2, -1, '2013-05-19 11:30'),
    (storms_singular(), 1, 0, '2013-05-18 13:40:00'),
    (storms_singular(), 1, -1, '2013-05-19 01:45:00'),
    (storms_singular(), 2, 0, '2013-05-19 07:00'),
    (storms_singular(), 2, -1, '2013-05-19 07:05'),
    (storms_firstobs(), 1, 0, '2013-05-18 12:05'),
    (storms_firstobs(), 1, -1, '2013-05-19 01:45:00'),
    (storms_firstobs(), 2, 0, '2013-05-19 06:10'),
    (storms_firstobs(), 2, -1, '2013-05-19 11:35'),
])
def test_storm_start_end(parsed, sn, idx, known):
    storm = parsed[parsed['storm'] == sn]
    assert storm.index[idx] == pandas.Timestamp(known)


class base_HydroRecordMixin(object):
    known_storm_stats_columns = [
        'Storm Number', 'Antecedent Days', 'Season', 'Start Date', 'End Date',
        'Duration Hours', 'Peak Precip Intensity', 'Total Precip Depth',
        'Total Inflow Volume', 'Peak Inflow', 'Total Outflow Volume',
        'Peak Outflow', 'Peak Lag Hours', 'Centroid Lag Hours'
    ]

    def teardown(self):
        pyplot.close('all')

    def test_attributes(self):
        assert hasattr(self.hr, '_raw_data')
        assert isinstance(self.hr._raw_data, pandas.DataFrame)

        assert hasattr(self.hr, 'data')
        assert isinstance(self.hr.data, pandas.DataFrame)
        assert isinstance(self.hr.data.index, pandas.DatetimeIndex)
        for col in self.known_std_columns:
            assert col in self.hr.data.columns.tolist()

        assert hasattr(self.hr, 'all_storms')
        assert isinstance(self.hr.all_storms, dict)

        assert hasattr(self.hr, 'storms')
        assert isinstance(self.hr.storms, dict)

        assert hasattr(self.hr, 'storm_stats')
        assert isinstance(self.hr.storm_stats, pandas.DataFrame)

    def test_check_type_and_columns(self):
        assert isinstance(self.hr.data, pandas.DataFrame)

    def test_check_nan_col(self):
        assert numpy.all(numpy.isnan(self.hr.data['outflow']))

    def test_number_storms(self):
        last_storm = self.hr.data['storm'].max()
        assert last_storm == self.known_number_of_storms

    def test_first_storm_start(self):
        storm1 = self.hr.data[self.hr.data['storm'] == 1]
        assert storm1.index[0] == self.known_storm1_start

    def test_first_storm_end(self):
        storm1 = self.hr.data[self.hr.data['storm'] == 1]
        assert storm1.index[-1] == self.known_storm1_end

    def test_second_storm_start(self):
        storm2 = self.hr.data[self.hr.data['storm'] == 2]
        assert storm2.index[0] == self.known_storm2_start

    def test_second_storm_end(self):
        storm2 = self.hr.data[self.hr.data['storm'] == 2]
        assert storm2.index[-1] == self.known_storm2_end

    def test_interevent_stuff(self):
        assert hasattr(self.hr, 'intereventPeriods')
        assert hasattr(self.hr, 'intereventHours')

        assert self.hr.intereventPeriods == self.known_ie_periods
        assert self.hr.intereventHours == self.known_ie_hours

    def test_storm_stat_columns(self):
        assert self.known_storm_stats_columns == self.hr.storm_stats.columns.tolist()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_HydroRecord_histogram():
    stormfile = helpers.test_data_path('teststorm_simple.csv')
    orig_record = pandas.read_csv(
        stormfile, index_col='date', parse_dates=True
    ).resample('5T').asfreq().fillna(0)
    hr = hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol=None, outputfreqMinutes=5, minprecip=1.5,
        intereventHours=3
    )

    fig = hr.histogram('Total Precip Depth', [4, 6, 8, 10])
    return fig


class Test_HydroRecord_Simple(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_number_of_bigstorms = 2
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = helpers.test_data_path('teststorm_simple.csv')

        self.known_std_columns = ['rain', 'influent', 'outflow', 'storm']
        self.known_stats_subset = pandas.DataFrame({
            'Storm Number': [1, 5],
            'Antecedent Days': [numpy.nan, 0.708333],
            'Peak Precip Intensity': [1.200, 1.200],
            'Total Precip Depth': [2.76, 4.14],
        })

        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').asfreq().fillna(0)
        self.hr = hydro.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5, minprecip=1.5,
            intereventHours=self.known_ie_hours
        )

        self.storm_date = pandas.Timestamp('2013-05-19 11:11')
        self.gap_date = pandas.Timestamp('2013-05-19 14:42')
        self.known_storm = 2

    def test_storm_attr_lengths(self):
        assert len(self.hr.storms) == self.known_number_of_bigstorms
        assert len(self.hr.all_storms) == self.known_number_of_storms

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
        assert sn == self.known_storm
        assert storm is None

    def test_getStormFromTimestamp_string(self):
        datestring = '{}'.format(self.storm_date)
        sn, storm = self.hr.getStormFromTimestamp(datestring)
        assert sn == self.known_storm
        assert storm is None

    def test_getStormFromTimestamp_datetime(self):
        pydt = self.storm_date.to_pydatetime()
        sn, storm = self.hr.getStormFromTimestamp(pydt)
        assert sn == self.known_storm
        assert storm is None

    def test_getStormFromTimestamp_small(self):
        sn, storm = self.hr.getStormFromTimestamp(self.storm_date, smallstorms=True)
        assert sn == self.known_storm
        assert isinstance(storm, hydro.Storm)

    def test_getStormFromTimestame_big(self):
        sn, storm = self.hr.getStormFromTimestamp(self.known_storm1_start)
        assert sn == 1
        assert isinstance(storm, hydro.Storm)

    def test_getStormFromTimestamp_lookback_6(self):
        sn, storm = self.hr.getStormFromTimestamp(self.gap_date, lookback_hours=6)
        assert sn == self.known_storm

    def test_getStormFromTimestamp_lookback_2(self):
        sn, storm = self.hr.getStormFromTimestamp(self.gap_date, lookback_hours=2)
        assert sn is None

    def test_getStormFromTimestamp_bad_date(self):
        with pytest.raises(ValueError):
            self.hr.getStormFromTimestamp('junk')

    def test_getStormFromTimestamp_bad_lookback(self):
        with pytest.raises(ValueError):
            sn = self.hr.getStormFromTimestamp(self.gap_date, lookback_hours=-3)


class Test_HydroRecord_Singular(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 07:00')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 07:05')
        self.storm_file = helpers.test_data_path('teststorm_singular.csv')
        self.known_std_columns = ['rain', 'influent', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').asfreq().fillna(0)
        self.hr = hydro.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5,
            intereventHours=self.known_ie_hours
        )


class Test_HydroRecord_FirstObservation(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 12:05')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = helpers.test_data_path('teststorm_firstobs.csv')
        self.known_std_columns = ['rain', 'influent', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').asfreq().fillna(0)
        self.hr = hydro.HydroRecord(
            self.orig_record, precipcol='rain', inflowcol='influent',
            outflowcol=None, outputfreqMinutes=5,
            intereventHours=self.known_ie_hours
        )


class TestHydroRecord_diffStormClass(base_HydroRecordMixin):
    def setup(self):
        self.known_ie_hours = 3
        self.known_ie_periods = 36
        self.known_number_of_storms = 5
        self.known_storm1_start = pandas.Timestamp('2013-05-18 13:40:00')
        self.known_storm1_end = pandas.Timestamp('2013-05-19 01:45:00')
        self.known_storm2_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_storm2_end = pandas.Timestamp('2013-05-19 11:35')
        self.storm_file = helpers.test_data_path('teststorm_simple.csv')

        self.known_std_columns = ['rain', 'influent', 'outflow', 'storm']
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').asfreq().fillna(0)
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
            assert self.hr.all_storms[sn].is_subclassed


class Test_Storm(object):
    def setup(self):
        # path stuff
        self.storm_file = helpers.test_data_path('teststorm_simple.csv')
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').asfreq().fillna(0)
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

        self.known_columns = ['rain', 'influent', 'effluent', 'baseflow', 'storm']
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
        pyplot.close('all')
        pass

    def test_columns(self):
        assert sorted(self.known_columns) == sorted(self.storm.data.columns.tolist())

    def test_index_type(self):
        assert isinstance(self.storm.data.index, pandas.DatetimeIndex)

    def test_start(self):
        assert hasattr(self.storm, 'start')
        assert self.storm.start == self.known_start

    def test_end(self):
        assert hasattr(self.storm, 'end')
        assert self.storm.end == self.known_end

    def test_season(self):
        assert hasattr(self.storm, 'season')
        assert self.storm.season == self.known_season

    def test_season_setter(self):
        self.storm.season = 'spring'
        assert self.storm.season == 'spring'

    def test_precip_start(self):
        assert hasattr(self.storm, 'precip_start')
        assert self.storm.precip_start == self.known_precip_start

    def test_precip_end(self):
        assert hasattr(self.storm, 'precip_end')
        assert self.storm.precip_end == self.known_precip_end

    def test_inflow_start(self):
        assert hasattr(self.storm, 'inflow_start')
        assert self.storm.inflow_start == self.known_inflow_start

    def test_inflow_end(self):
        assert hasattr(self.storm, 'inflow_end')
        assert self.storm.inflow_end == self.known_inflow_end

    def test_outflow_start(self):
        assert hasattr(self.storm, 'outflow_start')
        assert self.storm.outflow_start == self.known_outflow_start

    def test_outflow_end(self):
        assert hasattr(self.storm, 'outflow_end')
        assert self.storm.outflow_end == self.known_outflow_end

    def test_duration_hours(self):
        assert hasattr(self.storm, 'duration_hours')
        assert abs(self.storm.duration_hours - self.known_duration_hours) < 0.0001

    def test_antecedent_period_days(self):
        assert hasattr(self.storm, 'antecedent_period_days')
        assert abs(self.storm.antecedent_period_days - self.known_antecedent_period_days) < 0.0001

    def test_peak_precip_intensity(self):
        assert hasattr(self.storm, 'peak_precip_intensity')
        assert abs(self.storm.peak_precip_intensity - self.known_peak_precip_intensity) < 0.0001

    def test_peak_inflow(self):
        assert hasattr(self.storm, 'peak_inflow')
        assert abs(self.storm.peak_inflow - self.known_peak_inflow) < 0.0001

    def test_peak_outflow(self):
        assert hasattr(self.storm, 'peak_outflow')
        assert abs(self.storm.peak_outflow - self.known_peak_outflow) < 0.0001

    def test_peak_precip_intensity_time(self):
        assert hasattr(self.storm, 'peak_precip_intensity_time')
        assert self.storm.peak_precip_intensity_time == self.known_peak_precip_intensity_time

    def test_peak_inflow_time(self):
        assert hasattr(self.storm, 'peak_inflow_time')
        assert self.storm.peak_inflow_time == self.known_peak_inflow_time

    def test_peak_outflow_time(self):
        assert hasattr(self.storm, 'peak_outflow_time')
        assert self.storm.peak_outflow_time == self.known_peak_outflow_time

    def test_centroid_precip(self):
        assert hasattr(self.storm, 'centroid_precip_time')
        assert self.storm.centroid_precip_time.strftime('%x %X') == self.known_centroid_precip.strftime('%x %X')

    def test_centroid_inflow(self):
        assert hasattr(self.storm, 'centroid_inflow_time')
        assert self.storm.centroid_inflow_time.strftime('%x %X') == self.known_centroid_inflow.strftime('%x %X')

    def test_centroid_outflow(self):
        assert hasattr(self.storm, 'centroid_outflow_time')
        assert self.storm.centroid_outflow_time.strftime('%x %X') == self.known_centroid_outflow.strftime('%x %X')

    def test_total_precip_depth(self):
        assert hasattr(self.storm, 'total_precip_depth')
        assert abs(self.storm.total_precip_depth - self.known_total_precip_depth) < 0.0001

    def test_total_inflow_volume(self):
        assert hasattr(self.storm, 'total_inflow_volume')
        assert abs(self.storm.total_inflow_volume - self.known_total_inflow_volume) < 0.0001

    def test_total_outflow_volume(self):
        assert hasattr(self.storm, 'total_outflow_volume')
        assert abs(self.storm.total_outflow_volume - self.known_total_outflow_volume) < 0.0001

    def test_peak_lag_hours(self):
        assert hasattr(self.storm, 'peak_lag_hours')
        assert abs(self.storm.peak_lag_hours - self.known_peak_lag_time) < 0.0001

    def test_centroid_lag_hours(self):
        assert hasattr(self.storm, 'centroid_lag_hours')
        assert abs(self.storm.centroid_lag_hours - self.known_centroid_lag_time) < 0.0001

    def test_summary_dict(self):
        assert hasattr(self.storm, 'summary_dict')
        assert isinstance(self.storm.summary_dict, dict)
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
        assert sorted(keys) == sorted(known_keys)

    def test_is_small(self):
        assert self.storm.is_small(minprecip=5.0)
        assert not self.storm.is_small(minprecip=1.0)


def setup_storm():
    pyplot.rcdefaults()
    storm_file = helpers.test_data_path('teststorm_simple.csv')
    orig_record = (
        pandas.read_csv(storm_file, index_col='date', parse_dates=True)
              .resample('5T')
              .asfreq()
              .fillna(0)
    )

    hr = hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol=None, outputfreqMinutes=5, minprecip=0,
        intereventHours=3
    )

    return hr.storms[2]


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
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
    return fig


class Test_DrainageArea(object):
    def setup(self):
        self.total_area = 100.0
        self.imp_area = 75.0
        self.bmp_area = 10.0
        self.da = hydro.DrainageArea(self.total_area, self.imp_area, self.bmp_area)
        self.volume_conversion = 5
        self.known_storm_runoff = 82.5
        self.known_annual_runoff = 75.25
        self.storm_depth = 1.0
        self.storm_volume = self.storm_depth * (self.total_area + self.bmp_area)
        self.annualFactor = 0.9

    def test_total_area(self):
        assert hasattr(self.da, 'total_area')
        assert self.total_area == self.da.total_area

    def test_imp_area(self):
        assert hasattr(self.da, 'imp_area')
        assert self.imp_area == self.da.imp_area

    def test_bmp_area(self):
        assert hasattr(self.da, 'bmp_area')
        assert self.bmp_area == self.da.bmp_area

    def test_simple_method_noConversion(self):
        assert hasattr(self.da, 'simple_method')
        runoff = self.da.simple_method(1)
        assert abs(self.known_storm_runoff - runoff) < 0.001
        assert self.storm_volume > runoff

    def test_simple_method_Conversion(self):
        assert hasattr(self.da, 'simple_method')
        runoff = self.da.simple_method(1, volume_conversion=self.volume_conversion)
        assert abs(self.known_storm_runoff * self.volume_conversion - runoff) < 0.001
        assert self.storm_volume * self.volume_conversion > runoff

    def test_simple_method_annualFactor(self):
        assert hasattr(self.da, 'simple_method')
        runoff = self.da.simple_method(1, annualFactor=self.annualFactor)
        assert abs(self.known_annual_runoff - runoff) < 0.001
        assert self.storm_volume > runoff

    def teardown(self):
        pyplot.close('all')
