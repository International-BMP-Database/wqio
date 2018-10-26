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


def hr_simple():
    storm_file = helpers.test_data_path('teststorm_simple.csv')
    orig_record = (
        pandas.read_csv(storm_file, index_col='date', parse_dates=True)
              .resample('5T')
              .asfreq()
              .fillna(0)
    )

    return hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol='effluent', outputfreqMinutes=5, minprecip=1.5,
        intereventHours=3
    )


@pytest.fixture()
def hr_simple_fixture():
    return hr_simple()


def hr_singular():
    storm_file = helpers.test_data_path('teststorm_singular.csv')
    orig_record = (
        pandas.read_csv(storm_file, index_col='date', parse_dates=True)
              .resample('5T')
              .asfreq()
              .fillna(0)
    )
    return hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol=None, outputfreqMinutes=5,
        intereventHours=3
    )


def hr_first_obs():
    storm_file = helpers.test_data_path('teststorm_firstobs.csv')
    orig_record = (
        pandas.read_csv(storm_file, index_col='date', parse_dates=True)
              .resample('5T')
              .asfreq()
              .fillna(0)
    )
    return hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol=None, outputfreqMinutes=5,
        intereventHours=3
    )


def hr_diff_storm_class():
    storm_file = helpers.test_data_path('teststorm_simple.csv')
    orig_record = (
        pandas.read_csv(storm_file, index_col='date', parse_dates=True)
              .resample('5T')
              .asfreq()
              .fillna(0)
    )
    return hydro.HydroRecord(
        orig_record, precipcol='rain', inflowcol='influent',
        outflowcol=None, outputfreqMinutes=5, minprecip=1.5,
        stormclass=fakeStormSublcass,
        intereventHours=3
    )


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


def storms_simple():
    return setup_storms('teststorm_simple.csv')


def storms_firstobs():
    return setup_storms('teststorm_firstobs.csv')


def storms_singular():
    return setup_storms('teststorm_singular.csv')


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


@pytest.mark.parametrize(('stormfile', 'baseflow'), [
    ('teststorm_simple.csv', None),
    ('teststorm_firstobs.csv', None),
    ('teststorm_singular.csv', None),
    ('teststorm_simple.csv', 2)
])
def test_number_of_storms(stormfile, baseflow):
    parsed = setup_storms(stormfile, baseflow=baseflow)
    last_storm = parsed['storm'].max()
    assert last_storm == 5


@pytest.mark.parametrize(('stormfile', 'baseflow', 'sn', 'idx', 'known'), [
    ('teststorm_simple.csv', None, 1, 0, '2013-05-18 13:40:00'),
    ('teststorm_simple.csv', None, 1, -1, '2013-05-19 01:45:00'),
    ('teststorm_simple.csv', None, 2, 0, '2013-05-19 06:10'),
    ('teststorm_simple.csv', None, 2, -1, '2013-05-19 11:35'),
    ('teststorm_simple.csv', 2, 1, 0, '2013-05-18 14:05:00'),
    ('teststorm_simple.csv', 2, 1, -1, '2013-05-19 01:40:00'),
    ('teststorm_simple.csv', 2, 2, 0, '2013-05-19 06:35'),
    ('teststorm_simple.csv', 2, 2, -1, '2013-05-19 11:30'),
    ('teststorm_singular.csv', None, 1, 0, '2013-05-18 13:40:00'),
    ('teststorm_singular.csv', None, 1, -1, '2013-05-19 01:45:00'),
    ('teststorm_singular.csv', None, 2, 0, '2013-05-19 07:00'),
    ('teststorm_singular.csv', None, 2, -1, '2013-05-19 07:05'),
    ('teststorm_firstobs.csv', None, 1, 0, '2013-05-18 12:05'),
    ('teststorm_firstobs.csv', None, 1, -1, '2013-05-19 01:45:00'),
    ('teststorm_firstobs.csv', None, 2, 0, '2013-05-19 06:10'),
    ('teststorm_firstobs.csv', None, 2, -1, '2013-05-19 11:35'),
])
def test_storm_start_end(stormfile, baseflow, sn, idx, known):
    parsed = setup_storms(stormfile, baseflow=baseflow)
    storm = parsed[parsed['storm'] == sn]
    ts = pandas.Timestamp(known)
    assert storm.index[idx].strftime('%X %x') == ts.strftime('%X %x')


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
    return fig.fig


def test_HydroRecord_attr(hr_simple_fixture):
    expected_std_columns = ['baseflow', 'rain', 'influent', 'effluent', 'storm']
    assert isinstance(hr_simple_fixture._raw_data, pandas.DataFrame)

    assert isinstance(hr_simple_fixture.data, pandas.DataFrame)
    assert isinstance(hr_simple_fixture.data.index, pandas.DatetimeIndex)
    assert sorted(hr_simple_fixture.data.columns.tolist()) == sorted(expected_std_columns)

    assert isinstance(hr_simple_fixture.all_storms, dict)

    assert isinstance(hr_simple_fixture.storms, dict)

    assert isinstance(hr_simple_fixture.storm_stats, pandas.DataFrame)


@pytest.mark.parametrize(('hr', 'expected'), [
    (hr_simple(), 5),
    (hr_singular(), 5),
    (hr_first_obs(), 5),
    (hr_diff_storm_class(), 5),
])
def test_HydroRecord_number_storms(hr, expected):
    assert hr.data['storm'].max() == expected


@pytest.mark.parametrize(('hr', 'expected'), [
    (hr_simple(), pandas.Timestamp('2013-05-18 13:40:00')),
    (hr_singular(), pandas.Timestamp('2013-05-18 13:40:00')),
    (hr_first_obs(), pandas.Timestamp('2013-05-18 12:05')),
    (hr_diff_storm_class(), pandas.Timestamp('2013-05-18 13:40:00')),
])
def test_HydroRecord_first_storm_start(hr, expected):
    storm1 = hr.data[hr.data['storm'] == 1]
    assert storm1.index[0] == expected


@pytest.mark.parametrize(('hr', 'expected'), [
    (hr_simple(), pandas.Timestamp('2013-05-19 01:55:00')),
    (hr_singular(), pandas.Timestamp('2013-05-19 01:45:00')),
    (hr_first_obs(), pandas.Timestamp('2013-05-19 01:45:00')),
    (hr_diff_storm_class(), pandas.Timestamp('2013-05-19 01:45:00')),
])
def test_HydroRecord_first_storm_end(hr, expected):
    storm1 = hr.data[hr.data['storm'] == 1]
    assert storm1.index[-1] == expected


@pytest.mark.parametrize(('hr', 'expected'), [
    (hr_simple(), pandas.Timestamp('2013-05-19 06:10')),
    (hr_singular(), pandas.Timestamp('2013-05-19 07:00')),
    (hr_first_obs(), pandas.Timestamp('2013-05-19 06:10')),
    (hr_diff_storm_class(), pandas.Timestamp('2013-05-19 06:10')),
])
def test_HydroRecord_second_storm_start(hr, expected):
    storm2 = hr.data[hr.data['storm'] == 2]
    assert storm2.index[0] == expected


@pytest.mark.parametrize(('hr', 'expected'), [
    (hr_simple(), pandas.Timestamp('2013-05-19 11:55')),
    (hr_singular(), pandas.Timestamp('2013-05-19 07:05')),
    (hr_first_obs(), pandas.Timestamp('2013-05-19 11:35')),
    (hr_diff_storm_class(), pandas.Timestamp('2013-05-19 11:35')),
])
def test_HydroRecord_second_storm_end(hr, expected):
    storm2 = hr.data[hr.data['storm'] == 2]
    assert storm2.index[-1] == expected


def test_HydroRecord_interevent_stuff(hr_simple_fixture):
    assert hr_simple_fixture.intereventPeriods == 36
    assert hr_simple_fixture.intereventHours == 3


def test_HydroRecord_storm_stat_columns(hr_simple_fixture):
    expected = [
        'Storm Number', 'Antecedent Days', 'Season', 'Start Date', 'End Date',
        'Duration Hours', 'Peak Precip Intensity', 'Total Precip Depth',
        'Total Inflow Volume', 'Peak Inflow', 'Total Outflow Volume',
        'Peak Outflow', 'Peak Lag Hours', 'Centroid Lag Hours'
    ]
    assert hr_simple_fixture.storm_stats.columns.tolist() == expected


def test_HydroRecord_storm_attr_lengths(hr_simple_fixture):
    assert len(hr_simple_fixture.storms) == 2
    assert len(hr_simple_fixture.all_storms) == 5


def test_HydroRecord_storm_stats(hr_simple_fixture):
    cols = [
        'Storm Number',
        'Antecedent Days',
        'Peak Precip Intensity',
        'Total Precip Depth',
    ]

    expected = pandas.DataFrame({
        'Storm Number': [1, 5],
        'Antecedent Days': [numpy.nan, 0.701389],
        'Peak Precip Intensity': [1.200, 1.200],
        'Total Precip Depth': [2.76, 4.14],
    })
    pdtest.assert_frame_equal(hr_simple_fixture.storm_stats[cols], expected[cols])


@pytest.mark.parametrize(('value', 'error'), [
    (pandas.Timestamp('2013-05-19 11:11'), None),
    (str(pandas.Timestamp('2013-05-19 11:11')), None),
    (pandas.Timestamp('2013-05-19 11:11').to_pydatetime(), None),
    ('junk', ValueError),
])
@pytest.mark.parametrize('smallstorms', [True, False])
def test_getStormFromTimestamp_(hr_simple_fixture, value, smallstorms, error):
    with helpers.raises(error):
        sn, storm = hr_simple_fixture.getStormFromTimestamp(value, smallstorms=smallstorms)
        assert sn == 2
        if not smallstorms:
            assert storm is None
        else:
            assert isinstance(storm, hydro.Storm)


def test_getStormFromTimestame_big(hr_simple_fixture):
    ts = pandas.Timestamp('2013-05-18 13:40:00')
    sn, storm = hr_simple_fixture.getStormFromTimestamp(ts)
    assert sn == 1
    assert isinstance(storm, hydro.Storm)


@pytest.mark.parametrize(('lbh', 'expected', 'error'), [
    (6, 2, None), (2, None, None), (-3, None, ValueError)
])
def test_getStormFromTimestamp_lookback(hr_simple_fixture, lbh, expected, error):
    with helpers.raises(error):
        ts = pandas.Timestamp('2013-05-19 14:42')
        sn, storm = hr_simple_fixture.getStormFromTimestamp(ts, lookback_hours=lbh)
        if expected:
            assert sn == expected
        else:
            assert sn is None


def test_HydroRecord_subclassed_storms():
    hr = hr_diff_storm_class()
    assert all([
        hr.all_storms[sn].is_subclassed
        for sn in hr.all_storms
    ])


@pytest.fixture
def basic_storm():
    hr = hr_simple()
    storm = hydro.Storm(hr.data, 2, precipcol=hr.precipcol,
                        inflowcol=hr.inflowcol, outflowcol=hr.outflowcol,
                        freqMinutes=hr.outputfreq.n)
    return storm


def test_Storm_columns(basic_storm):
    expected = ['rain', 'influent', 'effluent', 'baseflow', 'storm']
    assert sorted(basic_storm.data.columns.tolist()) == sorted(expected)


def test_Storm_index_type(basic_storm):
    assert isinstance(basic_storm.data.index, pandas.DatetimeIndex)


def test_Storm_start(basic_storm):
    assert hasattr(basic_storm, 'start')
    ts = pandas.Timestamp('2013-05-19 06:10')
    assert basic_storm.start.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_end(basic_storm):
    assert hasattr(basic_storm, 'end')
    ts = pandas.Timestamp('2013-05-19 11:55')
    assert basic_storm.end.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_season(basic_storm):
    assert hasattr(basic_storm, 'season')
    assert basic_storm.season == 'spring'


def test_Storm_season_setter(basic_storm):
    basic_storm.season = 'fall'
    assert basic_storm.season == 'fall'


def test_Storm_precip_start(basic_storm):
    assert hasattr(basic_storm, 'precip_start')
    ts = pandas.Timestamp('2013-05-19 06:10')
    assert basic_storm.precip_start.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_precip_end(basic_storm):
    assert hasattr(basic_storm, 'precip_end')
    ts = pandas.Timestamp('2013-05-19 09:10')
    assert basic_storm.precip_end.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_inflow_start(basic_storm):
    assert hasattr(basic_storm, 'inflow_start')
    ts = pandas.Timestamp('2013-05-19 06:25')
    assert basic_storm.inflow_start.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_inflow_end(basic_storm):
    assert hasattr(basic_storm, 'inflow_end')
    ts = pandas.Timestamp('2013-05-19 11:30')
    assert basic_storm.inflow_end.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_outflow_start(basic_storm):
    assert hasattr(basic_storm, 'outflow_start')
    ts = pandas.Timestamp('2013-05-19 06:50')
    assert basic_storm.outflow_start.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_outflow_end(basic_storm):
    assert hasattr(basic_storm, 'outflow_end')
    ts = pandas.Timestamp('2013-05-19 11:50')
    assert basic_storm.outflow_end.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_duration_hours(basic_storm):
    assert hasattr(basic_storm, 'duration_hours')
    assert abs(basic_storm.duration_hours - 5.75) < 0.0001


def test_Storm_antecedent_period_days(basic_storm):
    assert hasattr(basic_storm, 'antecedent_period_days')
    assert abs(basic_storm.antecedent_period_days - 0.17708333) < 0.0001


def test_Storm_peak_precip_intensity(basic_storm):
    assert hasattr(basic_storm, 'peak_precip_intensity')
    assert abs(basic_storm.peak_precip_intensity - 1.32) < 0.0001


def test_Storm_peak_inflow(basic_storm):
    assert hasattr(basic_storm, 'peak_inflow')
    assert abs(basic_storm.peak_inflow - 123.0) < 0.0001


def test_Storm_peak_outflow(basic_storm):
    assert hasattr(basic_storm, 'peak_outflow')
    assert abs(basic_storm.peak_outflow - 41.0) < 0.0001


def test_Storm_peak_precip_intensity_time(basic_storm):
    assert hasattr(basic_storm, 'peak_precip_intensity_time')
    ts = pandas.Timestamp('2013-05-19 08:00')
    assert basic_storm.peak_precip_intensity_time.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_peak_inflow_time(basic_storm):
    assert hasattr(basic_storm, 'peak_inflow_time')
    ts = pandas.Timestamp('2013-05-19 08:35')
    assert basic_storm.peak_inflow_time.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_peak_outflow_time(basic_storm):
    assert hasattr(basic_storm, 'peak_outflow_time')
    ts = pandas.Timestamp('2013-05-19 08:45')
    assert basic_storm.peak_outflow_time.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_centroid_precip(basic_storm):
    assert hasattr(basic_storm, 'centroid_precip_time')
    ts = pandas.Timestamp('2013-05-19 07:41:58.705023')
    assert basic_storm.centroid_precip_time.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_centroid_inflow(basic_storm):
    assert hasattr(basic_storm, 'centroid_inflow_time')
    ts = pandas.Timestamp('2013-05-19 08:44:31.252553')
    assert basic_storm.centroid_inflow_time.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_centroid_outflow(basic_storm):
    assert hasattr(basic_storm, 'centroid_outflow_time')
    ts = pandas.Timestamp('2013-05-19 08:59:51.675231')
    assert basic_storm.centroid_outflow_time.strftime('%X %x') == ts.strftime('%X %x')


def test_Storm_total_precip_depth(basic_storm):
    assert hasattr(basic_storm, 'total_precip_depth')
    assert abs(basic_storm.total_precip_depth - 1.39) < 0.0001


def test_Storm_total_inflow_volume(basic_storm):
    assert hasattr(basic_storm, 'total_inflow_volume')
    assert abs(basic_storm.total_inflow_volume - 876600) < 0.0001


def test_Storm_total_outflow_volume(basic_storm):
    assert hasattr(basic_storm, 'total_outflow_volume')
    assert abs(basic_storm.total_outflow_volume - 291900) < 0.0001


def test_Storm_peak_lag_hours(basic_storm):
    assert hasattr(basic_storm, 'peak_lag_hours')
    assert abs(basic_storm.peak_lag_hours - 0.1666666667) < 0.0001


def test_Storm_centroid_lag_hours(basic_storm):
    assert hasattr(basic_storm, 'centroid_lag_hours')
    assert abs(basic_storm.centroid_lag_hours - 0.255672966) < 0.0001


def test_Storm_summary_dict(basic_storm):
    assert hasattr(basic_storm, 'summary_dict')
    assert isinstance(basic_storm.summary_dict, dict)
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
    keys = list(basic_storm.summary_dict.keys())
    assert sorted(keys) == sorted(known_keys)


def test_Storm_is_small(basic_storm):
    assert basic_storm.is_small(minprecip=5.0)
    assert not basic_storm.is_small(minprecip=1.0)


@pytest.fixture
def single_storm():
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
def test_plot_storm_summary(single_storm):
    labels = {
        single_storm.inflowcol: 'Effluent (l/s)',
        single_storm.precipcol: 'Precip Depth (mm)'
    }
    fig, artists, labels = single_storm.summaryPlot(outflow=False, serieslabels=labels)
    return fig


@pytest.fixture
def drainage_area():
    return hydro.DrainageArea(100, 75, 10)


def test_da_total_area(drainage_area):
    assert drainage_area.total_area == 100


def test_da_imp_area(drainage_area):
    assert drainage_area.imp_area == 75


def test_da_bmp_area(drainage_area):
    assert drainage_area.bmp_area == 10


@pytest.mark.parametrize(('conversion', 'factor', 'expected'), [
    (1.0, 1.0, 82.50),
    (1.0, 0.9, 75.25),
    (2.0, 1.0, 165.00),
    (2.0, 0.9, 150.50)
])
def test_simple_method(drainage_area, conversion, factor, expected):
    depth = 1
    result = drainage_area.simple_method(depth, annual_factor=factor,
                                         volume_conversion=conversion)
    area = (drainage_area.total_area + drainage_area.bmp_area)
    storm_volume = depth * conversion * factor * area
    assert abs(expected - result) < 0.001
    assert storm_volume > result
