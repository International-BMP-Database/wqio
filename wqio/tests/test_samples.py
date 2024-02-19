import pandas
import pytest
from matplotlib import pyplot

from wqio import samples
from wqio.tests import helpers

BASELINE_IMAGES = "_baseline_images/samples_tests"
TOLERANCE = helpers.get_img_tolerance()


@pytest.fixture
def sample_data():
    datafile = helpers.test_data_path("test_wqsample_data.csv")
    rawdata = pandas.read_csv(datafile, index_col=[0, 1, 2, 3, 4, 5, 6, 11, 12])
    return rawdata


@pytest.fixture
def xlims():
    xlims = (
        pandas.Timestamp("2013-02-24 16:45").to_pydatetime(),
        pandas.Timestamp("2013-02-25 03:30").to_pydatetime(),
    )
    return xlims


@pytest.fixture
def grab_sample(sample_data):
    starttime = "2013-02-24 16:54"
    endtime = "2013-02-24 16:59"
    freq = None

    wqs = samples.GrabSample(sample_data, starttime, endtime=endtime, samplefreq=freq, storm=None)
    wqs.marker = "D"
    wqs.markersize = 8

    return wqs


@pytest.fixture
def composite_sample(sample_data):
    starttime = "2013-02-24 16:59"
    endtime = "2013-02-25 02:59"
    freq = pandas.offsets.Minute(20)

    wqs = samples.CompositeSample(
        sample_data, starttime, endtime=endtime, samplefreq=freq, storm=None
    )
    wqs.marker = "D"
    wqs.markersize = 8

    return wqs


@pytest.mark.parametrize(
    ("param", "unit", "comma", "expected"),
    [
        ("Copper", "ug/L", False, "Copper (ug/L)"),
        ("Copper", "ug/L", True, "Copper, ug/L"),
        ("Nitrate & Nitrite", "mg/L", False, "Nitrate & Nitrite (mg/L)"),
        ("Nitrate & Nitrite", "mg/L", True, "Nitrate & Nitrite, mg/L"),
    ],
)
def test_parameter_unit(param, unit, comma, expected):
    parameter = samples.Parameter(name=param, units=unit)
    assert parameter.paramunit(usecomma=comma) == expected


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_grabsample_noStorm_notFocus(grab_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    grab_sample.plot_ts(ax, isFocus=False)
    return fig


@pytest.mark.xfail
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_grabsample_noStorm_notFocus_asRug(grab_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    grab_sample.plot_ts(ax, isFocus=False, asrug=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_grabsample_noStorm_asFocus(grab_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    grab_sample.plot_ts(ax, isFocus=True)
    return fig


@pytest.mark.xfail
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_grabsample_noStorm_Focus_asRug(grab_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    grab_sample.plot_ts(ax, isFocus=True, asrug=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_compsample_noStorm_notFocus(composite_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    composite_sample.plot_ts(ax, isFocus=False)
    return fig


@pytest.mark.xfail
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_compsample_noStorm_notFocus_asRug(composite_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    composite_sample.plot_ts(ax, isFocus=False, asrug=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_compsample_noStorm_Focus(composite_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    composite_sample.plot_ts(ax, isFocus=True)
    return fig


@pytest.mark.xfail
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_plot_compsample_noStorm_Focus_asRug(composite_sample, xlims):
    fig, ax = pyplot.subplots()
    ax.set_xlim(xlims)
    composite_sample.plot_ts(ax, isFocus=True, asrug=True)
    return fig
