import numpy
import numpy.testing as nptest
import pandas
import pytest
import seaborn
from matplotlib import pyplot

from wqio import utils, viz
from wqio.tests import helpers

BASELINE_IMAGES = "_baseline_images/viz_tests"
TOLERANCE = helpers.get_img_tolerance()


@pytest.fixture
def plot_data():
    data = numpy.array(
        [
            3.113,
            3.606,
            4.046,
            4.046,
            4.710,
            6.140,
            6.978,
            2.000,
            4.200,
            4.620,
            5.570,
            5.660,
            5.860,
            6.650,
            6.780,
            6.790,
            7.500,
            7.500,
            7.500,
            8.630,
            8.710,
            8.990,
            9.850,
            10.820,
            11.250,
            11.250,
            12.200,
            14.920,
            16.770,
            17.810,
            19.160,
            19.190,
            19.640,
            20.180,
            22.970,
        ]
    )
    return data


@pytest.fixture
def boxplot_data(plot_data):
    bs = [
        {
            "cihi": 10.82,
            "cilo": 6.1399999999999997,
            "fliers": numpy.array([22.97]),
            "iqr": 6.1099999999999994,
            "mean": 9.5888285714285733,
            "med": 7.5,
            "q1": 5.6150000000000002,
            "q3": 11.725,
            "whishi": 20.18,
            "whislo": 2.0,
        }
    ]
    return bs


@pytest.fixture
def whisk_flier_data():
    data = [
        2.20e-01,
        2.70e-01,
        3.08e-01,
        3.20e-01,
        4.10e-01,
        4.44e-01,
        4.82e-01,
        5.46e-01,
        6.05e-01,
        6.61e-01,
        7.16e-01,
        7.70e-01,
        8.24e-01,
        1.00e-03,
        4.90e-02,
        5.60e-02,
        1.40e-01,
        1.69e-01,
        1.83e-01,
        2.06e-01,
        2.10e-01,
        2.13e-01,
        2.86e-01,
        3.16e-01,
        3.40e-01,
        3.57e-01,
        3.71e-01,
        3.72e-01,
        3.78e-01,
        3.81e-01,
        3.86e-01,
        3.89e-01,
        3.90e-01,
        3.93e-01,
        4.00e-01,
        4.03e-01,
        4.10e-01,
        4.10e-01,
        4.29e-01,
        4.40e-01,
        4.40e-01,
        4.40e-01,
        4.46e-01,
        4.46e-01,
        4.50e-01,
        4.51e-01,
        4.52e-01,
        4.56e-01,
        4.60e-01,
        4.66e-01,
        4.72e-01,
        4.78e-01,
        4.81e-01,
        4.83e-01,
        4.86e-01,
        4.89e-01,
        4.98e-01,
        5.00e-01,
        5.00e-01,
        5.03e-01,
        5.18e-01,
        5.32e-01,
        5.37e-01,
        5.38e-01,
        5.68e-01,
        5.69e-01,
        5.78e-01,
        5.88e-01,
        5.94e-01,
        5.96e-01,
        6.02e-01,
        6.10e-01,
        6.10e-01,
        6.10e-01,
        6.19e-01,
        6.20e-01,
        6.20e-01,
        6.28e-01,
        6.38e-01,
        6.39e-01,
        6.42e-01,
        6.61e-01,
        6.71e-01,
        6.75e-01,
        6.80e-01,
        6.96e-01,
        7.00e-01,
        7.01e-01,
        7.09e-01,
        7.16e-01,
        7.17e-01,
        7.30e-01,
        7.62e-01,
        7.64e-01,
        7.69e-01,
        7.70e-01,
        7.77e-01,
        7.80e-01,
        8.06e-01,
        8.10e-01,
        8.23e-01,
        8.30e-01,
        8.50e-01,
        8.50e-01,
        8.56e-01,
        8.56e-01,
        8.80e-01,
        8.80e-01,
        8.93e-01,
        8.96e-01,
        8.97e-01,
        8.99e-01,
        9.22e-01,
        9.28e-01,
        9.30e-01,
        9.64e-01,
        9.65e-01,
        9.76e-01,
        9.79e-01,
        9.90e-01,
        9.99e-01,
        1.00e00,
        1.00e00,
        1.01e00,
        1.02e00,
        1.03e00,
        1.03e00,
        1.03e00,
        1.04e00,
        1.05e00,
        1.05e00,
        1.05e00,
        1.06e00,
        1.07e00,
        1.08e00,
        1.08e00,
        1.10e00,
        1.10e00,
        1.11e00,
        1.12e00,
        1.12e00,
        1.13e00,
        1.14e00,
        1.14e00,
        1.14e00,
        1.15e00,
        1.16e00,
        1.17e00,
        1.17e00,
        1.17e00,
        1.19e00,
        1.19e00,
        1.20e00,
        1.20e00,
        1.21e00,
        1.22e00,
        1.22e00,
        1.23e00,
        1.23e00,
        1.23e00,
        1.25e00,
        1.25e00,
        1.26e00,
        1.26e00,
        1.27e00,
        1.27e00,
        1.28e00,
        1.29e00,
        1.29e00,
        1.30e00,
        1.30e00,
        1.30e00,
        1.31e00,
        1.31e00,
        1.31e00,
        1.32e00,
        1.33e00,
        1.34e00,
        1.35e00,
        1.35e00,
        1.35e00,
        1.36e00,
        1.36e00,
        1.36e00,
        1.36e00,
        1.37e00,
        1.38e00,
        1.39e00,
        1.39e00,
        1.40e00,
        1.41e00,
        1.43e00,
        1.44e00,
        1.44e00,
        1.47e00,
        1.47e00,
        1.48e00,
        1.51e00,
        1.51e00,
        1.53e00,
        1.55e00,
        1.55e00,
        1.55e00,
        1.57e00,
        1.57e00,
        1.57e00,
        1.59e00,
        1.59e00,
        1.60e00,
        1.60e00,
        1.61e00,
        1.62e00,
        1.62e00,
        1.62e00,
        1.62e00,
        1.63e00,
        1.63e00,
        1.63e00,
        1.64e00,
        1.66e00,
        1.68e00,
        1.68e00,
        1.68e00,
        1.68e00,
        1.70e00,
        1.70e00,
        1.71e00,
        1.71e00,
        1.71e00,
        1.74e00,
        1.75e00,
        1.75e00,
        1.75e00,
        1.76e00,
        1.76e00,
        1.77e00,
        1.77e00,
        1.77e00,
        1.78e00,
        1.78e00,
        1.79e00,
        1.79e00,
        1.80e00,
        1.81e00,
        1.81e00,
        1.82e00,
        1.82e00,
        1.82e00,
        1.83e00,
        1.85e00,
        1.85e00,
        1.85e00,
        1.85e00,
        1.86e00,
        1.86e00,
        1.86e00,
        1.86e00,
        1.87e00,
        1.87e00,
        1.89e00,
        1.90e00,
        1.91e00,
        1.92e00,
        1.92e00,
        1.92e00,
        1.94e00,
        1.95e00,
        1.95e00,
        1.95e00,
        1.96e00,
        1.96e00,
        1.97e00,
        1.97e00,
        1.97e00,
        1.97e00,
        1.98e00,
        1.99e00,
        1.99e00,
        1.99e00,
        2.00e00,
        2.00e00,
        2.00e00,
        2.01e00,
        2.01e00,
        2.01e00,
        2.02e00,
        2.04e00,
        2.05e00,
        2.06e00,
        2.06e00,
        2.06e00,
        2.07e00,
        2.08e00,
        2.09e00,
        2.09e00,
        2.10e00,
        2.10e00,
        2.11e00,
        2.11e00,
        2.12e00,
        2.12e00,
        2.12e00,
        2.13e00,
        2.13e00,
        2.13e00,
        2.14e00,
        2.14e00,
        2.14e00,
        2.14e00,
        2.14e00,
        2.15e00,
        2.16e00,
        2.17e00,
        2.18e00,
        2.18e00,
        2.18e00,
        2.19e00,
        2.19e00,
        2.19e00,
        2.19e00,
        2.19e00,
        2.21e00,
        2.23e00,
        2.23e00,
        2.23e00,
        2.25e00,
        2.25e00,
        2.25e00,
        2.25e00,
        2.26e00,
        2.26e00,
        2.26e00,
        2.26e00,
        2.26e00,
        2.27e00,
        2.27e00,
        2.28e00,
        2.28e00,
        2.28e00,
        2.29e00,
        2.29e00,
        2.29e00,
        2.30e00,
        2.31e00,
        2.32e00,
        2.33e00,
        2.33e00,
        2.33e00,
        2.33e00,
        2.34e00,
        2.36e00,
        2.38e00,
        2.38e00,
        2.39e00,
        2.39e00,
        2.39e00,
        2.41e00,
        2.42e00,
        2.43e00,
        2.45e00,
        2.45e00,
        2.47e00,
        2.48e00,
        2.49e00,
        2.49e00,
        2.49e00,
        2.50e00,
        2.51e00,
        2.51e00,
        2.52e00,
        2.53e00,
        2.53e00,
        2.54e00,
        2.54e00,
        2.56e00,
        2.58e00,
        2.59e00,
        2.59e00,
        2.60e00,
        2.61e00,
        2.61e00,
        2.61e00,
        2.62e00,
        2.62e00,
        2.63e00,
        2.65e00,
        2.65e00,
        2.66e00,
        2.66e00,
        2.68e00,
        2.69e00,
        2.69e00,
        2.70e00,
        2.72e00,
        2.72e00,
        2.73e00,
        2.75e00,
        2.77e00,
        2.78e00,
        2.79e00,
        2.81e00,
        2.81e00,
        2.82e00,
        2.84e00,
        2.84e00,
        2.85e00,
        2.85e00,
        2.86e00,
        2.86e00,
        2.88e00,
        2.92e00,
        2.93e00,
        2.93e00,
        2.95e00,
        2.96e00,
        2.96e00,
        2.99e00,
        3.00e00,
        3.01e00,
        3.02e00,
        3.03e00,
        3.03e00,
        3.14e00,
        3.15e00,
        3.16e00,
        3.17e00,
        3.17e00,
        3.18e00,
        3.18e00,
        3.19e00,
        3.20e00,
        3.22e00,
        3.24e00,
        3.25e00,
        3.29e00,
        3.31e00,
        3.32e00,
        3.32e00,
        3.34e00,
        3.35e00,
        3.36e00,
        3.38e00,
        3.44e00,
        3.45e00,
        3.46e00,
        3.48e00,
        3.49e00,
        3.53e00,
        3.59e00,
        3.63e00,
        3.70e00,
        3.70e00,
        3.76e00,
        3.80e00,
        3.80e00,
        3.80e00,
        3.83e00,
        3.84e00,
        3.88e00,
        3.90e00,
        3.91e00,
        3.96e00,
        3.97e00,
        3.97e00,
        4.02e00,
        4.03e00,
        4.06e00,
        4.12e00,
        4.19e00,
        4.21e00,
        4.53e00,
        4.56e00,
        4.61e00,
        4.62e00,
        4.73e00,
        5.13e00,
        5.21e00,
        5.40e00,
        5.98e00,
        6.12e00,
        6.94e00,
        7.38e00,
        7.56e00,
        8.06e00,
        1.38e01,
        1.51e01,
        1.82e01,
    ]
    return data


@pytest.fixture
@helpers.seed
def jp_data():
    pyplot.rcdefaults()
    N = 37
    df = pandas.DataFrame(
        {
            "A": numpy.random.normal(size=N),
            "B": numpy.random.lognormal(mean=0.25, sigma=1.25, size=N),
            "C": numpy.random.lognormal(mean=1.25, sigma=0.75, size=N),
        }
    )

    return df


@pytest.fixture
@helpers.seed
def cat_hist_data():
    N = 100
    years = [2011, 2012, 2013, 2014]
    df = pandas.DataFrame(
        {
            "depth": numpy.random.uniform(low=0.2, high=39, size=N),
            "year": numpy.random.choice(years, size=N),
            "has_outflow": numpy.random.choice([False, True], size=N),
        }
    )
    return df


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_rotateTickLabels_xaxis():
    fig, ax = pyplot.subplots()
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["AAA", "BBB", "CCC"])
    viz.rotate_tick_labels(ax, 60, "x")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_rotateTickLabels_yaxis():
    fig, ax = pyplot.subplots()
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["AAA", "BBB", "CCC"])
    viz.rotate_tick_labels(ax, -30, "y")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_rotateTickLabels_both():
    fig, ax = pyplot.subplots()
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["XXX", "YYY", "ZZZ"])

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["AAA", "BBB", "CCC"])
    viz.rotate_tick_labels(ax, 45, "both")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_log_formatter():
    fig, ax = pyplot.subplots()
    ax.plot([1, 5], [0.0005, 5e6])
    ax.set_yscale("log")
    ax.set_ylim([0.0001, 1e7])
    ax.yaxis.set_major_formatter(viz.log_formatter())
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_log_formatter_alt():
    fig, ax = pyplot.subplots()
    ax.plot([1, 5], [0.0005, 5e6])
    ax.set_yscale("log")
    ax.set_ylim([0.0001, 1e7])
    ax.yaxis.set_major_formatter(viz.log_formatter(use_1x=False))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_log_formatter_alt_2():
    fig, (ax1, ax2) = pyplot.subplots(ncols=2)
    ax1.set_yscale("log")
    ax1.set_ylim((1e-4, 1e5))
    ax1.yaxis.set_major_formatter(viz.log_formatter(threshold=4))

    ax2.set_yscale("log")
    ax2.set_ylim((1e-7, 1e5))
    ax2.yaxis.set_major_formatter(viz.log_formatter(threshold=4, use_1x=False))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_gridlines_basic(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    viz.gridlines(ax, "xlabel", "ylabel")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_gridlines_ylog(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    viz.gridlines(ax, "xlabel", "ylabel", yscale="log")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_gridlines_ylog_noyminor(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    viz.gridlines(ax, "xlabel", "ylabel", yscale="log", yminor=False)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_one2one():
    fig, ax = pyplot.subplots()
    ax.set_xlim([-2, 5])
    ax.set_ylim([-3, 3])
    viz.one2one_line(ax, label="Equality", lw=5, ls="--")
    ax.legend()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_defaultlabels(jp_data):
    with seaborn.axes_style("ticks"):
        jg1 = viz.jointplot(x="B", y="C", data=jp_data, one2one=False, color="b")
    assert jg1.ax_joint.get_xlabel() == "B"
    assert jg1.ax_joint.get_ylabel() == "C"
    # nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_xlim()), [0, 16])
    # nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_ylim()), [0, 16])
    return jg1.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_xlabeled(jp_data):
    with seaborn.axes_style("ticks"):
        jg2 = viz.jointplot(
            x="B", y="C", data=jp_data, one2one=False, color="g", xlabel="Quantity B"
        )
    assert jg2.ax_joint.get_xlabel() == "Quantity B"
    return jg2.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_ylabeled(jp_data):
    with seaborn.axes_style("ticks"):
        jg3 = viz.jointplot(
            x="B", y="C", data=jp_data, one2one=False, color="r", ylabel="Quantity C"
        )
    assert jg3.ax_joint.get_ylabel() == "Quantity C"
    return jg3.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_bothlabeled(jp_data):
    with seaborn.axes_style("ticks"):
        jg4 = viz.jointplot(
            x="B",
            y="C",
            data=jp_data,
            one2one=False,
            color="k",
            xlabel="Quantity B",
            ylabel="Quantity C",
        )
    assert jg4.ax_joint.get_xlabel() == "Quantity B"
    assert jg4.ax_joint.get_ylabel() == "Quantity C"
    return jg4.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_zerominFalse(jp_data):
    with seaborn.axes_style("ticks"):
        jg1 = viz.jointplot(x="A", y="C", data=jp_data, zeromin=False, one2one=False)
    nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_xlim()), [-3, 3])
    nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_ylim()), [0, 16])
    return jg1.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_one2one(jp_data):
    with seaborn.axes_style("ticks"):
        jg1 = viz.jointplot(x="B", y="C", data=jp_data, one2one=True)
    nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_xlim()), [0, 16])
    nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_ylim()), [0, 16])
    return jg1.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE * 1.6)
def test_jointplot_one2one_zerominFalse(jp_data):
    with seaborn.axes_style("ticks"):
        jg1 = viz.jointplot(x="A", y="C", data=jp_data, one2one=True, zeromin=False)
    nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_xlim()), [-3, 16])
    nptest.assert_array_equal(numpy.round(jg1.ax_joint.get_ylim()), [-3, 16])
    return jg1.figure


@pytest.mark.parametrize(
    ("case", "xform_in", "xform_out"),
    [
        ("linear", utils.no_op, utils.no_op),
        ("natlog", numpy.log, numpy.exp),
        ("log10", numpy.log10, lambda x: 10**x),
    ],
)
@pytest.mark.parametrize("key", ["whishi", "whislo", "fliers"])
def test_whiskers_and_fliers(whisk_flier_data, case, xform_in, xform_out, key):
    expected_results = {
        "linear": {
            "whishi": 4.62,
            "whislo": 0.00111,
            "fliers": numpy.array(
                [
                    4.730,
                    5.130,
                    5.210,
                    5.400,
                    5.980,
                    6.120,
                    6.940,
                    7.380,
                    7.560,
                    8.060,
                    13.800,
                    15.100,
                    18.200,
                ]
            ),
        },
        "natlog": {
            "whishi": 8.060,
            "whislo": 0.2700,
            "fliers": numpy.array(
                [
                    2.200e-01,
                    1.000e-03,
                    4.900e-02,
                    5.600e-02,
                    1.400e-01,
                    1.690e-01,
                    1.830e-01,
                    2.060e-01,
                    2.100e-01,
                    2.130e-01,
                    1.380e01,
                    1.510e01,
                    1.820e01,
                ]
            ),
        },
        "log10": {
            "whishi": 8.060,
            "whislo": 0.2700,
            "fliers": numpy.array(
                [
                    2.200e-01,
                    1.000e-03,
                    4.900e-02,
                    5.600e-02,
                    1.400e-01,
                    1.690e-01,
                    1.830e-01,
                    2.060e-01,
                    2.100e-01,
                    2.130e-01,
                    1.380e01,
                    1.510e01,
                    1.820e01,
                ]
            ),
        },
    }
    q1 = numpy.percentile(whisk_flier_data, 25)
    q3 = numpy.percentile(whisk_flier_data, 75)
    result = viz.whiskers_and_fliers(
        xform_in(whisk_flier_data), xform_in(q1), xform_in(q3), transformout=xform_out
    )

    expected = expected_results[case]
    nptest.assert_array_almost_equal(result[key], expected[key], decimal=3)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_boxplot_basic(boxplot_data):
    fig, ax = pyplot.subplots()
    viz.boxplot(boxplot_data, ax=ax, shownotches=True, patch_artist=False)
    ax.set_xlim((0, 2))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_boxplot_with_mean(boxplot_data):
    fig, ax = pyplot.subplots()
    viz.boxplot(
        boxplot_data,
        ax=ax,
        shownotches=True,
        patch_artist=True,
        marker="^",
        color="red",
        showmean=True,
    )
    ax.set_xlim((0, 2))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_probplot_prob(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, xlabel="Test xlabel")
    assert isinstance(fig, pyplot.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_probplot_qq(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(
        plot_data, ax=ax, axtype="qq", ylabel="Test label", scatter_kws=dict(color="r")
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_probplot_pp(plot_data):
    fig, ax = pyplot.subplots()
    scatter_kws = dict(color="b", linestyle="--", markeredgecolor="g", markerfacecolor="none")
    fig = viz.probplot(
        plot_data,
        ax=ax,
        axtype="pp",
        yscale="linear",
        xlabel="test x",
        ylabel="test y",
        scatter_kws=scatter_kws,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_probplot_prob_bestfit(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, xlabel="Test xlabel", bestfit=True)
    assert isinstance(fig, pyplot.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_probplot_qq_bestfit(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, axtype="qq", bestfit=True, ylabel="Test label")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_probplot_pp_bestfit(plot_data):
    fig, ax = pyplot.subplots()
    scatter_kws = {"marker": "s", "color": "red"}
    line_kws = {"linestyle": "--", "linewidth": 3}
    fig = viz.probplot(
        plot_data,
        ax=ax,
        axtype="pp",
        yscale="linear",
        xlabel="test x",
        bestfit=True,
        ylabel="test y",
        scatter_kws=scatter_kws,
        line_kws=line_kws,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test__connect_spines():
    fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(ncols=2, nrows=2)
    viz._connect_spines(ax1, ax2, 0.5, 0.75)
    viz._connect_spines(ax3, ax4, 0.9, 0.1, linewidth=2, color="r", linestyle="dashed")
    return fig


@pytest.mark.xfail
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_parallel_coordinates():
    df = seaborn.load_dataset("iris")
    fig = viz.parallel_coordinates(df, hue="species")
    return fig


@pytest.mark.xfail(TOLERANCE > 15, reason="GH Action weirdness")
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_categorical_histogram_simple(cat_hist_data):
    with seaborn.axes_style("ticks"):
        bins = numpy.arange(5, 35, 5)
        fig = viz.categorical_histogram(cat_hist_data, "depth", bins)
        return fig.figure


@pytest.mark.xfail(TOLERANCE > 15, reason="GH Action weirdness")
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_categorical_histogram_complex(cat_hist_data):
    with seaborn.axes_style("ticks"):
        bins = numpy.arange(5, 35, 5)
        fig = viz.categorical_histogram(cat_hist_data, "depth", bins, hue="year", row="has_outflow")
        return fig.figure
