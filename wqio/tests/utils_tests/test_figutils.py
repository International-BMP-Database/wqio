import pytest
import numpy.testing as nptest

import numpy
import matplotlib
from matplotlib import pyplot

from wqio.utils import figutils


BASELINE_IMAGES = '../_baseline_images/figutils_tests'


@pytest.fixture
def plot_data():
    pyplot.rcdefaults()
    data = numpy.array([
         3.113,   3.606,   4.046,   4.046,   4.71 ,   6.14 ,   6.978,
         2.   ,   4.2  ,   4.62 ,   5.57 ,   5.66 ,   5.86 ,   6.65 ,
         6.78 ,   6.79 ,   7.5  ,   7.5  ,   7.5  ,   8.63 ,   8.71 ,
         8.99 ,   9.85 ,  10.82 ,  11.25 ,  11.25 ,  12.2  ,  14.92 ,
        16.77 ,  17.81 ,  19.16 ,  19.19 ,  19.64 ,  20.18 ,  22.97
    ])
    return data


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_setProblimits_x_ax_LT50():
    fig, ax = pyplot.subplots()
    ax.set_xscale('prob')
    figutils.setProbLimits(ax, 37, which='x')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_setProblimits_y_ax_LT50():
    fig, ax = pyplot.subplots()
    ax.set_yscale('prob')
    figutils.setProbLimits(ax, 37, which='y')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_setProblimits_y_ax_GT50():
    fig, ax = pyplot.subplots()
    ax.set_yscale('prob')
    figutils.setProbLimits(ax, 98, which='y')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_setProblimits_y_ax_GT100():
    fig, ax = pyplot.subplots()
    ax.set_yscale('prob')
    figutils.setProbLimits(ax, 457, which='y')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_gridlines_basic(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    figutils.gridlines(ax, 'xlabel', 'ylabel')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_gridlines_ylog(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    figutils.gridlines(ax, 'xlabel', 'ylabel', yscale='log')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_gridlines_ylog_noyminor(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    figutils.gridlines(ax, 'xlabel', 'ylabel', yscale='log', yminor=False)
    return fig


class Test_shiftedColorMap:
    def setup(self):
        self.orig_cmap = matplotlib.cm.coolwarm
        self.start = 0.0
        self.midpoint = 0.75
        self.stop = 1.0
        self.shifted_cmap = figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.midpoint,
            stop=self.stop
        )
        pass

    def teardown(self):
        pyplot.close('all')

    def test_bad_start_low(self):
        with pytest.raises(ValueError):
            figutils.shiftedColorMap(
                self.orig_cmap,
                start=-0.1,
                midpoint=self.midpoint,
                stop=self.stop
            )

    def test_bad_start_high(self):
        with pytest.raises(ValueError):
            figutils.shiftedColorMap(
                self.orig_cmap,
                start=self.midpoint+0.01,
                midpoint=self.midpoint,
                stop=self.stop
            )

    def test_bad_midpoint_low(self):
        with pytest.raises(ValueError):
            figutils.shiftedColorMap(
                self.orig_cmap,
                start=self.start,
                midpoint=self.start,
                stop=self.stop
            )

    def test_bad_midpoint_high(self):
        with pytest.raises(ValueError):
            figutils.shiftedColorMap(
                self.orig_cmap,
                start=self.start,
                midpoint=self.stop,
                stop=self.stop
            )

    def test_bad_stop_low(self):
        with pytest.raises(ValueError):
            figutils.shiftedColorMap(
                self.orig_cmap,
                start=self.start,
                midpoint=self.midpoint,
                stop=self.midpoint-0.01
            )

    def test_bad_stop_high(self):
        with pytest.raises(ValueError):
            figutils.shiftedColorMap(
                self.orig_cmap,
                start=self.start,
                midpoint=self.midpoint,
                stop=1.1
            )

    def test_start(self):
        assert self.orig_cmap(self.start) == self.shifted_cmap(0.0)

    def test_midpoint(self):
        nptest.assert_array_almost_equal(
            numpy.array(self.orig_cmap(0.5)),
            numpy.array(self.shifted_cmap(self.midpoint)),
            decimal=2
        )

    def test_stop(self):
        assert self.orig_cmap(self.stop) == self.shifted_cmap(1.0)
