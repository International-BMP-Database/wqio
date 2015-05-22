import sys
import os
import subprocess
from six import StringIO

import nose.tools as nt
import numpy as np
import numpy.testing as nptest

from wqio import testing
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.testing.decorators import image_comparison
import scipy.stats as stats

from wqio.utils import figutils, ros


class test_axes_methods:
    def setup(self):
        self.fig, self.ax = plt.subplots()
        self.data = testing.getTestROSData()
        self.ros = ros.MR(self.data)
        self.mean = self.ros.data.final_data.mean()
        self.median = self.ros.data.final_data.median()
        self.median_ci = [(self.median * 0.75, self.median * 1.25)]
        self.prefix = testing.testutils.setup_prefix('utils.figutils')

    def teardown(self):
        plt.close('all')

    @nt.nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nt.nottest
    def savefig(self, filename):
        self.fig.savefig(self.makePath(filename))

    def test_rotateTickLabels_xaxis(self):
        self.ax.set_xticks([1, 2, 3])
        self.ax.set_xticklabels(['AAA', 'BBB', 'CCC'])
        figutils.rotateTickLabels(self.ax, 60, 'x')
        self.savefig('rotate_xticks.png')

    def test_rotateTickLabels_yaxis(self):
        self.ax.set_yticks([1, 2, 3])
        self.ax.set_yticklabels(['AAA', 'BBB', 'CCC'])
        figutils.rotateTickLabels(self.ax, -30, 'y')
        self.savefig('rotate_yticks.png')

    def test_setProblimits_x_ax_LT50(self):
        self.ax.set_xscale('prob')
        figutils.setProbLimits(self.ax, 37, which='x')
        self.savefig('problimts_xaxLT50.png')

    def test_setProblimits_y_ax_LT50(self):
        self.ax.set_yscale('prob')
        figutils.setProbLimits(self.ax, 37, which='y')
        self.savefig('problimts_yaxLT50.png')

    def test_setProblimits_y_ax_GT50(self):
        self.ax.set_yscale('prob')
        figutils.setProbLimits(self.ax, 98, which='y')
        self.savefig('problimts_yaxGT50.png')

    def test_setProblimits_y_ax_GT100(self):
        self.ax.set_yscale('prob')
        figutils.setProbLimits(self.ax, 457, which='y')
        self.savefig('problimts_yaxGT100.png')

    def test_gridlines(self):
        self.ax.plot(self.ros.data.final_data)
        figutils.gridlines(self.ax, 'xlabel', 'ylabel')
        self.savefig('gridlines.png')

    def test_boxplot(self):
        figutils.boxplot(self.ax, self.ros.data.final_data, 1)
        self.savefig('boxplot.png')

    def test_boxplot_manual_median(self):
        figutils.boxplot(self.ax, self.ros.data.final_data,
                         1, median=self.median,
                         CIs=self.median_ci)
        self.savefig('boxplot_manualmed.png')

    def test_boxplot_with_mean(self):
        figutils.boxplot(self.ax, self.ros.data.final_data, 1,
                         median=self.median,
                         CIs=self.median_ci,
                         mean=self.mean,
                         meanmarker='^', meancolor='red')
        self.savefig('boxplot_with_mean.png')

    def test_formatBoxplot(self):
        bp = self.ax.boxplot(self.ros.data.final_data, notch=1)
        figutils.formatBoxplot(bp)
        self.savefig('formatted_boxplot.png')

    def test_probplot(self):
        fig = figutils.probplot(self.ros.data.final_data, ax=self.ax)
        self.savefig('probplot.png')

    @nptest.dec.skipif(sys.platform != 'win32' or not usetex)
    def test_logLabelFormatter(self):
        self.ax.plot([1, 5], [0.0005, 5e6])
        self.ax.set_yscale('log')
        self.ax.set_ylim([0.0001, 1e7])
        self.ax.yaxis.set_major_formatter(mticker.FuncFormatter(figutils.logLabelFormatter))
        plt.draw()
        rendered_labels = [tick.get_text() for tick in self.ax.get_yticklabels()]
        expected_labels = [
            r'', r'$1 \times 10 ^ {-4}$', r'0.001', r'0.01', r'0.1',
            r'1.0', r'10', r'100', r'1000', r'$1 \times 10 ^ {4}$',
            r'$1 \times 10 ^ {5}$', r'$1 \times 10 ^ {6}$',
            r'$1 \times 10 ^ {7}$', ''
        ]
        nt.assert_list_equal(rendered_labels, expected_labels)

    @nptest.dec.skipif(sys.platform != 'win32' or not usetex)
    def test_alt_logLabelFormatter(self):
        self.ax.plot([1, 5], [0.0005, 5e6])
        self.ax.set_yscale('log')
        self.ax.set_ylim([0.0001, 1e7])
        self.ax.yaxis.set_major_formatter(mticker.FuncFormatter(figutils.alt_logLabelFormatter))
        plt.draw()
        rendered_labels = [tick.get_text() for tick in self.ax.get_yticklabels()]
        expected_labels = [
            r'', r'$10 ^ {-4}$', r'0.001', r'0.01', r'0.1',
            r'1.0', r'10', r'100', r'1000', r'$10 ^ {4}$',
            r'$10 ^ {5}$', r'$10 ^ {6}$', r'$10 ^ {7}$', ''
        ]
        nt.assert_list_equal(rendered_labels, expected_labels)


@nt.raises(NotImplementedError)
def test_scatterHistogram():
    figutils.scatterHistogram()


class test_shiftedColorMap:
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
        plt.close('all')

    @nt.raises(ValueError)
    def test_bad_start_low(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=-0.1,
            midpoint=self.midpoint,
            stop=self.stop
        )

    @nt.raises(ValueError)
    def test_bad_start_high(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.midpoint+0.01,
            midpoint=self.midpoint,
            stop=self.stop
        )

    @nt.raises(ValueError)
    def test_bad_midpoint_low(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.start,
            stop=self.stop
        )

    @nt.raises(ValueError)
    def test_bad_midpoint_high(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.stop,
            stop=self.stop
        )

    @nt.raises(ValueError)
    def test_bad_stop_low(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.midpoint,
            stop=self.midpoint-0.01
        )

    @nt.raises(ValueError)
    def test_bad_stop_high(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.midpoint,
            stop=1.1
        )

    def test_start(self):
        nt.assert_tuple_equal(self.orig_cmap(self.start), self.shifted_cmap(0.0))

    def test_midpoint(self):
        nptest.assert_array_almost_equal(
            np.array(self.orig_cmap(0.5)),
            np.array(self.shifted_cmap(self.midpoint)),
            decimal=2
        )

    def test_stop(self):
        nt.assert_tuple_equal(self.orig_cmap(self.stop), self.shifted_cmap(1.0))
