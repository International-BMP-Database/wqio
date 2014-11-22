import sys
import os
import subprocess
if sys.version_info.major == 3:
    from io import StringIO
else:
    from StringIO import StringIO

from nose.tools import *
import numpy as np
import numpy.testing as nptest

from pybmp import testing
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.testing.decorators import image_comparison
import scipy.stats as stats

from pybmp.utils import figutils, ros


class test_axes_methods:
    def setup(self):
        self.fig, self.ax = plt.subplots()
        self.data = testing.getTestROSData()
        self.ros = ros.MR(self.data)
        self.mean = self.ros.data.final_data.mean()
        self.median = self.ros.data.final_data.median()
        self.median_ci = [(self.median * 0.75, self.median * 1.25)]
        self.prefix = testing.testutils.setup_prefix('utils.figutils')

    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nottest
    def savefig(self, filename):
        self.fig.savefig(self.makePath(filename))

    def teardown(self):
        plt.close(self.fig)

    @nptest.dec.skipif(True)
    def test__probability_axis(self):
        probs = figutils._get_probs(25)
        self.ax.plot([0.05, 0.15], [0.05, 0.15])
        figutils._probability_axis(self.ax, stats.norm, probs, axis='both')
        self.savefig('prob_axis_test.png')
        for ylabel, prob in zip(self.ax.get_yticklabels(), probs):
            assert_equal(ylabel.get_text(), str(prob*100))
            assert_equal(ylabel.get_horizontalalignment(), 'right')
            assert_equal(ylabel.get_verticalalignment(), 'center')
            assert_equal(ylabel.get_rotation_mode(), None)
            assert_equal(ylabel.get_rotation(), 0.0)

        for xlabel, prob in zip(self.ax.get_xticklabels(), probs):
            assert_equal(xlabel.get_text(), str(prob*100))
            assert_equal(xlabel.get_horizontalalignment(), 'right')
            assert_equal(xlabel.get_verticalalignment(), 'center')
            assert_equal(xlabel.get_rotation_mode(), 'anchor')
            assert_equal(xlabel.get_rotation(), 45.0)

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

    def test_formatStatAxes_prob(self):
        figutils.formatStatAxes(self.ax, 'probplot', N=37, doLegend=False)
        self.savefig('formatStatAxes_prob.png')

    @nptest.raises(ValueError)
    def test_formatStatAxes_prob_raises(self):
        figutils.formatStatAxes(self.ax, 'probplot')

    def test_formatStatAxes_box(self):
        figutils.formatStatAxes(self.ax, 'boxplot', datalabels=['1', '2'], pos=2)
        self.savefig('formatStatAxes_box.png')

    @nptest.raises(ValueError)
    def test_formatStatAxes_box_raises_pos(self):
        figutils.formatStatAxes(self.ax, 'boxplot', datalabels=['1', '2'])

    @nptest.raises(ValueError)
    def test_formatStatAxes_box_raises_labels(self):
        figutils.formatStatAxes(self.ax, 'boxplot', pos=2)

    @nptest.raises(ValueError)
    def test_formatStatAxes_raises(self):
        figutils.formatStatAxes(self.ax, 'junk', pos=2)

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
        assert_list_equal(rendered_labels, expected_labels)

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
        assert_list_equal(rendered_labels, expected_labels)


class test_scatterHistogram:
    def setup(self):
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2)

        self.x = np.array([
            7.48015314, 4.6222714 , 9.01095731, 7.98639611,
            6.56520105, 8.28668008, 8.88881000, 9.23505688,
            5.52075174, 7.1344543 , 5.36104144, 6.31328249,
            6.53392651, 8.96088653, 4.78989627, 3.27250932,
            8.41477798, 7.54291787, 7.36982318, 5.74657018,
            8.12647486, 4.6129123 , 6.02550153, 6.84299899,
            5.63095163, 8.20904575, 7.68361688, 10.90631482,
            7.20938852, 4.71375399, 8.11806228, 5.90488422,
            4.12479778, 8.15594211, 11.16551958, 7.45410655,
            8.96646264
        ])

        self.y = np.array([
            5.30995729, 4.90652812, 4.64388533, 4.58074539,
            3.97922263, 5.96186786, 5.11284324, 1.99402179,
            5.24351585, -0.45201096, 3.45148458, 6.72188106,
            3.59817747, 6.84682235, 5.17153649, 7.81965041,
            1.40317254, 4.39889747, -0.76829593, 1.4526049 ,
            3.36884806, 3.58727632, 3.25783271, 2.38286936,
            8.53593809, 4.00885338, 6.42957833, 5.91891113,
            3.14951288, -1.64253263, 1.52852978, 1.60945809,
            5.0613945 , 4.63739907, 1.58679716,  1.91932307,
            3.32511911
        ])
        self.prefix = testing.testutils.setup_prefix('utils.figutils')

    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    def teardown(self):
        plt.close('all')

    def test_baseline(self):
        figname = self.makePath('scatter_hist_basecase.png')
        fig = figutils.scatterHistogram(self.x, self.y, figname=figname)
        assert_true(isinstance(fig, plt.Figure))

    def test_manualaxes(self):
        figname = self.makePath('scatter_manualaxes.png')
        fig = figutils.scatterHistogram(self.x, self.y, figname=figname,
                                        axes=self.axes.flatten()[:3])
        assert_true(isinstance(fig, plt.Figure))

    def test_manuallimits(self):
        figname = self.makePath('scatter_manuallimits.png')
        fig = figutils.scatterHistogram(self.x, self.y, figname=figname,
                                        xlim={'left': -3, 'right': 12},
                                        ylim={'bottom': -3, 'top': 12})
        assert_true(isinstance(fig, plt.Figure))

    def test_customsymbology(self):
        scatter_symbols = dict(
            linestyle='none',
            marker='o',
            markerfacecolor='CornflowerBlue',
            markeredgecolor='white',
            alpha=0.75,
            markeredgewidth=0.5
        )

        bar_symbols = dict(
            alpha=0.75,
            facecolor='DarkGreen',
            edgecolor='none',
            bins=10,
            rwidth=0.8
        )
        figname = self.makePath('scatter_custom.png')
        fig = figutils.scatterHistogram(self.x, self.y, figname=figname,
                                        symbolkwds=scatter_symbols,
                                        barkwds=bar_symbols)
        assert_true(isinstance(fig, plt.Figure))

    @raises(ValueError)
    def test_badaxeslength(self):
        fig = figutils.scatterHistogram(self.x, self.y, axes=self.axes)

    @raises(ValueError)
    def test_badaxestype(self):
        fig = figutils.scatterHistogram(self.x, self.y, axes=[1, 2, 3])


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
        pass

    @raises(ValueError)
    def test_bad_start_low(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=-0.1,
            midpoint=self.midpoint,
            stop=self.stop
        )

    @raises(ValueError)
    def test_bad_start_high(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.midpoint+0.01,
            midpoint=self.midpoint,
            stop=self.stop
        )

    @raises(ValueError)
    def test_bad_midpoint_low(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.start,
            stop=self.stop
        )

    @raises(ValueError)
    def test_bad_midpoint_high(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.stop,
            stop=self.stop
        )

    @raises(ValueError)
    def test_bad_stop_low(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.midpoint,
            stop=self.midpoint-0.01
        )

    @raises(ValueError)
    def test_bad_stop_high(self):
        figutils.shiftedColorMap(
            self.orig_cmap,
            start=self.start,
            midpoint=self.midpoint,
            stop=1.1
        )

    def test_start(self):
        assert_tuple_equal(self.orig_cmap(self.start), self.shifted_cmap(0.0))

    def test_midpoint(self):
        nptest.assert_array_almost_equal(
            np.array(self.orig_cmap(0.5)),
            np.array(self.shifted_cmap(self.midpoint)),
            decimal=2
        )

    def test_stop(self):
        assert_tuple_equal(self.orig_cmap(self.stop), self.shifted_cmap(1.0))
