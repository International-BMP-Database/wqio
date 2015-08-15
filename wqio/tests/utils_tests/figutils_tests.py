import sys
import os
import subprocess
from six import StringIO

import numpy as np
import pandas

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import seaborn.apionly as seaborn

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.testing.decorators import image_comparison, cleanup
import scipy.stats as stats

import nose.tools as nt
import numpy.testing as nptest
from wqio import testing

from wqio.utils import figutils
from wqio.algo import ros


@nt.nottest
def setup_plot_data():
    plt.rcdefaults()
    data = testing.getTestROSData()
    mr = ros.MR(data)
    return mr

@cleanup
class test__check_ax(object):
    @nt.raises(ValueError)
    def test_bad_value(self):
        figutils._check_ax('junk')

    def test_with_ax(self):
        fig, ax = plt.subplots()
        fig1, ax1 = figutils._check_ax(ax)
        nt.assert_true(isinstance(ax1, plt.Axes))
        nt.assert_true(isinstance(fig1, plt.Figure))
        nt.assert_true(ax1 is ax)
        nt.assert_true(fig1 is fig)

    def test_with_None(self):
        fig1, ax1 = figutils._check_ax(None)
        nt.assert_true(isinstance(ax1, plt.Axes))
        nt.assert_true(isinstance(fig1, plt.Figure))


@image_comparison(baseline_images=['test_rotateTickLabels_xaxis'], extensions=['png'])
def test_rotateTickLabels_xaxis():
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['AAA', 'BBB', 'CCC'])
    figutils.rotateTickLabels(ax, 60, 'x')


@image_comparison(baseline_images=['test_rotateTickLabels_yaxis'], extensions=['png'])
def test_rotateTickLabels_yaxis():
    fig, ax = plt.subplots()
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['AAA', 'BBB', 'CCC'])
    figutils.rotateTickLabels(ax, -30, 'y')


@image_comparison(baseline_images=['test_rotateTickLabels_both'], extensions=['png'])
def test_rotateTickLabels_both():
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['XXX', 'YYY', 'ZZZ'])

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['AAA', 'BBB', 'CCC'])
    figutils.rotateTickLabels(ax, 45, 'both')


@image_comparison(baseline_images=['test_setProblimits_x_ax_LT50'], extensions=['png'])
def test_setProblimits_x_ax_LT50():
    fig, ax = plt.subplots()
    ax.set_xscale('prob')
    figutils.setProbLimits(ax, 37, which='x')


@image_comparison(baseline_images=['test_setProblimits_y_ax_LT50'], extensions=['png'])
def test_setProblimits_y_ax_LT50():
    fig, ax = plt.subplots()
    ax.set_yscale('prob')
    figutils.setProbLimits(ax, 37, which='y')


@image_comparison(baseline_images=['test_setProblimits_y_ax_GT50'], extensions=['png'])
def test_setProblimits_y_ax_GT50():
    fig, ax = plt.subplots()
    ax.set_yscale('prob')
    figutils.setProbLimits(ax, 98, which='y')


@image_comparison(baseline_images=['test_setProblimits_y_ax_GT100'], extensions=['png'])
def test_setProblimits_y_ax_GT100():
    fig, ax = plt.subplots()
    ax.set_yscale('prob')
    figutils.setProbLimits(ax, 457, which='y')


@image_comparison(baseline_images=['test_gridlines_basic'], extensions=['png'])
def test_gridlines_basic():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    ax.plot(mr.data.final_data)
    figutils.gridlines(ax, 'xlabel', 'ylabel')


@image_comparison(baseline_images=['test_gridlines_ylog'], extensions=['png'])
def test_gridlines_ylog():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    ax.plot(mr.data.final_data)
    figutils.gridlines(ax, 'xlabel', 'ylabel', yscale='log')


@image_comparison(baseline_images=['test_gridlines_ylog_noyminor'], extensions=['png'])
def test_gridlines_ylog_noyminor():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    ax.plot(mr.data.final_data)
    figutils.gridlines(ax, 'xlabel', 'ylabel', yscale='log', yminor=False)


@nt.nottest
def setup_jointplot():
    plt.rcdefaults()
    np.random.seed(0)
    N = 37
    df = pandas.DataFrame({
        'A': np.random.normal(size=N),
        'B': np.random.lognormal(mean=0.25, sigma=1.25, size=N),
        'C': np.random.lognormal(mean=1.25, sigma=0.75, size=N)
    })

    return df

@image_comparison(
    baseline_images=['test_jointplot_defaultlabels', 'test_jointplot_xlabeled',
                     'test_jointplot_ylabeled', 'test_jointplot_bothlabeled'],
    extensions=['png']
)
def test_jointplot_basic():
    df = setup_jointplot()
    jg1 = figutils.jointplot(x='B', y='C', data=df, one2one=False, color='b')
    fig1 = jg1.fig

    jg2 = figutils.jointplot(x='B', y='C', data=df, one2one=False, color='g', xlabel='Quantity B')
    fig2 = jg2.fig

    jg3 = figutils.jointplot(x='B', y='C', data=df, one2one=False, color='r', ylabel='Quantity C')
    fig3 = jg3.fig

    jg4 = figutils.jointplot(x='B', y='C', data=df, one2one=False, color='k',
                             xlabel='Quantity B', ylabel='Quantity C')
    fig4 = jg4.fig


@image_comparison(baseline_images=['test_jointplot_zerominFalse'], extensions=['png'])
def test_jointplot_zerominFalse():
    df = setup_jointplot()
    jg1 = figutils.jointplot(x='A', y='C', data=df, zeromin=False, one2one=False)
    fig1 = jg1.fig


@image_comparison(baseline_images=['test_jointplot_one2one'], extensions=['png'])
def test_jointplot_one2one():
    df = setup_jointplot()
    jg1 = figutils.jointplot(x='B', y='C', data=df, one2one=True)
    fig1 = jg1.fig


@nt.raises(NotImplementedError)
def test_scatterHistogram():
    figutils.scatterHistogram()


@image_comparison(baseline_images=['test_boxplot_basic'], extensions=['png'])
def test_boxplot_basic():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    figutils.boxplot(ax, mr.data.final_data, 1)


@image_comparison(baseline_images=['test_boxplot_manual_median'], extensions=['png'])
def test_boxplot_manual_median():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    figutils.boxplot(ax, mr.data.final_data, 1, median=7.75, CIs=[7.5, 8.25])


@image_comparison(baseline_images=['test_boxplot_with_mean'], extensions=['png'])
def test_boxplot_with_mean():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    figutils.boxplot(ax, mr.data.final_data, 1, median=7.75, CIs=[7.5, 8.25],
                     mean=7.25, meanmarker='^', meancolor='red')


@image_comparison(baseline_images=['test_boxplot_formatted'], extensions=['png'])
def test_boxplot_formatted():
    mr = setup_plot_data()
    fig, ax = plt.subplots()
    bp = figutils.boxplot(ax, mr.data.final_data, 1, median=7.75, CIs=[7.5, 8.25],
                          mean=7.25, meanmarker='^', meancolor='red')
    figutils.formatBoxplot(bp)


@image_comparison(baseline_images=['test_probplot_prob'], extensions=['png'])
def test_probplot_prob():
    fig, ax = plt.subplots()
    mr = setup_plot_data()
    fig = figutils.probplot(mr.data.final_data, ax=ax, xlabel='Test xlabel')
    nt.assert_true(isinstance(fig, plt.Figure))


@image_comparison(baseline_images=['test_probplot_qq'], extensions=['png'])
def test_probplot_qq():
    fig, ax = plt.subplots()
    mr = setup_plot_data()
    fig = figutils.probplot(mr.data.final_data, ax=ax, axtype='qq', color='r', ylabel='Test label')


@image_comparison(baseline_images=['test_probplot_pp'], extensions=['png'])
def test_probplot_pp():
    fig, ax = plt.subplots()
    mr = setup_plot_data()
    fig = figutils.probplot(mr.data.final_data, ax=ax, axtype='pp', color='g',
                            linestyle='--', yscale='linear', xlabel='test x',
                            ylabel='test y')


@image_comparison(baseline_images=['test_probplot_prob_bestfit'], extensions=['png'])
def test_probplot_prob_bestfit():
    fig, ax = plt.subplots()
    mr = setup_plot_data()
    fig = figutils.probplot(mr.data.final_data, ax=ax, xlabel='Test xlabel',
                            bestfit=True)
    nt.assert_true(isinstance(fig, plt.Figure))


@image_comparison(baseline_images=['test_probplot_qq_bestfit'], extensions=['png'])
def test_probplot_qq_bestfit():
    fig, ax = plt.subplots()
    mr = setup_plot_data()
    fig = figutils.probplot(mr.data.final_data, ax=ax, axtype='qq', color='r',
                            bestfit=True, ylabel='Test label')


@image_comparison(baseline_images=['test_probplot_pp_bestfit'], extensions=['png'])
def test_probplot_pp_bestfit():
    fig, ax = plt.subplots()
    mr = setup_plot_data()
    fig = figutils.probplot(mr.data.final_data, ax=ax, axtype='pp', color='g',
                            linestyle='--', yscale='linear', xlabel='test x',
                            bestfit=True, ylabel='test y')


@image_comparison(baseline_images=['test_logLabelFormatter'], extensions=['png'])
def test_logLabelFormatter():
    fig, ax = plt.subplots()
    ax.plot([1, 5], [0.0005, 5e6])
    ax.set_yscale('log')
    ax.set_ylim([0.0001, 1e7])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(figutils.logLabelFormatter))


@image_comparison(baseline_images=['test_logLabelFormatter_alt'], extensions=['png'])
def test_logLabelFormatter_alt():
    fig, ax = plt.subplots()
    ax.plot([1, 5], [0.0005, 5e6])
    ax.set_yscale('log')
    ax.set_ylim([0.0001, 1e7])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(figutils.alt_logLabelFormatter))


@image_comparison(baseline_images=['test_connect_spines'], extensions=['png'])
def test__connect_spines():
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    figutils._connect_spines(ax1, ax2, 0.5, 0.75)
    figutils._connect_spines(ax3, ax4, 0.9, 0.1, linewidth=2, color='r' , linestyle='dashed')


@image_comparison(baseline_images=['test_parallel_coordinates'], extensions=['png'])
def test_parallel_coordinates():
    df = seaborn.load_dataset('iris')
    fig = figutils.parallel_coordinates(df, hue='species')


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
