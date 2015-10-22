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
from wqio.algo import robustros


@nt.nottest
def setup_plot_data():
    plt.rcdefaults()
    data = np.array([
         3.113,   3.606,   4.046,   4.046,   4.71 ,   6.14 ,   6.978,
         2.   ,   4.2  ,   4.62 ,   5.57 ,   5.66 ,   5.86 ,   6.65 ,
         6.78 ,   6.79 ,   7.5  ,   7.5  ,   7.5  ,   8.63 ,   8.71 ,
         8.99 ,   9.85 ,  10.82 ,  11.25 ,  11.25 ,  12.2  ,  14.92 ,
        16.77 ,  17.81 ,  19.16 ,  19.19 ,  19.64 ,  20.18 ,  22.97
    ])
    return data


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
    data = setup_plot_data()
    fig, ax = plt.subplots()
    ax.plot(data)
    figutils.gridlines(ax, 'xlabel', 'ylabel')


@image_comparison(baseline_images=['test_gridlines_ylog'], extensions=['png'])
def test_gridlines_ylog():
    data = setup_plot_data()
    fig, ax = plt.subplots()
    ax.plot(data)
    figutils.gridlines(ax, 'xlabel', 'ylabel', yscale='log')


@image_comparison(baseline_images=['test_gridlines_ylog_noyminor'], extensions=['png'])
def test_gridlines_ylog_noyminor():
    data = setup_plot_data()
    fig, ax = plt.subplots()
    ax.plot(data)
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


class base_whiskers_and_fliersMixin(object):
    def teardown(self):
        pass

    def setup(self):
        self.base_setup()
        self.decimal = 3
        self.data = [
            2.20e-01, 2.70e-01, 3.08e-01, 3.20e-01, 4.10e-01, 4.44e-01,
            4.82e-01, 5.46e-01, 6.05e-01, 6.61e-01, 7.16e-01, 7.70e-01,
            8.24e-01, 1.00e-03, 4.90e-02, 5.60e-02, 1.40e-01, 1.69e-01,
            1.83e-01, 2.06e-01, 2.10e-01, 2.13e-01, 2.86e-01, 3.16e-01,
            3.40e-01, 3.57e-01, 3.71e-01, 3.72e-01, 3.78e-01, 3.81e-01,
            3.86e-01, 3.89e-01, 3.90e-01, 3.93e-01, 4.00e-01, 4.03e-01,
            4.10e-01, 4.10e-01, 4.29e-01, 4.40e-01, 4.40e-01, 4.40e-01,
            4.46e-01, 4.46e-01, 4.50e-01, 4.51e-01, 4.52e-01, 4.56e-01,
            4.60e-01, 4.66e-01, 4.72e-01, 4.78e-01, 4.81e-01, 4.83e-01,
            4.86e-01, 4.89e-01, 4.98e-01, 5.00e-01, 5.00e-01, 5.03e-01,
            5.18e-01, 5.32e-01, 5.37e-01, 5.38e-01, 5.68e-01, 5.69e-01,
            5.78e-01, 5.88e-01, 5.94e-01, 5.96e-01, 6.02e-01, 6.10e-01,
            6.10e-01, 6.10e-01, 6.19e-01, 6.20e-01, 6.20e-01, 6.28e-01,
            6.38e-01, 6.39e-01, 6.42e-01, 6.61e-01, 6.71e-01, 6.75e-01,
            6.80e-01, 6.96e-01, 7.00e-01, 7.01e-01, 7.09e-01, 7.16e-01,
            7.17e-01, 7.30e-01, 7.62e-01, 7.64e-01, 7.69e-01, 7.70e-01,
            7.77e-01, 7.80e-01, 8.06e-01, 8.10e-01, 8.23e-01, 8.30e-01,
            8.50e-01, 8.50e-01, 8.56e-01, 8.56e-01, 8.80e-01, 8.80e-01,
            8.93e-01, 8.96e-01, 8.97e-01, 8.99e-01, 9.22e-01, 9.28e-01,
            9.30e-01, 9.64e-01, 9.65e-01, 9.76e-01, 9.79e-01, 9.90e-01,
            9.99e-01, 1.00e+00, 1.00e+00, 1.01e+00, 1.02e+00, 1.03e+00,
            1.03e+00, 1.03e+00, 1.04e+00, 1.05e+00, 1.05e+00, 1.05e+00,
            1.06e+00, 1.07e+00, 1.08e+00, 1.08e+00, 1.10e+00, 1.10e+00,
            1.11e+00, 1.12e+00, 1.12e+00, 1.13e+00, 1.14e+00, 1.14e+00,
            1.14e+00, 1.15e+00, 1.16e+00, 1.17e+00, 1.17e+00, 1.17e+00,
            1.19e+00, 1.19e+00, 1.20e+00, 1.20e+00, 1.21e+00, 1.22e+00,
            1.22e+00, 1.23e+00, 1.23e+00, 1.23e+00, 1.25e+00, 1.25e+00,
            1.26e+00, 1.26e+00, 1.27e+00, 1.27e+00, 1.28e+00, 1.29e+00,
            1.29e+00, 1.30e+00, 1.30e+00, 1.30e+00, 1.31e+00, 1.31e+00,
            1.31e+00, 1.32e+00, 1.33e+00, 1.34e+00, 1.35e+00, 1.35e+00,
            1.35e+00, 1.36e+00, 1.36e+00, 1.36e+00, 1.36e+00, 1.37e+00,
            1.38e+00, 1.39e+00, 1.39e+00, 1.40e+00, 1.41e+00, 1.43e+00,
            1.44e+00, 1.44e+00, 1.47e+00, 1.47e+00, 1.48e+00, 1.51e+00,
            1.51e+00, 1.53e+00, 1.55e+00, 1.55e+00, 1.55e+00, 1.57e+00,
            1.57e+00, 1.57e+00, 1.59e+00, 1.59e+00, 1.60e+00, 1.60e+00,
            1.61e+00, 1.62e+00, 1.62e+00, 1.62e+00, 1.62e+00, 1.63e+00,
            1.63e+00, 1.63e+00, 1.64e+00, 1.66e+00, 1.68e+00, 1.68e+00,
            1.68e+00, 1.68e+00, 1.70e+00, 1.70e+00, 1.71e+00, 1.71e+00,
            1.71e+00, 1.74e+00, 1.75e+00, 1.75e+00, 1.75e+00, 1.76e+00,
            1.76e+00, 1.77e+00, 1.77e+00, 1.77e+00, 1.78e+00, 1.78e+00,
            1.79e+00, 1.79e+00, 1.80e+00, 1.81e+00, 1.81e+00, 1.82e+00,
            1.82e+00, 1.82e+00, 1.83e+00, 1.85e+00, 1.85e+00, 1.85e+00,
            1.85e+00, 1.86e+00, 1.86e+00, 1.86e+00, 1.86e+00, 1.87e+00,
            1.87e+00, 1.89e+00, 1.90e+00, 1.91e+00, 1.92e+00, 1.92e+00,
            1.92e+00, 1.94e+00, 1.95e+00, 1.95e+00, 1.95e+00, 1.96e+00,
            1.96e+00, 1.97e+00, 1.97e+00, 1.97e+00, 1.97e+00, 1.98e+00,
            1.99e+00, 1.99e+00, 1.99e+00, 2.00e+00, 2.00e+00, 2.00e+00,
            2.01e+00, 2.01e+00, 2.01e+00, 2.02e+00, 2.04e+00, 2.05e+00,
            2.06e+00, 2.06e+00, 2.06e+00, 2.07e+00, 2.08e+00, 2.09e+00,
            2.09e+00, 2.10e+00, 2.10e+00, 2.11e+00, 2.11e+00, 2.12e+00,
            2.12e+00, 2.12e+00, 2.13e+00, 2.13e+00, 2.13e+00, 2.14e+00,
            2.14e+00, 2.14e+00, 2.14e+00, 2.14e+00, 2.15e+00, 2.16e+00,
            2.17e+00, 2.18e+00, 2.18e+00, 2.18e+00, 2.19e+00, 2.19e+00,
            2.19e+00, 2.19e+00, 2.19e+00, 2.21e+00, 2.23e+00, 2.23e+00,
            2.23e+00, 2.25e+00, 2.25e+00, 2.25e+00, 2.25e+00, 2.26e+00,
            2.26e+00, 2.26e+00, 2.26e+00, 2.26e+00, 2.27e+00, 2.27e+00,
            2.28e+00, 2.28e+00, 2.28e+00, 2.29e+00, 2.29e+00, 2.29e+00,
            2.30e+00, 2.31e+00, 2.32e+00, 2.33e+00, 2.33e+00, 2.33e+00,
            2.33e+00, 2.34e+00, 2.36e+00, 2.38e+00, 2.38e+00, 2.39e+00,
            2.39e+00, 2.39e+00, 2.41e+00, 2.42e+00, 2.43e+00, 2.45e+00,
            2.45e+00, 2.47e+00, 2.48e+00, 2.49e+00, 2.49e+00, 2.49e+00,
            2.50e+00, 2.51e+00, 2.51e+00, 2.52e+00, 2.53e+00, 2.53e+00,
            2.54e+00, 2.54e+00, 2.56e+00, 2.58e+00, 2.59e+00, 2.59e+00,
            2.60e+00, 2.61e+00, 2.61e+00, 2.61e+00, 2.62e+00, 2.62e+00,
            2.63e+00, 2.65e+00, 2.65e+00, 2.66e+00, 2.66e+00, 2.68e+00,
            2.69e+00, 2.69e+00, 2.70e+00, 2.72e+00, 2.72e+00, 2.73e+00,
            2.75e+00, 2.77e+00, 2.78e+00, 2.79e+00, 2.81e+00, 2.81e+00,
            2.82e+00, 2.84e+00, 2.84e+00, 2.85e+00, 2.85e+00, 2.86e+00,
            2.86e+00, 2.88e+00, 2.92e+00, 2.93e+00, 2.93e+00, 2.95e+00,
            2.96e+00, 2.96e+00, 2.99e+00, 3.00e+00, 3.01e+00, 3.02e+00,
            3.03e+00, 3.03e+00, 3.14e+00, 3.15e+00, 3.16e+00, 3.17e+00,
            3.17e+00, 3.18e+00, 3.18e+00, 3.19e+00, 3.20e+00, 3.22e+00,
            3.24e+00, 3.25e+00, 3.29e+00, 3.31e+00, 3.32e+00, 3.32e+00,
            3.34e+00, 3.35e+00, 3.36e+00, 3.38e+00, 3.44e+00, 3.45e+00,
            3.46e+00, 3.48e+00, 3.49e+00, 3.53e+00, 3.59e+00, 3.63e+00,
            3.70e+00, 3.70e+00, 3.76e+00, 3.80e+00, 3.80e+00, 3.80e+00,
            3.83e+00, 3.84e+00, 3.88e+00, 3.90e+00, 3.91e+00, 3.96e+00,
            3.97e+00, 3.97e+00, 4.02e+00, 4.03e+00, 4.06e+00, 4.12e+00,
            4.19e+00, 4.21e+00, 4.53e+00, 4.56e+00, 4.61e+00, 4.62e+00,
            4.73e+00, 5.13e+00, 5.21e+00, 5.40e+00, 5.98e+00, 6.12e+00,
            6.94e+00, 7.38e+00, 7.56e+00, 8.06e+00, 1.38e+01, 1.51e+01,
            1.82e+01
        ]
        self.q1 = np.percentile(self.data, 25)
        self.q3 = np.percentile(self.data, 75)


        self.bs = figutils.whiskers_and_fliers(
            self.transformin(self.data),
            self.transformin(self.q1),
            self.transformin(self.q3),
            transformout=self.transformout
        )

    def test_hi_whiskers(self):
        nptest.assert_almost_equal(
            self.known_bs['whishi'],
            self.bs['whishi'],
            decimal=self.decimal
        )

    def test_hi_whiskers(self):
        nptest.assert_almost_equal(
            self.known_bs['whislo'],
            self.bs['whislo'],
            decimal=self.decimal
        )

    def test_fliers(self):
        nptest.assert_array_almost_equal(
            self.known_bs['fliers'],
            self.bs['fliers'],
            decimal=self.decimal
        )


class test_whiskers_and_fliers_ari(base_whiskers_and_fliersMixin):
    @nt.nottest
    def base_setup(self):
        self.known_bs = {
            'whishi': 4.7300000190734863,
            'fliers': np.array([
                4.730,   5.130,   5.210,   5.400,
                5.980,   6.120,   6.940,   7.380,
                7.560,   8.060,  13.800,  15.100,
               18.200
            ]),
            'whislo': 0.0011100000143051147
        }
        self.transformin = lambda x: x
        self.transformout = lambda x: x


class test_whiskers_and_fliers_natlog(base_whiskers_and_fliersMixin):
    @nt.nottest
    def base_setup(self):
        self.known_bs = {
            'whishi': 8.0606803894042987,
            'fliers': np.array([
                2.200e-01, 1.000e-03, 4.900e-02, 5.600e-02, 1.400e-01,
                1.690e-01, 1.830e-01, 2.060e-01, 2.100e-01, 2.130e-01,
                1.380e+01, 1.510e+01, 1.820e+01
            ]),
            'whislo': 0.27031713639148325
        }
        self.transformin = lambda x: np.log(x)
        self.transformout = lambda x: np.exp(x)


class test_whiskers_and_fliers_log10(base_whiskers_and_fliersMixin):
    @nt.nottest
    def base_setup(self):
        self.known_bs = {
            'whishi': 0.3741651859057793,
            'fliers': np.array([
                2.200e-01, 1.000e-03, 4.900e-02, 5.600e-02, 1.400e-01,
                1.690e-01, 1.830e-01, 2.060e-01, 2.100e-01, 2.130e-01,
                1.380e+01, 1.510e+01, 1.820e+01
            ]),
            'whislo':  0.27000000000000002
        }
        self.transformin = lambda x: np.log10(x)
        self.transformout = lambda x: 10**x



def test__boxplot_legend():
    fig, ax = plt.subplots()
    figutils.boxplot_legend(ax, notch=True)


@image_comparison(baseline_images=['test_boxplot_basic'], extensions=['png'])
def test_boxplot_basic():
    data = setup_plot_data()
    fig, ax = plt.subplots()
    figutils.boxplot(ax, data, 1)


@image_comparison(baseline_images=['test_boxplot_manual_median'], extensions=['png'])
def test_boxplot_manual_median():
    data = setup_plot_data()
    fig, ax = plt.subplots()
    figutils.boxplot(ax, data, 1, median=7.75, CIs=[7.5, 8.25])


@image_comparison(baseline_images=['test_boxplot_with_mean'], extensions=['png'])
def test_boxplot_with_mean():
    data = setup_plot_data()
    fig, ax = plt.subplots()
    figutils.boxplot(ax, data, 1, median=7.75, CIs=[7.5, 8.25],
                     mean=7.25, meanmarker='^', meancolor='red')


@image_comparison(baseline_images=['test_boxplot_formatted'], extensions=['png'])
def test_boxplot_formatted():
    data = setup_plot_data()
    fig, ax = plt.subplots()
    bp = figutils.boxplot(ax, data, 1, median=7.75, CIs=[7.5, 8.25],
                          mean=7.25, meanmarker='^', meancolor='red')
    figutils.formatBoxplot(bp)


@image_comparison(baseline_images=['test_probplot_prob'], extensions=['png'])
def test_probplot_prob():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = figutils.probplot(data, ax=ax, xlabel='Test xlabel')
    nt.assert_true(isinstance(fig, plt.Figure))


@image_comparison(baseline_images=['test_probplot_qq'], extensions=['png'])
def test_probplot_qq():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = figutils.probplot(data, ax=ax, axtype='qq', ylabel='Test label',
                            scatter_kws=dict(color='r'))


@image_comparison(baseline_images=['test_probplot_pp'], extensions=['png'])
def test_probplot_pp():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    scatter_kws = dict(color='b', linestyle='--', markeredgecolor='g', markerfacecolor='none')
    fig = figutils.probplot(data, ax=ax, axtype='pp', yscale='linear',
                            xlabel='test x', ylabel='test y', scatter_kws=scatter_kws)


@image_comparison(baseline_images=['test_probplot_prob_bestfit'], extensions=['png'])
def test_probplot_prob_bestfit():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = figutils.probplot(data, ax=ax, xlabel='Test xlabel',
                            bestfit=True)
    nt.assert_true(isinstance(fig, plt.Figure))


@image_comparison(baseline_images=['test_probplot_qq_bestfit'], extensions=['png'])
def test_probplot_qq_bestfit():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = figutils.probplot(data, ax=ax, axtype='qq',
                            bestfit=True, ylabel='Test label')


@image_comparison(baseline_images=['test_probplot_pp_bestfit'], extensions=['png'])
def test_probplot_pp_bestfit():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    scatter_kws = {'marker': 's', 'color': 'red'}
    line_kws = {'linestyle': '--', 'linewidth': 3}
    fig = figutils.probplot(data, ax=ax, axtype='pp', yscale='linear',
                            xlabel='test x', bestfit=True, ylabel='test y',
                            scatter_kws=scatter_kws, line_kws=line_kws)


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


@image_comparison(baseline_images=['test_connect_spines'], extensions=['png'], remove_text=True)
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
