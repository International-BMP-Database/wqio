import pytest
import numpy.testing as nptest

import numpy
import pandas
from matplotlib import pyplot
import seaborn

from wqio import viz
from wqio import utils
from wqio.tests import helpers


BASELINE_IMAGES = '_baseline_images/viz_tests'


@pytest.fixture
def plot_data():
    data = numpy.array([
        3.113, 3.606, 4.046, 4.046, 4.710, 6.140, 6.978,
        2.000, 4.200, 4.620, 5.570, 5.660, 5.860, 6.650,
        6.780, 6.790, 7.500, 7.500, 7.500, 8.630, 8.710,
        8.990, 9.850, 10.820, 11.250, 11.250, 12.200, 14.920,
        16.770, 17.810, 19.160, 19.190, 19.640, 20.180, 22.970
    ])
    return data


@pytest.fixture
def boxplot_data(plot_data):
    bs = [{
        'cihi': 10.82,
        'cilo': 6.1399999999999997,
        'fliers': numpy.array([22.97]),
        'iqr': 6.1099999999999994,
        'mean': 9.5888285714285733,
        'med': 7.5,
        'q1': 5.6150000000000002,
        'q3': 11.725,
        'whishi': 20.18,
        'whislo': 2.0
    }]
    return bs


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_rotateTickLabels_xaxis():
    fig, ax = pyplot.subplots()
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['AAA', 'BBB', 'CCC'])
    viz.rotateTickLabels(ax, 60, 'x')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_rotateTickLabels_yaxis():
    fig, ax = pyplot.subplots()
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['AAA', 'BBB', 'CCC'])
    viz.rotateTickLabels(ax, -30, 'y')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_rotateTickLabels_both():
    fig, ax = pyplot.subplots()
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['XXX', 'YYY', 'ZZZ'])

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['AAA', 'BBB', 'CCC'])
    viz.rotateTickLabels(ax, 45, 'both')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_log_formatter():
    fig, ax = pyplot.subplots()
    ax.plot([1, 5], [0.0005, 5e6])
    ax.set_yscale('log')
    ax.set_ylim([0.0001, 1e7])
    ax.yaxis.set_major_formatter(viz.log_formatter())
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_log_formatter_alt():
    fig, ax = pyplot.subplots()
    ax.plot([1, 5], [0.0005, 5e6])
    ax.set_yscale('log')
    ax.set_ylim([0.0001, 1e7])
    ax.yaxis.set_major_formatter(viz.log_formatter(use_1x=False))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_log_formatter_alt_2():
    fig, (ax1, ax2) = pyplot.subplots(ncols=2)
    ax1.set_yscale('log')
    ax1.set_ylim((1e-4, 1e5))
    ax1.yaxis.set_major_formatter(viz.log_formatter(threshold=4))

    ax2.set_yscale('log')
    ax2.set_ylim((1e-7, 1e5))
    ax2.yaxis.set_major_formatter(viz.log_formatter(threshold=4, use_1x=False))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_gridlines_basic(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    viz.gridlines(ax, 'xlabel', 'ylabel')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_gridlines_ylog(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    viz.gridlines(ax, 'xlabel', 'ylabel', yscale='log')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_gridlines_ylog_noyminor(plot_data):
    fig, ax = pyplot.subplots()
    ax.plot(plot_data)
    viz.gridlines(ax, 'xlabel', 'ylabel', yscale='log', yminor=False)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_one2one():
    fig, ax = pyplot.subplots()
    ax.set_xlim([-2, 5])
    ax.set_ylim([-3, 3])
    viz.one2one(ax, label='Equality', lw=5, ls='--')
    ax.legend()
    return fig


@pytest.fixture
@helpers.seed
def jp_data():
    pyplot.rcdefaults()
    N = 37
    df = pandas.DataFrame({
        'A': numpy.random.normal(size=N),
        'B': numpy.random.lognormal(mean=0.25, sigma=1.25, size=N),
        'C': numpy.random.lognormal(mean=1.25, sigma=0.75, size=N)
    })

    return df


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_jointplot_defaultlabels(jp_data):
    jg1 = viz.jointplot(x='B', y='C', data=jp_data, one2one=False, color='b')
    return jg1.fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_jointplot_xlabeled(jp_data):
    jg2 = viz.jointplot(x='B', y='C', data=jp_data, one2one=False, color='g',
                        xlabel='Quantity B')
    return jg2.fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_jointplot_ylabeled(jp_data):
    jg3 = viz.jointplot(x='B', y='C', data=jp_data, one2one=False, color='r',
                        ylabel='Quantity C')
    return jg3.fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_jointplot_bothlabeled(jp_data):
    jg4 = viz.jointplot(x='B', y='C', data=jp_data, one2one=False, color='k',
                        xlabel='Quantity B', ylabel='Quantity C')
    return jg4.fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_jointplot_zerominFalse(jp_data):
    jg1 = viz.jointplot(x='A', y='C', data=jp_data, zeromin=False, one2one=False)
    return jg1.fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_jointplot_one2one(jp_data):
    jg1 = viz.jointplot(x='B', y='C', data=jp_data, one2one=True)
    return jg1.fig


class base_whiskers_and_fliersMixin(object):
    def teardown(self):
        pass

    def setup(self):
        self.basic_setup()
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
        self.q1 = numpy.percentile(self.data, 25)
        self.q3 = numpy.percentile(self.data, 75)

        self.bs = viz.whiskers_and_fliers(
            self.transformin(self.data),
            self.transformin(self.q1),
            self.transformin(self.q3),
            transformout=self.transformout
        )

    def test_hi_whiskers(self):
        nptest.assert_almost_equal(
            self.bs['whishi'],
            self.known_bs['whishi'],
            decimal=self.decimal
        )

    def test_lo_whiskers(self):
        nptest.assert_almost_equal(
            self.bs['whislo'],
            self.known_bs['whislo'],
            decimal=self.decimal
        )

    def test_fliers(self):
        nptest.assert_array_almost_equal(
            self.bs['fliers'],
            self.known_bs['fliers'],
            decimal=self.decimal
        )


class Test_whiskers_and_fliers_ari(base_whiskers_and_fliersMixin):
    def basic_setup(self):
        self.known_bs = {
            'whishi': 4.62,
            'fliers': numpy.array([
                4.730, 5.130, 5.210, 5.400, 5.980, 6.120, 6.940,
                7.380, 7.560, 8.060, 13.800, 15.100, 18.200
            ]),
            'whislo': 0.0011100000143051147
        }
        self.transformin = utils.no_op
        self.transformout = utils.no_op


class Test_whiskers_and_fliers_natlog(base_whiskers_and_fliersMixin):
    def basic_setup(self):
        self.known_bs = {
            'whishi': 8.060,
            'fliers': numpy.array([
                2.200e-01, 1.000e-03, 4.900e-02, 5.600e-02, 1.400e-01,
                1.690e-01, 1.830e-01, 2.060e-01, 2.100e-01, 2.130e-01,
                1.380e+01, 1.510e+01, 1.820e+01
            ]),
            'whislo': 0.2700
        }
        self.transformin = lambda x: numpy.log(x)
        self.transformout = lambda x: numpy.exp(x)


class Test_whiskers_and_fliers_log10(base_whiskers_and_fliersMixin):
    def basic_setup(self):
        self.known_bs = {
            'whishi': 8.060,
            'fliers': numpy.array([
                2.200e-01, 1.000e-03, 4.900e-02, 5.600e-02, 1.400e-01,
                1.690e-01, 1.830e-01, 2.060e-01, 2.100e-01, 2.130e-01,
                1.380e+01, 1.510e+01, 1.820e+01
            ]),
            'whislo': 0.2700
        }
        self.transformin = lambda x: numpy.log10(x)
        self.transformout = lambda x: 10**x


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boxplot_basic(boxplot_data):
    fig, ax = pyplot.subplots()
    viz.boxplot(boxplot_data, ax=ax, shownotches=True, patch_artist=False)
    ax.set_xlim((0, 2))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boxplot_with_mean(boxplot_data):
    fig, ax = pyplot.subplots()
    viz.boxplot(boxplot_data, ax=ax, shownotches=True, patch_artist=True,
                marker='^', color='red', showmean=True)
    ax.set_xlim((0, 2))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_probplot_prob(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, xlabel='Test xlabel')
    assert isinstance(fig, pyplot.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_probplot_qq(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, axtype='qq', ylabel='Test label',
                       scatter_kws=dict(color='r'))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_probplot_pp(plot_data):
    fig, ax = pyplot.subplots()
    scatter_kws = dict(color='b', linestyle='--', markeredgecolor='g', markerfacecolor='none')
    fig = viz.probplot(plot_data, ax=ax, axtype='pp', yscale='linear',
                       xlabel='test x', ylabel='test y', scatter_kws=scatter_kws)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_probplot_prob_bestfit(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, xlabel='Test xlabel',
                       bestfit=True)
    assert isinstance(fig, pyplot.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_probplot_qq_bestfit(plot_data):
    fig, ax = pyplot.subplots()
    fig = viz.probplot(plot_data, ax=ax, axtype='qq',
                       bestfit=True, ylabel='Test label')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_probplot_pp_bestfit(plot_data):
    fig, ax = pyplot.subplots()
    scatter_kws = {'marker': 's', 'color': 'red'}
    line_kws = {'linestyle': '--', 'linewidth': 3}
    fig = viz.probplot(plot_data, ax=ax, axtype='pp', yscale='linear',
                       xlabel='test x', bestfit=True, ylabel='test y',
                       scatter_kws=scatter_kws, line_kws=line_kws)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test__connect_spines():
    fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(ncols=2, nrows=2)
    viz._connect_spines(ax1, ax2, 0.5, 0.75)
    viz._connect_spines(ax3, ax4, 0.9, 0.1, linewidth=2, color='r', linestyle='dashed')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_parallel_coordinates():
    df = seaborn.load_dataset('iris')
    fig = viz.parallel_coordinates(df, hue='species')
    return fig


@pytest.fixture
@helpers.seed
def cat_hist_data():
    N = 100
    years = [2011, 2012, 2013, 2014]
    df = pandas.DataFrame({
        'depth': numpy.random.uniform(low=0.2, high=39, size=N),
        'year': numpy.random.choice(years, size=N),
        'has_outflow': numpy.random.choice([False, True], size=N)
    })
    return df


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_categorical_histogram_simple(cat_hist_data):
    bins = numpy.arange(5, 35, 5)
    fig = viz.categorical_histogram(cat_hist_data, 'depth', bins)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_categorical_histogram_complex(cat_hist_data):
    bins = numpy.arange(5, 35, 5)
    fig = viz.categorical_histogram(cat_hist_data, 'depth', bins, hue='year', row='has_outflow')
    return fig
