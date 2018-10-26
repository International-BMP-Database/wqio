from distutils.version import LooseVersion
from textwrap import dedent
from io import StringIO

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest
from wqio.tests import helpers

import numpy
import scipy
import pandas

from wqio.features import (
    Location,
    Dataset,
)


OLD_SCIPY = LooseVersion(scipy.version.version) < LooseVersion('0.19')


@pytest.fixture(params=[True, False])
def location(request):
    data = helpers.getTestROSData()
    return Location(data, station_type='inflow', bsiter=1500,
                    rescol='res', qualcol='qual', useros=request.param)


@pytest.mark.parametrize('attr', [
    'name',
    'station_type',
    'analysis_space',
    'rescol',
    'qualcol',
    'plot_marker',
    'scatter_marker',
    'hasData',
    'all_positive',
    'include',
    'exclude',
    'NUnique',
    'bsiter',
    'N',
    'ND',
])
def test_locations_strings_ints(location, attr):
    expected = {
        'name': 'Influent',
        'station_type': 'inflow',
        'analysis_space': 'lognormal',
        'rescol': 'res',
        'qualcol': 'qual',
        'plot_marker': 'o',
        'scatter_marker': 'v',
        'hasData': True,
        'all_positive': True,
        'exclude': False,
        'include': True,
        'NUnique': 30,
        'bsiter': 1500,
        'N': 35,
        'ND': 7,
    }
    assert getattr(location, attr) == expected[attr]


@pytest.mark.parametrize('attr', [
    'fractionND',
    'min',
    'min_DL',
    'min_detect',
    'max',
])
def test_locations_numbers(location, attr):
    expected = {
        'fractionND': 0.2,
        'min': 2.0,
        'min_DL': 5.0,
        'min_detect': 2.0,
        'max': 22.97,
    }
    nptest.assert_approx_equal(getattr(location, attr), expected[attr])


@pytest.mark.parametrize('attr', [
    'useros',
    'cov',
    'geomean',
    'geostd',
    'logmean',
    'logstd',
    'mean',
    'median',
    'pctl10',
    'pctl25',
    'pctl75',
    'pctl90',
    'pnorm',
    'plognorm',
    'skew',
    'std',
])
@helpers.seed
def test_location_stats_scalars(location, attr):
    expected = {
        'useros': {True: True, False: False},
        'cov': {True: 0.5887644, False: 0.5280314},
        'geomean': {True: 8.0779865, False: 8.8140731},
        'geostd': {True: 1.8116975, False: 1.7094616},
        'logmean': {True: 2.0891426, False: 2.1763497},
        'logstd': {True: 0.5942642, False: 0.5361785},
        'mean': {True: 9.5888515, False: 10.120571},
        'median': {True: 7.5000000, False: 8.7100000},
        'pctl10': {True: 4.0460279, False: 5.0000000},
        'pctl25': {True: 5.6150000, False: 5.8050000},
        'pctl75': {True: 11.725000, False: 11.725000},
        'pctl90': {True: 19.178000, False: 19.178000},
        'pnorm': {True: 0.0017891, False: 0.0032362},
        'plognorm': {True: 0.5209492, False: 0.3064355},
        'skew': {True: 0.8692107, False: 0.8537566},
        'std': {True: 5.6455746, False: 5.3439797},
    }
    nptest.assert_approx_equal(
        getattr(location, attr),
        expected[attr][location.useros],
        significant=5
    )


@pytest.mark.parametrize('attr', [
    'geomean_conf_interval',
    'logmean_conf_interval',
    'mean_conf_interval',
    'median_conf_interval',
    'shapiro',
    'shapiro_log',
    'lilliefors',
    'lilliefors_log',
    'color',
])
def test_location_stats_arrays(location, attr):
    expected = {
        'color': {True: [0.32157, 0.45271, 0.66667], False: [0.32157, 0.45271, 0.66667]},
        'geomean_conf_interval': {True: [6.55572, 9.79677], False: [7.25255, 10.34346]},
        'logmean_conf_interval': {True: [1.88631, 2.27656], False: [1.97075, 2.34456]},
        'mean_conf_interval': {True: [7.74564, 11.49393], False: [8.52743, 11.97627]},
        'median_conf_interval': {True: [5.66000, 8.71000], False: [6.65000, 9.850000]},
        'shapiro': {True: [0.886889, 0.001789], False: [0.896744, 0.003236]},
        'shapiro_log': {True: [0.972679, 0.520949], False: [0.964298, 0.306435]},
        'lilliefors': {True: [0.185180, 0.003756], False: [0.160353, 0.023078]},
        'lilliefors_log': {True: [0.091855, 0.2], False: [0.08148, 0.2]},
    }
    nptest.assert_array_almost_equal(
        getattr(location, attr),
        expected[attr][location.useros],
        decimal=5
    )


@pytest.mark.parametrize('attr', ['anderson', 'anderson_log'])
@pytest.mark.parametrize('index', range(4))
def test_location_anderson(location, attr, index):
    expected = {
        'anderson': {
            True: (
                1.54388800,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15., 10., 5., 2.5, 1.],
                0.000438139
            ),
            False: (
                1.4392085,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15., 10., 5., 2.5, 1.],
                0.00080268
            )
        },
        'anderson_log': {
            True: (
                0.30409634,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15., 10., 5., 2.5, 1.],
                0.552806894
            ),
            False: (
                0.3684061,
                [0.527, 0.6, 0.719, 0.839, 0.998],
                [15., 10., 5., 2.5, 1.],
                0.41004028
            )
        }
    }
    result = expected[attr][location.useros][index]
    if index in [0, 3]:
        nptest.assert_approx_equal(
            getattr(location, attr)[index],
            result,
            significant=5
        )
    else:
        nptest.assert_array_almost_equal(
            getattr(location, attr)[index],
            result,
            decimal=5
        )


class Test_Dataset(object):
    def setup(self):
        self.maxDiff = None

        # basic test data
        self.tolerance = 0.05
        self.known_bsiter = 750

        in_data = helpers.getTestROSData()
        in_data['res'] += 3

        out_data = helpers.getTestROSData()
        out_data['res'] -= 1.5

        self.influent = Location(in_data, station_type='inflow',
                                 bsiter=self.known_bsiter, rescol='res',
                                 qualcol='qual', useros=False)

        self.effluent = Location(out_data, station_type='outflow',
                                 bsiter=self.known_bsiter, rescol='res',
                                 qualcol='qual', useros=False)

        self.ds = Dataset(self.influent, self.effluent)

        self.known_dumpFile = None
        self.known_kendall_stats = (1.00, 5.482137e-17)
        self.known_kendall_tau = self.known_kendall_stats[0]
        self.known_kendall_p = self.known_kendall_stats[1]

        self.known_mannwhitney_stats = (927.0, 2.251523e-04)
        self.known_mannwhitney_u = self.known_mannwhitney_stats[0]
        self.known_mannwhitney_p = self.known_mannwhitney_stats[1]

        self.known_spearman_stats = (1.0, 0.0)
        self.known_spearman_rho = self.known_spearman_stats[0]
        self.known_spearman_p = self.known_spearman_stats[1]

        self.known_theil_stats = (1.0, -4.5, 1.0, 1.0)
        self.known_theil_hislope = self.known_theil_stats[0]
        self.known_theil_intercept = self.known_theil_stats[1]
        self.known_theil_loslope = self.known_theil_stats[2]
        self.known_theil_medslope = self.known_theil_stats[3]

        self.known_wilcoxon_stats = (0.0, 2.4690274207037342e-07)
        self.known_wilcoxon_z = self.known_wilcoxon_stats[0]
        self.known_wilcoxon_p = self.known_wilcoxon_stats[1]

        self.known__non_paired_stats = True
        self.known__paired_stats = True
        self.known_definition = {'attr1': 'test1', 'attr2': 'test2'}
        self.known_include = True
        self.known_exclude = not self.known_include

        self.known_medianCIsOverlap = False

    def test_data(self):
        assert hasattr(self.ds, 'data')
        assert isinstance(self.ds.data, pandas.DataFrame)

    def test_paired_data(self):
        assert hasattr(self.ds, 'paired_data')
        assert isinstance(self.ds.paired_data, pandas.DataFrame)

    def test__non_paired_stats(self):
        assert hasattr(self.ds, '_non_paired_stats')
        assert self.ds._non_paired_stats, self.known__non_paired_stats

    def test__paired_stats(self):
        assert hasattr(self.ds, '_paired_stats')
        assert self.ds._paired_stats, self.known__paired_stats

    def test_name(self):
        assert hasattr(self.ds, 'name')
        assert self.ds.name is None

    def test_name_set(self):
        assert hasattr(self.ds, 'name')
        testname = 'Test Name'
        self.ds.name = testname
        assert self.ds.name == testname

    def test_defintion_default(self):
        assert hasattr(self.ds, 'definition')
        assert self.ds.definition == {}

    def test_defintion_set(self):
        assert hasattr(self.ds, 'definition')
        self.ds.definition = self.known_definition
        assert self.ds.definition == self.known_definition

    def test_include(self):
        assert hasattr(self.ds, 'include')
        assert self.ds.include == self.known_include

    def test_exclude(self):
        assert hasattr(self.ds, 'exclude')
        assert self.ds.exclude == self.known_exclude

    def test_wilcoxon_z(self):
        assert hasattr(self.ds, 'wilcoxon_z')
        nptest.assert_allclose(self.ds.wilcoxon_z, self.known_wilcoxon_z, rtol=self.tolerance)

    def test_wilcoxon_p(self):
        assert hasattr(self.ds, 'wilcoxon_p')
        nptest.assert_allclose(self.ds.wilcoxon_p, self.known_wilcoxon_p, rtol=self.tolerance)

    def test_mannwhitney_u(self):
        assert hasattr(self.ds, 'mannwhitney_u')
        nptest.assert_allclose(self.ds.mannwhitney_u, self.known_mannwhitney_u, rtol=self.tolerance)

    def test_mannwhitney_p(self):
        assert hasattr(self.ds, 'mannwhitney_p')
        nptest.assert_allclose(self.ds.mannwhitney_p, self.known_mannwhitney_p, rtol=self.tolerance)

    @pytest.mark.xfail(OLD_SCIPY, reason='Scipy < 0.19')
    def test_kendall_tau(self):
        assert hasattr(self.ds, 'kendall_tau')
        nptest.assert_allclose(self.ds.kendall_tau, self.known_kendall_tau, rtol=self.tolerance)

    @pytest.mark.xfail(OLD_SCIPY, reason='Scipy < 0.19')
    def test_kendall_p(self):
        assert hasattr(self.ds, 'kendall_p')
        nptest.assert_allclose(self.ds.kendall_p, self.known_kendall_p, rtol=self.tolerance)

    def test_spearman_rho(self):
        assert hasattr(self.ds, 'spearman_rho')
        nptest.assert_allclose(self.ds.spearman_rho, self.known_spearman_rho, atol=0.0001)

    def test_spearman_p(self):
        assert hasattr(self.ds, 'spearman_p')
        nptest.assert_allclose(self.ds.spearman_p, self.known_spearman_p, atol=0.0001)

    def test_theil_medslope(self):
        assert hasattr(self.ds, 'theil_medslope')
        nptest.assert_allclose(self.ds.theil_medslope, self.known_theil_medslope, rtol=self.tolerance)

    def test_theil_intercept(self):
        assert hasattr(self.ds, 'theil_intercept')
        nptest.assert_allclose(self.ds.theil_intercept, self.known_theil_intercept, rtol=self.tolerance)

    def test_theil_loslope(self):
        assert hasattr(self.ds, 'theil_loslope')
        nptest.assert_allclose(self.ds.theil_loslope, self.known_theil_loslope, rtol=self.tolerance)

    def test_theil_hilope(self):
        assert hasattr(self.ds, 'theil_hislope')
        nptest.assert_allclose(self.ds.theil_hislope, self.known_theil_hislope, rtol=self.tolerance)

    def test_wilcoxon_stats(self):
        assert hasattr(self.ds, '_wilcoxon_stats')
        nptest.assert_allclose(self.ds._wilcoxon_stats, self.known_wilcoxon_stats, rtol=self.tolerance)

    def test_mannwhitney_stats(self):
        assert hasattr(self.ds, '_mannwhitney_stats')
        nptest.assert_allclose(self.ds._mannwhitney_stats, self.known_mannwhitney_stats, rtol=self.tolerance)

    @pytest.mark.xfail(OLD_SCIPY, reason='Scipy < 0.19')
    def test_kendall_stats(self):
        assert hasattr(self.ds, '_kendall_stats')
        nptest.assert_allclose(self.ds._kendall_stats, self.known_kendall_stats, rtol=self.tolerance)

    def test_spearman_stats(self):
        assert hasattr(self.ds, '_spearman_stats')
        nptest.assert_allclose(self.ds._spearman_stats, self.known_spearman_stats, atol=0.0001)

    def test_theil_stats(self):
        assert hasattr(self.ds, '_theil_stats')
        nptest.assert_almost_equal(self.ds._theil_stats['medslope'],
                                   self.known_theil_stats[0],
                                   decimal=4)

        nptest.assert_almost_equal(self.ds._theil_stats['intercept'],
                                   self.known_theil_stats[1],
                                   decimal=4)

        nptest.assert_almost_equal(self.ds._theil_stats['loslope'],
                                   self.known_theil_stats[2],
                                   decimal=4)

        nptest.assert_almost_equal(self.ds._theil_stats['hislope'],
                                   self.known_theil_stats[3],
                                   decimal=4)

        assert not self.ds._theil_stats['is_inverted']

        assert 'estimated_effluent' in list(self.ds._theil_stats.keys())
        assert 'estimate_error' in list(self.ds._theil_stats.keys())

    def test_medianCIsOverlap(self):
        assert self.known_medianCIsOverlap == self.ds.medianCIsOverlap

    def test__repr__normal(self):
        self.ds.__repr__

    def test_repr__None(self):
        self.ds.definition = None
        self.ds.__repr__
