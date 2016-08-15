import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest
from wqio.tests import helpers

import numpy
import pandas

from wqio.features import (
    Parameter,
    Location,
    Dataset,
    DataCollection
)


class _base_Parameter_Mixin(object):
    def setup(self):
        self.known_name = 'Copper'
        self.known_units = 'ug/L'
        self.known_pu_nocomma = 'Copper (ug/L)'
        self.known_pu_comma = 'Copper, ug/L'
        self.param = Parameter(name=self.known_name, units=self.known_units)

    def teardown(self):
        pass

    def test_name(self):
        assert self.known_name == self.param.name

    def test_units(self):
        assert self.known_units == self.param.units

    def test_paramunit_nocomma(self):
        assert self.known_pu_nocomma == self.param.paramunit(usecomma=False)

    def test_paramunit_comma(self):
        assert self.known_pu_comma == self.param.paramunit(usecomma=True)


class Test_Parameter_simple(_base_Parameter_Mixin):
    def setup(self):
        self.known_name = 'Copper'
        self.known_units = 'ug/L'
        self.known_pu_nocomma = 'Copper (ug/L)'
        self.known_pu_comma = 'Copper, ug/L'
        self.param = Parameter(name=self.known_name, units=self.known_units)


class Test_Parameter_complex(_base_Parameter_Mixin):
    #place holder for when TeX/validation/etc gets integrated
    def setup(self):
        self.known_name = 'Nitrate & Nitrite'
        self.known_units = 'mg/L'
        self.known_pu_nocomma = 'Nitrate & Nitrite (mg/L)'
        self.known_pu_comma = 'Nitrate & Nitrite, mg/L'
        self.param = Parameter(name=self.known_name, units=self.known_units)


class _base_LocationMixin(object):

    def main_setup(self):

        # basic test data
        self.tolerance = 0.05
        self.known_bsIter = 1500
        self.data = helpers.getTestROSData()

        # Location stuff
        self.known_station_name = 'Influent'
        self.known_station_type = 'inflow'
        self.known_plot_marker = 'o'
        self.known_scatter_marker = 'v'
        self.known_color = numpy.array((0.32157, 0.45271, 0.66667))
        self.known_rescol = 'res'
        self.known_qualcol = 'qual'
        self.known_all_positive = True
        self.known_include = True
        self.known_exclude = False
        self.known_hasData = True
        self.known_min_detect = 2.00
        self.known_min_DL = 5.00
        self.known_filtered_include = False

    def test_bsIter(self):
        assert hasattr(self.loc, 'bsIter')
        assert self.loc.bsIter == self.known_bsIter

    def test_useROS(self):
        assert hasattr(self.loc, 'useROS')
        assert self.loc.useROS == self.known_useRos

    def test_name(self):
        assert hasattr(self.loc, 'name')
        assert self.loc.name == self.known_station_name

    def test_N(self):
        assert hasattr(self.loc, 'N')
        assert self.loc.N == self.known_N

    def test_hasData(self):
        assert hasattr(self.loc, 'hasData')
        assert self.loc.hasData == self.known_hasData

    def test_ND(self):
        assert hasattr(self.loc, 'ND')
        assert self.loc.ND == self.known_ND

    def test_fractionND(self):
        assert hasattr(self.loc, 'fractionND')
        assert self.loc.fractionND == self.known_fractionND

    def test_NUnique(self):
        assert hasattr(self.loc, 'NUnique')
        assert self.loc.NUnique == self.known_NUnique

    def test_analysis_space(self):
        assert hasattr(self.loc, 'analysis_space')
        assert self.loc.analysis_space == self.known_analysis_space

    def test_cov(self):
        assert hasattr(self.loc, 'cov')
        nptest.assert_allclose(self.loc.cov, self.known_cov, rtol=self.tolerance)

    def test_geomean(self):
        assert hasattr(self.loc, 'geomean')
        nptest.assert_allclose(self.loc.geomean, self.known_geomean, rtol=self.tolerance)

    def test_geomean_conf_interval(self):
        assert hasattr(self.loc, 'geomean_conf_interval')
        nptest.assert_allclose(
            self.loc.geomean_conf_interval,
            self.known_geomean_conf_interval,
            rtol=self.tolerance
        )

    def test_geostd(self):
        assert hasattr(self.loc, 'geostd')
        nptest.assert_allclose(
            self.loc.geostd,
            self.known_geostd,
            rtol=self.tolerance
        )

    def test_logmean(self):
        assert hasattr(self.loc, 'logmean')
        nptest.assert_allclose(
            self.loc.logmean,
            self.known_logmean,
            rtol=self.tolerance
        )

    def test_logmean_conf_interval(self):
        assert hasattr(self.loc, 'logmean_conf_interval')
        nptest.assert_allclose(self.loc.logmean_conf_interval, self.known_logmean_conf_interval, rtol=self.tolerance)

    def test_logstd(self):
        assert hasattr(self.loc, 'logstd')
        nptest.assert_allclose(self.loc.logstd, self.known_logstd, rtol=self.tolerance)

    def test_max(self):
        assert hasattr(self.loc, 'max')
        assert self.loc.max == self.known_max

    def test_mean(self):
        assert hasattr(self.loc, 'mean')
        nptest.assert_allclose(self.loc.mean, self.known_mean, rtol=self.tolerance)

    def test_mean_conf_interval(self):
        assert hasattr(self.loc, 'mean_conf_interval')
        nptest.assert_allclose(self.loc.mean_conf_interval, self.known_mean_conf_interval, rtol=self.tolerance)

    def test_median(self):
        assert hasattr(self.loc, 'median')
        nptest.assert_allclose(self.loc.median, self.known_median, rtol=self.tolerance)

    def test_median_conf_interval(self):
        assert hasattr(self.loc, 'median_conf_interval')
        nptest.assert_allclose(self.loc.median_conf_interval, self.known_median_conf_interval, rtol=self.tolerance*1.5)

    def test_min(self):
        assert hasattr(self.loc, 'min')
        assert self.loc.min == self.known_min

    def test_min_detect(self):
        assert hasattr(self.loc, 'min_detect')
        assert self.loc.min_detect == self.known_min_detect

    def test_min_DL(self):
        assert hasattr(self.loc, 'min_DL')
        assert self.loc.min_DL == self.known_min_DL

    def test_pctl10(self):
        assert hasattr(self.loc, 'pctl10')
        nptest.assert_allclose(self.loc.pctl10, self.known_pctl10, rtol=self.tolerance)

    def test_pctl25(self):
        assert hasattr(self.loc, 'pctl25')
        nptest.assert_allclose(self.loc.pctl25, self.known_pctl25, rtol=self.tolerance)

    def test_pctl75(self):
        assert hasattr(self.loc, 'pctl75')
        nptest.assert_allclose(self.loc.pctl75, self.known_pctl75, rtol=self.tolerance)

    def test_pctl90(self):
        assert hasattr(self.loc, 'pctl90')
        nptest.assert_allclose(self.loc.pctl90, self.known_pctl90, rtol=self.tolerance)

    def test_pnorm(self):
        assert hasattr(self.loc, 'pnorm')
        nptest.assert_allclose(self.loc.pnorm, self.known_pnorm, rtol=self.tolerance)

    def test_plognorm(self):
        assert hasattr(self.loc, 'plognorm')
        nptest.assert_allclose(self.loc.plognorm, self.known_plognorm, rtol=self.tolerance)

    def test_shapiro(self):
        assert hasattr(self.loc, 'shapiro')
        nptest.assert_allclose(self.loc.shapiro, self.known_shapiro, rtol=self.tolerance)

    def test_shapiro_log(self):
        assert hasattr(self.loc, 'shapiro_log')
        nptest.assert_allclose(self.loc.shapiro_log, self.known_shapiro_log, rtol=self.tolerance)

    def test_lillifors(self):
        assert hasattr(self.loc, 'lillifors')
        nptest.assert_allclose(self.loc.lillifors, self.known_lillifors, rtol=self.tolerance)

    def test_lillifors_log(self):
        assert hasattr(self.loc, 'lillifors_log')
        nptest.assert_allclose(self.loc.lillifors_log, self.known_lillifors_log, rtol=self.tolerance)

    def test_anderson(self):
        assert hasattr(self.loc, 'anderson')

        # Ad stat
        assert abs(self.loc.anderson[0] - self.known_anderson[0]) < 0.0001

        # critical values
        nptest.assert_allclose(self.loc.anderson[1], self.known_anderson[1], rtol=self.tolerance)

        # significance level
        nptest.assert_allclose(self.loc.anderson[2], self.known_anderson[2], rtol=self.tolerance)

        # p-value
        assert abs(self.loc.anderson[3] - self.known_anderson[3]) < 0.0000001

    def test_anderson_log(self):
        assert hasattr(self.loc, 'anderson_log')

        # Ad stat
        assert abs(self.loc.anderson_log[0] - self.known_anderson_log[0]) < 0.0001

        # critical values
        nptest.assert_allclose(self.loc.anderson_log[1], self.known_anderson_log[1], rtol=self.tolerance)

        # significance level
        nptest.assert_allclose(self.loc.anderson_log[2], self.known_anderson_log[2], rtol=self.tolerance)

        # p-value
        assert abs(self.loc.anderson_log[3] - self.known_anderson_log[3]) < 0.0000001

    def test_skew(self):
        assert hasattr(self.loc, 'skew')
        nptest.assert_allclose(self.loc.skew, self.known_skew, rtol=self.tolerance)

    def test_std(self):
        assert hasattr(self.loc, 'std')
        nptest.assert_allclose(self.loc.std, self.known_std, rtol=self.tolerance)

    def test_station_name(self):
        assert hasattr(self.loc, 'station_name')
        assert self.loc.station_name == self.known_station_name

    def test_station_type(self):
        assert hasattr(self.loc, 'station_type')
        assert self.loc.station_type == self.known_station_type

    def test_symbology_plot_marker(self):
        assert hasattr(self.loc, 'plot_marker')
        assert self.loc.plot_marker == self.known_plot_marker

    def test_symbology_scatter_marker(self):
        assert hasattr(self.loc, 'scatter_marker')
        assert self.loc.scatter_marker == self.known_scatter_marker

    def test_symbology_color(self):
        assert hasattr(self.loc, 'color')
        nptest.assert_almost_equal(
            numpy.array(self.loc.color),
            self.known_color,
            decimal=4
        )

    def test_data(self):
        assert hasattr(self.loc, 'data')
        assert isinstance(self.loc.data, numpy.ndarray)

    def test_full_data(self):
        assert hasattr(self.loc, 'full_data')
        assert isinstance(self.loc.full_data, pandas.DataFrame)

    def test_full_data_columns(self):
        assert 'res' in self.data.columns
        assert 'qual' in self.data.columns

    def test_all_positive(self):
        assert hasattr(self.loc, 'all_positive')
        assert self.loc.all_positive == self.known_all_positive

    def test_include(self):
        assert hasattr(self.loc, 'include')
        assert self.loc.include == self.known_include

    def test_exclude(self):
        assert hasattr(self.loc, 'exclude')
        assert self.loc.exclude == self.known_exclude

    def test_include_setter(self):
        self.loc.include = not self.known_include
        assert self.loc.include == (not self.known_include)
        assert self.loc.exclude == (not self.known_exclude)


class Test_Location_ROS(_base_LocationMixin):
    def setup(self):
        self.main_setup()
        self.loc = Location(self.data, station_type='inflow', bsIter=self.known_bsIter,
                            rescol='res', qualcol='qual', useROS=True)

        # known statistics
        self.known_N = 35
        self.known_ND = 7
        self.known_fractionND = 0.2
        self.known_NUnique = 30
        self.known_analysis_space = 'lognormal'
        self.known_cov = 0.597430584141
        self.known_geomean = 8.08166947637
        self.known_geomean_conf_interval = [6.63043647, 9.79058872]
        self.known_geostd = 1.79152565521
        self.known_logmean = 2.08959846955
        self.known_logmean_conf_interval = [1.89167063, 2.28142159]
        self.known_logstd = 0.583067578178
        self.known_max = 22.97
        self.known_mean = 9.59372041292
        self.known_mean_conf_interval = [7.75516047, 11.45197482]
        self.known_median = 7.73851689962
        self.known_median_conf_interval = [ 5.57,  8.63]
        self.known_min = 2.0
        self.known_pctl10 = 4.04355908285
        self.known_pctl25 = 5.615
        self.known_pctl75 = 11.725
        self.known_pctl90 = 19.178
        self.known_pnorm = 0.00179254170507
        self.known_plognorm = 0.521462738514
        self.known_shapiro = [ 0.886889,  0.001789]
        self.known_shapiro_log = [ 0.972679,  0.520949]
        self.known_lillifors = [ 0.18518 ,  0.003756]
        self.known_lillifors_log = [ 0.091855,  0.635536]
        self.known_anderson = (
            1.54388800,
            [0.527, 0.6, 0.719, 0.839, 0.998],
            [ 15., 10., 5., 2.5, 1.],
            0.000438139
        )
        self.known_anderson_log = (
            0.30409634,
            [0.527, 0.6, 0.719, 0.839, 0.998],
            [ 15., 10., 5., 2.5, 1.],
            0.552806894
        )
        self.known_skew = 0.869052892573
        self.known_std = 5.52730949374
        self.known_useRos = True
        self.known_all_positive = True
        self.known_filtered_shape = (27,2)
        self.known_filtered_data_shape = (27,)


class Test_Location_noROS(_base_LocationMixin):
    def setup(self):
        self.main_setup()

        # set useROS to True, then turn it off later to make sure that everything
        # propogated correctly
        self.loc = Location(self.data, station_type='inflow', bsIter=self.known_bsIter,
                            rescol='res', qualcol='qual', useROS=True)

        # turn ROS off
        self.loc.useROS = False

        # known statistics
        self.known_N = 35
        self.known_ND = 7
        self.known_fractionND = 0.2
        self.known_NUnique = 30
        self.known_analysis_space = 'lognormal'
        self.known_cov = 0.534715517405
        self.known_geomean = 8.82772846655
        self.known_geomean_conf_interval = [7.37541563, 10.5446939]
        self.known_geostd = 1.68975502971
        self.known_logmean = 2.17789772971
        self.known_logmean_conf_interval = [1.99815226, 2.35562279]
        self.known_logstd = 0.524583565596
        self.known_max = 22.97
        self.known_mean = 10.1399679143
        self.known_mean_conf_interval = [8.3905866, 11.96703843]
        self.known_median = 8.671859
        self.known_median_conf_interval1 = [6.65, 9.85]
        self.known_median_conf_interval2 = [6.65, 10.82]
        self.known_min = 2.0
        self.known_pctl10 = 5.0
        self.known_pctl25 = 5.805
        self.known_pctl75 = 11.725
        self.known_pctl90 = 19.178
        self.known_pnorm = 0.00323620648123
        self.known_plognorm = 0.306435495615
        self.known_shapiro = [ 0.896744,  0.003236]
        self.known_shapiro_log = [ 0.964298,  0.306435]
        self.known_lillifors = [ 0.160353,  0.023078]
        self.known_lillifors_log = [ 0.08148,  0.84545]
        self.known_anderson = (
            1.4392085,
            [0.527, 0.6, 0.719, 0.839, 0.998],
            [ 15., 10., 5., 2.5, 1.],
            0.00080268
        )
        self.known_anderson_log = (
            0.3684061,
            [0.527, 0.6, 0.719, 0.839, 0.998],
            [ 15., 10., 5., 2.5, 1.],
            0.41004028
        )
        self.known_skew = 0.853756570358
        self.known_std = 5.24122841148
        self.known_useRos = False
        self.known_filtered_shape = (30,2)
        self.known_filtered_data_shape = (30,)

    def test_median_conf_interval(self):
        assert hasattr(self.loc, 'median_conf_interval')
        try:
            nptest.assert_allclose(self.loc.median_conf_interval,
                                   self.known_median_conf_interval1,
                                   rtol=self.tolerance)
        except AssertionError:
            nptest.assert_allclose(self.loc.median_conf_interval,
                                   self.known_median_conf_interval2,
                                   rtol=self.tolerance)


class Test_Dataset(object):
    def setup(self):
        self.maxDiff = None

        # basic test data
        self.tolerance = 0.05
        self.known_bsIter = 750

        in_data = helpers.getTestROSData()
        in_data['res'] += 3

        out_data = helpers.getTestROSData()
        out_data['res'] -= 1.5

        self.influent = Location(in_data, station_type='inflow', bsIter=self.known_bsIter,
                                 rescol='res', qualcol='qual', useROS=False)

        self.effluent = Location(out_data, station_type='outflow', bsIter=self.known_bsIter,
                                 rescol='res', qualcol='qual', useROS=False)

        self.ds = Dataset(self.influent, self.effluent)

        self.known_dumpFile = None
        self.known_kendall_stats = (1.00, 2.916727e-17)
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

    def test_kendall_tau(self):
        assert hasattr(self.ds, 'kendall_tau')
        nptest.assert_allclose(self.ds.kendall_tau, self.known_kendall_tau, rtol=self.tolerance)

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


class _base_DataCollecionMixin(object):
    known_rescol = 'ros_res'
    known_raw_rescol = 'res'
    known_roscol = 'ros_res'
    known_qualcol = 'qual'
    known_stationcol = 'loc'
    known_paramcol = 'param'
    known_ndval = 'ND'
    expected_comp_index = pandas.MultiIndex.from_tuples([
        ('A', 'Inflow', 'Outflow'),
        ('A', 'Inflow', 'Reference'),
        ('A', 'Outflow', 'Inflow'),
        ('A', 'Outflow', 'Reference'),
        ('A', 'Reference', 'Inflow'),
        ('A', 'Reference', 'Outflow'),
        ('B', 'Inflow', 'Outflow'),
        ('B', 'Inflow', 'Reference'),
        ('B', 'Outflow', 'Inflow'),
        ('B', 'Outflow', 'Reference'),
        ('B', 'Reference', 'Inflow'),
        ('B', 'Reference', 'Outflow'),
    ], names=['param', 'loc_1', 'loc_2'])

    def prep_comp_result(self, df):
        return df.query("param in ['A', 'B']")

    def prep_comp_expected(self, data):
        return pandas.DataFrame(data, index=self.expected_comp_index)

    def test__raw_rescol(self):
        assert self.dc._raw_rescol == self.known_raw_rescol

    def test_data(self):
        assert isinstance(self.dc.data, pandas.DataFrame)

    def test_roscol(self):
        assert self.dc.roscol == self.known_roscol

    def test_rescol(self):
        assert self.dc.rescol == self.known_rescol

    def test_qualcol(self):
        assert self.dc.qualcol == self.known_qualcol

    def test_stationncol(self):
        assert self.dc.stationcol == self.known_stationcol

    def test_paramcol(self):
        assert self.dc.paramcol == self.known_paramcol

    def test_ndval(self):
        assert self.dc.ndval == [self.known_ndval]

    def test_bsIter(self):
        assert self.dc.bsIter == self.known_bsIter

    def test_groupby(self):
        assert self.dc.groupcols == self.known_groupcols

    def test_columns(self):
        assert self.dc.columns == self.known_columns

    def test_filterfxn(self):
        assert hasattr(self.dc, 'filterfxn')

    def test_tidy(self):
        assert isinstance(self.dc.tidy, pandas.DataFrame)
        tidycols = self.dc.tidy.columns.tolist()
        knowncols = self.known_columns + [self.known_roscol]
        assert sorted(tidycols) == sorted(knowncols)

    @helpers.seed
    def test_means(self):
        pdtest.assert_frame_equal(
            numpy.round(self.dc.means, 3),
            self.known_means,
            check_names=False
        )

    @helpers.seed
    def test_medians(self):
        pdtest.assert_frame_equal(
            numpy.round(self.dc.medians, 3),
            self.known_medians,
            check_names=False
        )

    def test_shaprio(self):
        expected_data = {
            ('Inflow', 'pvalue'): [
                1.35000277e-07, 1.15147395e-06, 2.84853918e-08, 7.25922699e-09,
                4.04597494e-06, 5.37404680e-08, 6.98694420e-05, 2.09285303e-06,
            ],
            ('Outflow', 'shapiro'): [
                0.68407392, 0.74749016, 0.897427022, 0.55306959,
                0.78847879, 0.88128829, 0.778801679, 0.71769607,
            ],
            ('Reference', 'shapiro'): [
                0.64490425, 0.66652446, 0.497832655, 0.65883779,
                0.57689213, 0.39423871, 0.583670616, 0.78648996,
            ],
            ('Inflow', 'shapiro'): [
                0.59733164, 0.67138719, 0.537596583, 0.48065340,
                0.71066832, 0.56260156, 0.789419353, 0.69042921,
            ],
            ('Outflow', 'pvalue'): [
                1.71045326e-06, 1.4417389138543513e-05, 0.0099637247622013092, 4.2077829220943386e-08,
                6.73221293e-05, 0.0042873905040323734, 4.616594742401503e-05, 5.1183847062929999e-06,
            ],
            ('Reference', 'pvalue'): [
                5.18675221e-07, 9.91817046e-07, 1.08479882e-08, 7.85432064e-07,
                7.80523379e-08, 1.08831232e-09, 9.34287243e-08, 6.22547740e-05,
            ],
        }
        result = self.dc.shapiro
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_shapiro_log(self):
        expected_data = {
            ('Inflow', 'log-shapiro'): [
                0.962131, 0.983048, 0.974349, 0.971814,
                0.953440, 0.986218, 0.970588, 0.980909,
            ],
            ('Inflow', 'pvalue'): [
                0.3912280, 0.916009, 0.700492, 0.630001,
                0.2414900, 0.964459, 0.596430, 0.871599,
            ],
            ('Outflow', 'log-shapiro'): [
                0.969009, 0.915884, 0.9498569, 0.972526,
                0.973539, 0.964473, 0.9782290, 0.982024,
            ],
            ('Outflow', 'pvalue'): [
                0.554274, 0.027473, 0.196495, 0.649692,
                0.677933, 0.442456, 0.806073, 0.895776,
            ],
            ('Reference', 'log-shapiro'): [
                0.980489, 0.946262, 0.986095, 0.991098,
                0.966805, 0.958543, 0.975473, 0.971864,
            ],
            ('Reference', 'pvalue'): [
                0.861968, 0.159437, 0.963002, 0.996583,
                0.498018, 0.321844, 0.731752, 0.631380,
            ]
        }
        result = self.dc.shapiro_log
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_lillifors(self):
        expected_data = {
            ('Inflow', 'lillifors'): [
                0.27843886, 0.24802344, 0.33527872, 0.37170719,
                0.22752955, 0.28630622, 0.20595218, 0.27953043
            ],
            ('Inflow', 'pvalue'): [
                6.24148409e-06, 0.000119332, 8.62060186e-09, 6.08463945e-11,
                0.000695739, 2.72633652e-06, 0.003660092, 5.57279417e-06
            ],
            ('Outflow', 'lillifors'): [
                0.31416123, 0.23332031, 0.13066911, 0.32460209,
                0.20829954, 0.18088983, 0.25659462, 0.22520102
            ],
            ('Outflow', 'pvalue'): [
                1.17145253e-07, 0.000430609, 0.248687678, 3.30298603e-08,
                0.003085194, 0.019562376, 5.41005765e-05, 0.000840341
            ],
            ('Reference', 'lillifors'): [
                0.28351398, 0.24930868, 0.29171088, 0.22593144,
                0.32020481, 0.42722648, 0.32208719, 0.16592420
            ],
            ('Reference', 'pvalue'): [
                3.66921686e-06, 0.000106205, 1.51965475e-06, 0.000792213,
                5.66187072e-08, 1.06408145e-14, 4.49998108e-08, 0.046766975
            ],
        }
        result = self.dc.lillifors
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_lillifors_log(self):
        expected_data = {
            ('Inflow', 'log-lillifors'): [
                0.10979498, 0.08601191, 0.13721881, 0.123985937,
                0.12767141, 0.09254070, 0.11319291, 0.107527896,
            ],
            ('Inflow', 'pvalue'): [
                0.51940912, 0.95582245, 0.18986954, 0.321330151,
                0.27964797, 0.82837543, 0.46666969, 0.556327244,
            ],
            ('Outflow', 'log-lillifors'): [
                0.10661711, 0.12321271, 0.15586768, 0.144033649,
                0.09536450, 0.11033031, 0.08012353, 0.089479881,
            ],
            ('Outflow', 'pvalue'): [
                0.57153020, 0.33058849, 0.07956179, 0.140596426,
                0.77423131, 0.51088991, 1.07047546, 0.887890467,
            ],
            ('Reference', 'log-lillifors'): [
                0.08280941, 0.12379594, 0.10251285, 0.070249172,
                0.08910822, 0.14824279, 0.10622305, 0.109876879,
            ],
            ('Reference', 'pvalue'): [
                1.01845443, 0.32358845, 0.64250052, 1.251670597,
                0.89515574, 0.11562015, 0.57817185, 0.518100907,
            ],
        }
        result = self.dc.lillifors_log
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    @helpers.seed
    def test__generic_stat(self):
        result = self.dc._generic_stat(numpy.min, use_bootstrap=False)
        pdtest.assert_frame_equal(
            numpy.round(result, 3),
            self.known_genericstat,
            check_names=False
        )

    def test_mann_whitney(self):
        expected_data = {
            'pvalue': [
                0.9934626, 0.1029978, 0.9934626, 0.0701802, 0.1029978, 0.07018023,
                0.3214884, 0.0930252, 0.3214884, 0.5174506, 0.0930252, 0.51745067,
            ],
            'mann_whitney': [
                391.0, 492.0, 393.0, 503.0, 292.0, 281.0,
                453.0, 495.0, 331.0, 432.0, 289.0, 352.0,
                ]
            }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.mann_whitney),
            self.prep_comp_expected(expected_data),
        )

    def test_t_test(self):
        expected_data = {
            't_test': [
                0.5069615, 1.7827515, -0.5069615, 1.6629067, -1.7827515, -1.6629067,
                0.2994807, 0.9528661, -0.2994807, 0.7150517, -0.9528661, -0.7150517,
            ],
            'pvalue': [
                0.6145228, 0.0835157, 0.6145228, 0.1038642, 0.0835157, 0.1038642,
                0.7657606, 0.3452425, 0.7657606, 0.4776926, 0.3452425, 0.4776926,
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.t_test),
            self.prep_comp_expected(expected_data),
        )

    def test_levene(self):
        expected_data = {
            'levene': [
                0.3312645, 2.8502753, 0.3312645, 2.1520503, 2.8502753, 2.1520503,
                0.0024326, 0.2962336, 0.0024325, 0.4589038, 0.2962336, 0.4589038,
            ],
            'pvalue': [
                0.5673062, 0.0971261, 0.5673062, 0.1481797, 0.0971261, 0.1481797,
                0.9608452, 0.5884937, 0.9608452, 0.5010288, 0.5884937, 0.5010288
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.levene),
            self.prep_comp_expected(expected_data),
            check_less_precise=True,
        )

    def test__comparison_stat(self):
        result = self.dc._comparison_stat(helpers.comp_statfxn, statname='Tester')

        expected_data = {
            'pvalue': [
                8.947202, 9.003257,  6.018320,  6.032946,
                3.276959, 3.235531, 10.306480, 10.331898,
                5.552678, 5.584064,  5.795812,  5.801780,
            ],
            'Tester': [
                35.788809, 36.013028, 24.073280, 24.131785,
                13.107839, 12.942125, 41.225920, 41.327592,
                22.210713, 22.336259, 23.183249, 23.207123,
            ]
        }

        expected = pandas.DataFrame(expected_data, index=self.expected_comp_index)
        pdtest.assert_frame_equal(
            self.prep_comp_result(result),
            expected
        )

    def test_locations(self):
        for l in self.dc.locations:
            assert isinstance(l, Location)
        assert len(self.dc.locations) == 24
        assert self.dc.locations[0].definition == {'loc': 'Inflow', 'param': 'A'}
        assert self.dc.locations[1].definition == {'loc': 'Inflow', 'param': 'B'}
        assert self.dc.locations[6].definition == {'loc': 'Inflow', 'param': 'G'}
        assert self.dc.locations[8].definition == {'loc': 'Outflow', 'param': 'A'}

    def test_datasets(self):
        for d in self.dc.datasets:
            assert isinstance(d, Dataset)
        assert len(self.dc.datasets) == 8
        assert self.dc.datasets[0].definition == {'param': 'A'}
        assert self.dc.datasets[1].definition == {'param': 'B'}
        assert self.dc.datasets[6].definition == {'param': 'G'}
        assert self.dc.datasets[7].definition == {'param': 'H'}


class Test_DataCollection_baseline(_base_DataCollecionMixin):
    def setup(self):
        self.data = helpers.make_dc_data(ndval=self.known_ndval, rescol=self.known_raw_rescol,
                                         qualcol=self.known_qualcol)
        self.dc = DataCollection(self.data, paramcol='param', stationcol='loc',
                                 ndval=self.known_ndval, rescol=self.known_raw_rescol,
                                 qualcol=self.known_qualcol)

        self.known_groupcols = ['loc', 'param']
        self.known_columns = ['loc', 'param', self.known_raw_rescol, 'cen']
        self.known_bsIter = 10000
        self.known_means = pandas.DataFrame({
            ('Outflow', 'upper'): {
                'H': 3.830, 'C': 4.319, 'D': 7.455, 'G': 3.880,
                'F': 4.600, 'A': 7.166, 'B': 9.050, 'E': 5.615
            },
            ('Inflow', 'mean'): {
                'H': 3.252, 'C': 3.674, 'D': 4.241, 'G': 3.908,
                'F': 5.355, 'A': 5.882, 'B': 6.969, 'E': 3.877
            },
            ('Reference', 'lower'): {
                'H': 1.708, 'C': 1.817, 'D': 2.200, 'G': 1.826,
                'F': 1.477, 'A': 1.397, 'B': 2.537, 'E': 1.241
            },
            ('Reference', 'upper'): {
                'H': 3.411, 'C': 6.405, 'D': 5.602, 'G': 5.626,
                'F': 9.489, 'A': 3.870, 'B': 7.396, 'E': 4.094
            },
            ('Inflow', 'lower'): {
                'H': 1.907, 'C': 1.644, 'D': 1.786, 'G': 2.262,
                'F': 2.539, 'A': 2.784, 'B': 3.781, 'E': 2.403
            },
            ('Inflow', 'upper'): {
                'H': 4.694, 'C': 6.186, 'D':  7.300, 'G': 5.583,
                'F': 8.444, 'A': 9.396, 'B': 10.239, 'E': 5.511
            },
            ('Reference', 'mean'): {
                'H': 2.515, 'C': 3.892, 'D': 3.819, 'G': 3.619,
                'F': 4.988, 'A': 2.555, 'B': 4.919, 'E': 2.587
            },
            ('Outflow', 'mean'): {
                'H': 2.633, 'C': 3.327, 'D': 4.451, 'G': 2.758,
                'F': 3.515, 'A': 4.808, 'B': 6.297, 'E': 4.011
            },
            ('Outflow', 'lower'): {
                'H': 1.568, 'C': 2.346, 'D': 1.863, 'G': 1.643,
                'F': 2.477, 'A': 2.712, 'B': 3.627, 'E': 2.503
            }
        })

        self.known_medians = pandas.DataFrame({
            ('Outflow', 'upper'): {
                'G': 2.421, 'E': 3.512, 'H': 2.387, 'F': 4.227,
                'B': 5.167, 'C': 3.736, 'D': 1.930, 'A': 3.554
            },
            ('Inflow', 'lower'): {
                'G': 0.735, 'E': 1.328, 'H': 1.305, 'F': 1.178,
                'B': 1.717, 'C': 0.924, 'D': 0.766, 'A': 1.329
            },
            ('Outflow', 'lower'): {
                'G': 0.817, 'E': 1.317, 'H': 0.662, 'F': 1.500,
                'B': 0.889, 'C': 1.288, 'D': 0.877, 'A': 1.234
            },
            ('Reference', 'lower'): {
                'G': 1.137, 'E': 0.615, 'H': 0.976, 'F': 0.776,
                'B': 0.832, 'C': 0.938, 'D': 1.190, 'A': 0.691
            },
            ('Reference', 'median'): {
                'G': 2.008, 'E': 1.211, 'H': 1.940, 'F': 1.473,
                'B': 2.118, 'C': 1.833, 'D': 2.187, 'A': 1.293
            },
            ('Inflow', 'upper'): {
                'G': 3.841, 'E': 4.113, 'H': 2.829, 'F': 4.345,
                'B': 6.327, 'C': 2.012, 'D': 2.915, 'A': 3.485
            },
            ('Reference', 'upper'): {
                'G': 2.513, 'E': 1.917, 'H': 2.846, 'F': 2.131,
                'B': 4.418, 'C': 2.893, 'D': 3.719, 'A': 2.051
            },
            ('Outflow', 'median'): {
                'G': 1.559, 'E': 2.101, 'H': 1.604, 'F': 2.495,
                'B': 2.485, 'C': 3.075, 'D': 1.613, 'A': 2.500
            },
            ('Inflow', 'median'): {
                'G': 1.844, 'E': 2.883, 'H': 1.729, 'F': 2.630,
                'B': 3.511, 'C': 1.410, 'D': 1.873, 'A': 2.830
            }
        })

        self.known_genericstat = pandas.DataFrame({
            ('Reference', 'stat'): {
                'D': 0.209, 'A': 0.119, 'B': 0.307, 'H': 0.221,
                'G': 0.189, 'F': 0.099, 'E': 0.211, 'C': 0.135,
            },
            ('Inflow', 'stat'): {
                'D': 0.107, 'A': 0.178, 'B': 0.433, 'H': 0.210,
                'G': 0.096, 'F': 0.157, 'E': 0.236, 'C': 0.116,
            },
            ('Outflow', 'stat'): {
                'D': 0.118, 'A': 0.344, 'B': 0.409, 'H': 0.128,
                'G': 0.124, 'F': 0.300, 'E': 0.126, 'C': 0.219,
            }
        })


class Test_DataCollection_customNDval(Test_DataCollection_baseline):
    known_raw_rescol = 'conc'
    known_roscol = 'ros_conc'
    known_rescol = 'ros_conc'
    known_qualcol = 'anote'
    known_stationcol = 'loc'
    known_paramcol = 'param'
