import os

from nose.tools import *
import numpy as np
import numpy.testing as nptest

from pybmp import testing
from pybmp.testing.testutils import setup_prefix
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex
import matplotlib.pyplot as plt

import pandas
import pandas.util.testing as pdtest

from pybmp.core.features import (
    Parameter,
    DrainageArea,
    Location,
    Dataset,
    DataCollection
)

from pybmp import utils

@nottest
def testfilter(data):
    return data[data['res'] > 5], False


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
        assert_equal(self.known_name, self.param.name)

    def test_units(self):
        assert_equal(self.known_units, self.param.units)

    def test_paramunit_nocomma(self):
        assert_equal(self.known_pu_nocomma, self.param.paramunit(usecomma=False))

    def test_paramunit_comma(self):
        assert_equal(self.known_pu_comma, self.param.paramunit(usecomma=True))


class test_Parameter_simple(_base_Parameter_Mixin):
    def setup(self):
        self.known_name = 'Copper'
        self.known_units = 'ug/L'
        self.known_pu_nocomma = 'Copper (ug/L)'
        self.known_pu_comma = 'Copper, ug/L'
        self.param = Parameter(name=self.known_name, units=self.known_units)


class test_Parameter_complex(_base_Parameter_Mixin):
    #place holder for when TeX/validation/etc gets integrated
    def setup(self):
        self.known_name = 'Nitrate & Nitrite'
        self.known_units = 'mg/L'
        self.known_pu_nocomma = 'Nitrate & Nitrite (mg/L)'
        self.known_pu_comma = 'Nitrate & Nitrite, mg/L'
        self.param = Parameter(name=self.known_name, units=self.known_units)


class test_DrainageArea(object):
    def setup(self):
        self.total_area = 100.0
        self.imp_area = 75.0
        self.bmp_area = 10.0
        self.da = DrainageArea(self.total_area, self.imp_area, self.bmp_area)
        self.volume_conversion = 5
        self.known_storm_runoff = 82.5
        self.known_annual_runoff = 75.25
        self.storm_depth = 1.0
        self.storm_volume = self.storm_depth * (self.total_area + self.bmp_area)
        self.annualFactor = 0.9

    def test_total_area(self):
        assert_true(hasattr(self.da, 'total_area'))
        assert_equal(self.total_area, self.da.total_area)

    def test_imp_area(self):
        assert_true(hasattr(self.da, 'imp_area'))
        assert_equal(self.imp_area, self.da.imp_area)

    def test_bmp_area(self):
        assert_true(hasattr(self.da, 'bmp_area'))
        assert_equal(self.bmp_area, self.da.bmp_area)

    def test_simple_method_noConversion(self):
        assert_true(hasattr(self.da, 'simple_method'))
        runoff = self.da.simple_method(1)
        nptest.assert_almost_equal(self.known_storm_runoff, runoff, decimal=3)
        assert_greater(self.storm_volume, runoff)

    def test_simple_method_Conversion(self):
        assert_true(hasattr(self.da, 'simple_method'))
        runoff = self.da.simple_method(1, volume_conversion=self.volume_conversion)
        nptest.assert_almost_equal(self.known_storm_runoff * self.volume_conversion, runoff, decimal=3)
        assert_greater(self.storm_volume * self.volume_conversion, runoff)

    def test_simple_method_annualFactor(self):
        assert_true(hasattr(self.da, 'simple_method'))
        runoff = self.da.simple_method(1, annualFactor=self.annualFactor)
        nptest.assert_almost_equal(self.known_annual_runoff, runoff, decimal=3)
        assert_greater(self.storm_volume, runoff)


class _base_LocationMixin(object):
    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nottest
    def main_setup(self):
        # path stuff
        self.prefix = setup_prefix('core.features')


        # basic test data
        self.tolerance = 0.05
        self.known_bsIter = 10000
        self.data = utils.ros.getTestData()

        # Location stuff
        self.known_station_name = 'Influent'
        self.known_station_type = 'inflow'
        self.known_plot_marker = 'o'
        self.known_scatter_marker = 'v'
        self.known_color = np.array((0.32157, 0.45271, 0.66667))
        self.known_rescol = 'res'
        self.known_qualcol = 'qual'
        self.known_all_positive = True
        self.known_include = True
        self.known_exclude = False
        self.known_hasData = True
        self.known_min_detect = 2.00
        self.known_min_DL = 5.00
        self.fig, self.ax = plt.subplots()

        self.known_filtered_include = False

    def teardown(self):
        self.ax.cla()
        self.fig.clf()
        plt.close('all')

    def test_bsIter(self):
        assert_true(hasattr(self.loc, 'bsIter'))
        assert_equal(self.loc.bsIter, self.known_bsIter)

    def test_useROS(self):
        assert_true(hasattr(self.loc, 'useROS'))
        assert_equal(self.loc.useROS, self.known_useRos)

    def test_name(self):
        assert_true(hasattr(self.loc, 'name'))
        assert_equal(self.loc.name, self.known_station_name)

    def test_N(self):
        assert_true(hasattr(self.loc, 'N'))
        assert_equal(self.loc.N, self.known_N)

    def test_hasData(self):
        assert_true(hasattr(self.loc, 'hasData'))
        assert_equal(self.loc.hasData, self.known_hasData)

    def test_ND(self):
        assert_true(hasattr(self.loc, 'ND'))
        assert_equal(self.loc.ND, self.known_ND)

    def test_fractionND(self):
        assert_true(hasattr(self.loc, 'fractionND'))
        assert_equal(self.loc.fractionND, self.known_fractionND)

    def test_NUnique(self):
        assert_true(hasattr(self.loc, 'NUnique'))
        assert_equal(self.loc.NUnique, self.known_NUnique)

    def test_analysis_space(self):
        assert_true(hasattr(self.loc, 'analysis_space'))
        assert_equal(self.loc.analysis_space, self.known_analysis_space)

    def test_cov(self):
        assert_true(hasattr(self.loc, 'cov'))
        nptest.assert_allclose(self.loc.cov, self.known_cov, rtol=self.tolerance)

    def test_geomean(self):
        assert_true(hasattr(self.loc, 'geomean'))
        nptest.assert_allclose(self.loc.geomean, self.known_geomean, rtol=self.tolerance)

    def test_geomean_conf_interval(self):
        assert_true(hasattr(self.loc, 'geomean_conf_interval'))
        nptest.assert_allclose(self.loc.geomean_conf_interval, self.known_geomean_conf_interval, rtol=self.tolerance)

    def test_geostd(self):
        assert_true(hasattr(self.loc, 'geostd'))
        nptest.assert_allclose(self.loc.geostd, self.known_geostd, rtol=self.tolerance)

    def test_logmean(self):
        assert_true(hasattr(self.loc, 'logmean'))
        nptest.assert_allclose(self.loc.logmean, self.known_logmean, rtol=self.tolerance)

    def test_logmean_conf_interval(self):
        assert_true(hasattr(self.loc, 'logmean_conf_interval'))
        nptest.assert_allclose(self.loc.logmean_conf_interval, self.known_logmean_conf_interval, rtol=self.tolerance)

    def test_logstd(self):
        assert_true(hasattr(self.loc, 'logstd'))
        nptest.assert_allclose(self.loc.logstd, self.known_logstd, rtol=self.tolerance)

    def test_max(self):
        assert_true(hasattr(self.loc, 'max'))
        assert_equal(self.loc.max, self.known_max)

    def test_mean(self):
        assert_true(hasattr(self.loc, 'mean'))
        nptest.assert_allclose(self.loc.mean, self.known_mean, rtol=self.tolerance)

    def test_mean_conf_interval(self):
        assert_true(hasattr(self.loc, 'mean_conf_interval'))
        nptest.assert_allclose(self.loc.mean_conf_interval, self.known_mean_conf_interval, rtol=self.tolerance)

    def test_median(self):
        assert_true(hasattr(self.loc, 'median'))
        nptest.assert_allclose(self.loc.median, self.known_median, rtol=self.tolerance)

    def test_median_conf_interval(self):
        assert_true(hasattr(self.loc, 'median_conf_interval'))
        nptest.assert_allclose(self.loc.median_conf_interval, self.known_median_conf_interval, rtol=self.tolerance)

    def test_min(self):
        assert_true(hasattr(self.loc, 'min'))
        assert_equal(self.loc.min, self.known_min)

    def test_min_detect(self):
        assert_true(hasattr(self.loc, 'min_detect'))
        assert_equal(self.loc.min_detect, self.known_min_detect)

    def test_min_DL(self):
        assert_true(hasattr(self.loc, 'min_DL'))
        assert_equal(self.loc.min_DL, self.known_min_DL)

    def test_pctl10(self):
        assert_true(hasattr(self.loc, 'pctl10'))
        nptest.assert_allclose(self.loc.pctl10, self.known_pctl10, rtol=self.tolerance)

    def test_pctl25(self):
        assert_true(hasattr(self.loc, 'pctl25'))
        nptest.assert_allclose(self.loc.pctl25, self.known_pctl25, rtol=self.tolerance)

    def test_pctl75(self):
        assert_true(hasattr(self.loc, 'pctl75'))
        nptest.assert_allclose(self.loc.pctl75, self.known_pctl75, rtol=self.tolerance)

    def test_pctl90(self):
        assert_true(hasattr(self.loc, 'pctl90'))
        nptest.assert_allclose(self.loc.pctl90, self.known_pctl90, rtol=self.tolerance)

    def test_plognorm(self):
        assert_true(hasattr(self.loc, 'plognorm'))
        nptest.assert_allclose(self.loc.plognorm, self.known_plognorm, rtol=self.tolerance)

    def test_pnorm(self):
        assert_true(hasattr(self.loc, 'pnorm'))
        nptest.assert_allclose(self.loc.pnorm, self.known_pnorm, rtol=self.tolerance)

    def test_skew(self):
        assert_true(hasattr(self.loc, 'skew'))
        nptest.assert_allclose(self.loc.skew, self.known_skew, rtol=self.tolerance)

    def test_std(self):
        assert_true(hasattr(self.loc, 'std'))
        nptest.assert_allclose(self.loc.std, self.known_std, rtol=self.tolerance)

    def test_station_name(self):
        assert_true(hasattr(self.loc, 'station_name'))
        assert_equal(self.loc.station_name, self.known_station_name)

    def test_station_type(self):
        assert_true(hasattr(self.loc, 'station_type'))
        assert_equal(self.loc.station_type, self.known_station_type)

    def test_symbology_plot_marker(self):
        assert_true(hasattr(self.loc, 'plot_marker'))
        assert_equal(self.loc.plot_marker, self.known_plot_marker)

    def test_symbology_scatter_marker(self):
        assert_true(hasattr(self.loc, 'scatter_marker'))
        assert_equal(self.loc.scatter_marker, self.known_scatter_marker)

    def test_symbology_color(self):
        assert_true(hasattr(self.loc, 'color'))
        nptest.assert_almost_equal(
            np.array(self.loc.color),
            self.known_color,
            decimal=4
        )

    def test_data(self):
        assert_true(hasattr(self.loc, 'data'))
        assert_true(isinstance(self.loc.data, pandas.Series))
        assert_equal(self.loc.data.name, 'res')

    def test_full_data(self):
        assert_true(hasattr(self.loc, 'full_data'))
        assert_true(isinstance(self.loc.full_data, pandas.DataFrame))

    def test_full_data_columns(self):
        assert_true('res' in self.data.columns)
        assert_true('qual' in self.data.columns)

    def test_all_positive(self):
        assert_true(hasattr(self.loc, 'all_positive'))
        assert_equal(self.loc.all_positive, self.known_all_positive)

    def test_include(self):
        assert_true(hasattr(self.loc, 'include'))
        assert_equal(self.loc.include, self.known_include)

    def test_exclude(self):
        assert_true(hasattr(self.loc, 'exclude'))
        assert_equal(self.loc.exclude, self.known_exclude)

    def test_include_setter(self):
        self.loc.include = not self.known_include
        assert_equal(self.loc.include, not self.known_include)
        assert_equal(self.loc.exclude, not self.known_exclude)

    def test_boxplot_baseline(self):
        assert_true(hasattr(self.loc, 'boxplot'))
        fig = self.loc.boxplot(ax=self.ax, pos=1, yscale='log', notch=True,
                               showmean=True, width=0.8, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_Loc_Box_useROS_{0}.png'.format(self.loc.useROS)))

    @raises(ValueError)
    def test_boxplot_badAxes(self):
        fig = self.loc.boxplot(ax=5, pos=1, yscale='log', notch=True,
                               showmean=True, width=0.8)

    @raises(ValueError)
    def test_boxplot_badYscale(self):
        fig = self.loc.boxplot(ax=self.ax, pos=1, yscale='JUNK', notch=True,
                               showmean=True, width=0.8)

    def test_probplot_baseline(self):
        assert_true(hasattr(self.loc, 'probplot'))
        self.loc.probplot(yscale='log')
        fig = self.loc.probplot(ax=self.ax, yscale='log', ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_Loc_Prob_useROS_{0}.png'.format(self.loc.useROS)))

    def test_probplot_baseline_QQ(self):
        assert_true(hasattr(self.loc, 'probplot'))
        self.loc.probplot(yscale='log')
        fig = self.loc.probplot(ax=self.ax, yscale='log', probAxis=False)
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_Loc_QQ_useROS_{0}.png'.format(self.loc.useROS)))

    @raises(ValueError)
    def test_probplot_badAxes(self):
        fig = self.loc.probplot(ax=3, yscale='log')

    @raises(ValueError)
    def test_probplot_badYscale(self):
        self.loc.probplot(yscale='JUNK')

    def test_statplot_baseline(self):
        assert_true(hasattr(self.loc, 'statplot'))
        fig = self.loc.statplot(yscale='log', probAxis=True, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_Loc_StatProb_useROS_{0}.png'.format(self.loc.useROS)))

    def test_statplot_baseline_QQ(self):
        assert_true(hasattr(self.loc, 'statplot'))
        fig = self.loc.statplot(yscale='log', probAxis=False, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_Loc_StatQQ_useROS_{0}.png'.format(self.loc.useROS)))

    @raises(ValueError)
    def test_statplot_badYScale(self):
        assert_true(hasattr(self.loc, 'statplot'))
        fig = self.loc.statplot(yscale='JUNK', probAxis=False, ylabel='Test Label')

    def test_verticalScatter_baseline(self):
        assert_true(hasattr(self.loc, 'verticalScatter'))
        fig = self.loc.verticalScatter(ax=self.ax, pos=2, width=1, alpha=0.5)
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_Loc_verticalScatter_useROS_{0}.png'.format(self.loc.useROS)))

    def test_verticalScatter_ylabel_yscale(self):
        fig = self.loc.verticalScatter(ax=self.ax, pos=2, width=1, alpha=0.5,
                                        ylabel='testlabel', yscale='linear')
        fig.savefig(self.makePath('test_Loc_verticalScatter_YScaleYLabel_useROS_{0}.png'.format(self.loc.useROS)))

    def test_verticalScatter_DontIgnoreROS(self):
        fig = self.loc.verticalScatter(ax=self.ax, pos=2, width=1, alpha=0.5,
                                        ylabel='testlabel', yscale='linear',
                                        ignoreROS=False)
        fig.savefig(self.makePath('test_Loc_verticalScatter_DontIgnoreROS_useROS_{0}.png'.format(self.loc.useROS)))

    @raises(ValueError)
    def test_verticalScatter_badAxes(self):
        fig = self.loc.verticalScatter(ax='JUNK', pos=2, width=1, alpha=0.5)

    @raises(ValueError)
    def test_verticalScatter_badYScale(self):
        fig = self.loc.verticalScatter(ax=self.ax, pos=2, width=1, alpha=0.5, yscale='JUNK')

    def test_applyFilter_exists(self):
        assert_true(hasattr(self.loc, 'applyFilter'))

    def test_applyFilter_works(self):
        self.loc.applyFilter(testfilter)
        assert_tuple_equal(self.loc.filtered_data.shape, self.known_filtered_shape)
        assert_tuple_equal(self.loc.full_data.shape, self.known_filtered_shape)
        assert_tuple_equal(self.loc.data.shape, self.known_filtered_data_shape)
        assert_equal(self.loc.include, self.known_filtered_include)

    @raises(ValueError)
    def test_applyFilter_badDataOutput(self):

        def badTestFilter(data):
            return 4, False

        self.loc.applyFilter(badTestFilter)

    @raises(ValueError)
    def test_applyFilter_badIncludeOutput(self):

        def badTestFilter(data):
            df = pandas.Series(np.random.normal(size=37))
            return df, 'JUNK'

        self.loc.applyFilter(badTestFilter)


class test_Location_ROS(_base_LocationMixin):
    def setup(self):
        self.main_setup()
        self.loc = Location(self.data, station_type='inflow', bsIter=self.known_bsIter,
                            rescol='res', qualcol='qual', useROS=True)

        # known statistics
        self.known_N = 35
        self.known_ND = 7
        self.known_fractionND = 0.2
        self.known_NUnique = 31
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
        self.known_median_conf_interval = [5.66, 8.71]
        self.known_min = 2.0
        self.known_pctl10 = 4.04355908285
        self.known_pctl25 = 5.615
        self.known_pctl75 = 11.725
        self.known_pctl90 = 19.178
        self.known_plognorm = 0.521462738514
        self.known_pnorm = 0.00179254170507
        self.known_skew = 0.869052892573
        self.known_std = 5.52730949374
        self.known_useRos = True
        self.known_all_positive = True
        self.known_filtered_shape = (27,2)
        self.known_filtered_data_shape = (27,)


class test_Location_noROS(_base_LocationMixin):
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
        self.known_plognorm = 0.306435495615
        self.known_pnorm = 0.00323620648123
        self.known_skew = 0.853756570358
        self.known_std = 5.24122841148
        self.known_useRos = False
        self.known_filtered_shape = (30,2)
        self.known_filtered_data_shape = (30,)

    def test_median_conf_interval(self):
        assert_true(hasattr(self.loc, 'median_conf_interval'))
        try:
            nptest.assert_allclose(self.loc.median_conf_interval,
                                   self.known_median_conf_interval1,
                                   rtol=self.tolerance)
        except AssertionError:
            nptest.assert_allclose(self.loc.median_conf_interval,
                                   self.known_median_conf_interval2,
                                   rtol=self.tolerance)


class test_Dataset(object):
    def setup(self):
        self.maxDiff = None
        # path stuff
        self.prefix = setup_prefix('core.features')

        # basic test data
        self.tolerance = 0.05
        self.known_bsIter = 10000

        in_data = utils.ros.getTestData()
        in_data['res'] += 3

        out_data = utils.ros.getTestData()
        out_data['res'] -= 1.5

        self.influent = Location(in_data, station_type='inflow', bsIter=self.known_bsIter,
                                 rescol='res', qualcol='qual', useROS=False)

        self.effluent = Location(out_data, station_type='outflow', bsIter=self.known_bsIter,
                                 rescol='res', qualcol='qual', useROS=False)

        self.ds = Dataset(self.influent, self.effluent)

        self.fig, self.ax = plt.subplots()

        self.known_dumpFile = None
        self.known_kendall_stats = (1.000000e+00,   2.916727e-17)
        self.known_kendall_tau = self.known_kendall_stats[0]
        self.known_kendall_p = self.known_kendall_stats[1]

        self.known_mannwhitney_stats = (2.980000e+02,   1.125761e-04)
        self.known_mannwhitney_u = self.known_mannwhitney_stats[0]
        self.known_mannwhitney_p = self.known_mannwhitney_stats[1] * 2

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

    def teardown(self):
        self.ax.cla()
        self.fig.clf()
        plt.close('all')

    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    def test_data(self):
        assert_true(hasattr(self.ds, 'data'))
        assert_true(isinstance(self.ds.data, pandas.DataFrame))

    def test_paired_data(self):
        assert_true(hasattr(self.ds, 'paired_data'))
        assert_true(isinstance(self.ds.paired_data, pandas.DataFrame))

    def test__non_paired_stats(self):
        assert_true(hasattr(self.ds, '_non_paired_stats'))
        assert_equal(self.ds._non_paired_stats, self.known__non_paired_stats)

    def test__paired_stats(self):
        assert_true(hasattr(self.ds, '_paired_stats'))
        assert_equal(self.ds._paired_stats, self.known__paired_stats)

    def test_name(self):
        assert_true(hasattr(self.ds, 'name'))
        assert_true(self.ds.name is None)

    def test_name_set(self):
        assert_true(hasattr(self.ds, 'name'))
        testname = 'Test Name'
        self.ds.name = testname
        assert_equal(self.ds.name, testname)

    def test_defintion_default(self):
        assert_true(hasattr(self.ds, 'definition'))
        assert_dict_equal(self.ds.definition, {})

    def test_defintion_set(self):
        assert_true(hasattr(self.ds, 'definition'))
        self.ds.definition = self.known_definition
        assert_dict_equal(self.ds.definition, self.known_definition)

    def test_include(self):
        assert_true(hasattr(self.ds, 'include'))
        assert_equal(self.ds.include, self.known_include)

    def test_exclude(self):
        assert_true(hasattr(self.ds, 'exclude'))
        assert_equal(self.ds.exclude, self.known_exclude)

    def test_wilcoxon_z(self):
        assert_true(hasattr(self.ds, 'wilcoxon_z'))
        nptest.assert_allclose(self.ds.wilcoxon_z, self.known_wilcoxon_z, rtol=self.tolerance)

    def test_wilcoxon_p(self):
        assert_true(hasattr(self.ds, 'wilcoxon_p'))
        nptest.assert_allclose(self.ds.wilcoxon_p, self.known_wilcoxon_p, rtol=self.tolerance)

    def test_mannwhitney_u(self):
        assert_true(hasattr(self.ds, 'mannwhitney_u'))
        nptest.assert_allclose(self.ds.mannwhitney_u, self.known_mannwhitney_u, rtol=self.tolerance)

    def test_mannwhitney_p(self):
        assert_true(hasattr(self.ds, 'mannwhitney_p'))
        nptest.assert_allclose(self.ds.mannwhitney_p, self.known_mannwhitney_p, rtol=self.tolerance)

    def test_kendall_tau(self):
        assert_true(hasattr(self.ds, 'kendall_tau'))
        nptest.assert_allclose(self.ds.kendall_tau, self.known_kendall_tau, rtol=self.tolerance)

    def test_kendall_p(self):
        assert_true(hasattr(self.ds, 'kendall_p'))
        nptest.assert_allclose(self.ds.kendall_p, self.known_kendall_p, rtol=self.tolerance)

    def test_spearman_rho(self):
        assert_true(hasattr(self.ds, 'spearman_rho'))
        nptest.assert_allclose(self.ds.spearman_rho, self.known_spearman_rho, rtol=self.tolerance)

    def test_spearman_p(self):
        assert_true(hasattr(self.ds, 'spearman_p'))
        nptest.assert_allclose(self.ds.spearman_p, self.known_spearman_p, rtol=self.tolerance)

    def test_theil_medslope(self):
        assert_true(hasattr(self.ds, 'theil_medslope'))
        nptest.assert_allclose(self.ds.theil_medslope, self.known_theil_medslope, rtol=self.tolerance)

    def test_theil_intercept(self):
        assert_true(hasattr(self.ds, 'theil_intercept'))
        nptest.assert_allclose(self.ds.theil_intercept, self.known_theil_intercept, rtol=self.tolerance)

    def test_theil_loslope(self):
        assert_true(hasattr(self.ds, 'theil_loslope'))
        nptest.assert_allclose(self.ds.theil_loslope, self.known_theil_loslope, rtol=self.tolerance)

    def test_theil_hilope(self):
        assert_true(hasattr(self.ds, 'theil_hislope'))
        nptest.assert_allclose(self.ds.theil_hislope, self.known_theil_hislope, rtol=self.tolerance)

    def test_wilcoxon_stats(self):
        assert_true(hasattr(self.ds, '_wilcoxon_stats'))
        nptest.assert_allclose(self.ds._wilcoxon_stats, self.known_wilcoxon_stats, rtol=self.tolerance)

    def test_mannwhitney_stats(self):
        assert_true(hasattr(self.ds, '_mannwhitney_stats'))
        nptest.assert_allclose(self.ds._mannwhitney_stats, self.known_mannwhitney_stats, rtol=self.tolerance)

    def test_kendall_stats(self):
        assert_true(hasattr(self.ds, '_kendall_stats'))
        nptest.assert_allclose(self.ds._kendall_stats, self.known_kendall_stats, rtol=self.tolerance)

    def test_spearman_stats(self):
        assert_true(hasattr(self.ds, '_spearman_stats'))
        nptest.assert_allclose(self.ds._spearman_stats, self.known_spearman_stats, rtol=self.tolerance)

    def test_theil_stats(self):
        assert_true(hasattr(self.ds, '_theil_stats'))
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

        assert_true(not self.ds._theil_stats['is_inverted'])

        assert_true('estimated_effluent' in list(self.ds._theil_stats.keys()))
        assert_true('estimate_error' in list(self.ds._theil_stats.keys()))

    def test_theil_effluent_ties(self):
        cache_theil_stats = self.ds._theil_stats
        self.ds.useROS = True
        self.ds.effluent.useROS = False # restores ties in the effl data
        assert_true(self.ds._theil_stats['is_inverted'])

    def test_medianCIsOverlap(self):
        assert_equal(self.known_medianCIsOverlap, self.ds.medianCIsOverlap)

    def test_boxplot_baseline(self):
        assert_true(hasattr(self.ds, 'boxplot'))
        fig = self.ds.boxplot(ax=self.ax, pos=1, yscale='log', notch=True,
                               showmean=True, width=0.8, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_DS_Box_BothLocs_NoName.png'))

    def test_boxplot_baseline_withName(self):
        self.ds.name = 'Test Dataset'
        fig = self.ds.boxplot(ax=self.ax, pos=1, yscale='log', notch=True,
                               showmean=True, width=0.8, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_DS_Box_BothLocs_Name.png'))

    @raises(ValueError)
    def test_boxplot_badAxes(self):
        fig = self.ds.boxplot(ax=5, pos=1, yscale='log', notch=True,
                              showmean=True, width=0.8)

    @raises(ValueError)
    def test_boxplot_badYscale(self):
        fig = self.ds.boxplot(ax=self.ax, pos=1, yscale='JUNK', notch=True,
                               showmean=True, width=0.8)

    def test_probplot_baseline(self):
        assert_true(hasattr(self.ds, 'probplot'))
        self.ds.probplot(yscale='log')
        fig = self.ds.probplot(ax=self.ax, yscale='log', ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_DS_Prob.png'))

    def test_probplot_baseline_QQ(self):
        assert_true(hasattr(self.ds, 'probplot'))
        self.ds.probplot(yscale='log')
        fig = self.ds.probplot(ax=self.ax, yscale='log', probAxis=False)
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_DS_QQ'))

    @raises(ValueError)
    def test_probplot_badAxes(self):
        assert_true(hasattr(self.ds, 'probplot'))
        self.ds.probplot(yscale='log')
        fig = self.ds.probplot(ax=3, yscale='log')
        assert_true(isinstance(fig, plt.Figure))

    @raises(ValueError)
    def test_probplot_badYscale(self):
        assert_true(hasattr(self.ds, 'probplot'))
        self.ds.probplot(yscale='JUNK')

    def test_statplot_baseline(self):
        assert_true(hasattr(self.ds, 'statplot'))
        fig = self.ds.statplot(yscale='log', probAxis=True, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_DS_StatProb'))

    def test_statplot_baseline_QQ(self):
        assert_true(hasattr(self.ds, 'statplot'))
        fig = self.ds.statplot(yscale='log', probAxis=False, ylabel='Test Label')
        assert_true(isinstance(fig, plt.Figure))
        fig.savefig(self.makePath('test_DS_StatQQ'))

    @raises(ValueError)
    def test_statplot_badYScale(self):
        assert_true(hasattr(self.ds, 'statplot'))
        fig = self.ds.statplot(yscale='JUNK', probAxis=False, ylabel='Test Label')

    def test_joinplot(self):
        assert_true(hasattr(self.ds, 'jointplot'))
        self.ds.jointplot()


    def test_scatterplot_baseline(self):
        assert_true(hasattr(self.ds, 'scatterplot'))
        fig = self.ds.scatterplot(ax=self.ax)
        assert_true(isinstance(fig, plt.Figure))
        leg = fig.axes[0].get_legend()
        fig.savefig(self.makePath('test_DS_Scatter_Baseline.png'),
                    bbox_extra_artists=(leg,), bbox_inches='tight')

    def test_scatterplot_noROS(self):
        assert_true(hasattr(self.ds, 'scatterplot'))
        fig = self.ds.scatterplot(ax=self.ax, useROS=False)
        assert_true(isinstance(fig, plt.Figure))
        leg = fig.axes[0].get_legend()
        fig.savefig(self.makePath('test_DS_Scatter_NoROS.png'),
                    bbox_extra_artists=(leg,), bbox_inches='tight')

    def test_scatterplot_xlinear(self):
        fig = self.ds.scatterplot(ax=self.ax, xscale='linear', yscale='log', one2one=True)
        leg = fig.axes[0].get_legend()
        fig.savefig(self.makePath('test_DS_Scatter_xlinear.png'),
                    bbox_inches='tight', bbox_extra_artists=(leg,))

    def test_scatterplot_ylinear(self):
        fig = self.ds.scatterplot(ax=self.ax, xscale='log', yscale='linear', one2one=True)
        leg = fig.axes[0].get_legend()
        fig.savefig(self.makePath('test_DS_Scatter_ylinear.png'),
                    bbox_inches='tight', bbox_extra_artists=(leg,))

    @raises(ValueError)
    def test_scatterplot_xlog_bad(self):
        fig = self.ds.scatterplot(ax=self.ax, xscale='junk')

    @raises(ValueError)
    def test_scatterplot_ylog_bad(self):
        fig = self.ds.scatterplot(ax=self.ax, yscale='junk')

    @raises(ValueError)
    def test_scatterplot_badAxes(self):
        fig = self.ds.scatterplot(ax='junk')

    def test_scatterplot_one2one(self):
        fig = self.ds.scatterplot(ax=self.ax, one2one=True)
        fig.savefig(self.makePath('test_DS_Scatter_with1-1.png'),
                    bbox_inches='tight', bbox_extra_artists=(fig.axes[0].get_legend(),))

    def test_scatterplot_one2one_noROS(self):
        fig = self.ds.scatterplot(ax=self.ax, one2one=True, useROS=False)
        fig.savefig(self.makePath('test_DS_Scatter_with1-1_noROS.png'),
                    bbox_inches='tight', bbox_extra_artists=(fig.axes[0].get_legend(),))

    def test_scatterplot_check_axes_limits(self):
        fig = self.ds.scatterplot(ax=self.ax)
        assert_tuple_equal(self.ax.get_xlim(), self.ax.get_ylim())

    def test__plot_nds_both(self):
        self.ds._plot_nds(self.ax, which='both')

    def test__plot_nds_effluent(self):
        self.ds._plot_nds(self.ax, which='effluent')

    def test__plot_nds_influent(self):
        self.ds._plot_nds(self.ax, which='influent')

    def test__plot_nds_neither(self):
        self.ds._plot_nds(self.ax, which='neither')

    @raises(ValueError)
    def test__plot_nds_error(self):
        self.ds._plot_nds(self.ax, which='JUNK')

    def test__repr__normal(self):
        self.ds.__repr__

    def test_repr__None(self):
        self.ds.definition = None
        self.ds.__repr__

    def test_reset_useROS(self):
        self.ds.useROS = True
        infl_ros_mean = self.ds.influent.data.iloc[0]
        effl_ros_mean = self.ds.effluent.data.iloc[0]

        self.ds.useROS = False
        infl_raw_mean = self.ds.influent.data.iloc[0]
        effl_raw_mean = self.ds.effluent.data.iloc[0]

        assert_true(infl_ros_mean != infl_raw_mean)
        assert_true(effl_ros_mean != effl_raw_mean)


class _base_DataCollecionMixin(object):
    @nottest
    def _base_setup(self):
        self.known_raw_rescol = 'res'
        self.known_roscol = 'ros_res'
        self.known_qualcol = 'qual'
        self.known_stationcol = 'loc'
        self.known_paramcol = 'param'
        self.known_ndval = 'ND'

    def teardown(self):
        plt.close('all')

    def test__raw_rescol(self):
        assert_equal(self.dc._raw_rescol, self.known_raw_rescol)

    def test_data(self):
        assert_true(isinstance(self.dc.data, pandas.DataFrame))

    def test_roscol(self):
        assert_equal(self.dc.roscol, self.known_roscol)

    def test_rescol(self):
        assert_equal(self.dc.rescol, self.known_rescol)

    def test_qualcol(self):
        assert_equal(self.dc.qualcol, self.known_qualcol)

    def test_stationncol(self):
        assert_equal(self.dc.stationcol, self.known_stationcol)

    def test_paramcol(self):
        assert_equal(self.dc.paramcol, self.known_paramcol)

    def test_ndval(self):
        assert_equal(self.dc.ndval, self.known_ndval)

    def test_bsIter(self):
        assert_equal(self.dc.bsIter, self.known_bsIter)

    def test_groupby(self):
        assert_equal(self.dc.groupby, self.known_groupby)

    def test_columns(self):
        assert_equal(self.dc.columns, self.known_columns)

    def test_filterfxn(self):
        assert_true(hasattr(self.dc, 'filterfxn'))

    def test_tidy_exists(self):
        assert_true(hasattr(self.dc, 'tidy'))

    def test_tidy_type(self):
        assert_true(isinstance(self.dc.tidy, pandas.DataFrame))

    def test_tidy_cols(self):
        tidycols = self.dc.tidy.columns.tolist()
        knowncols = self.known_columns + [self.known_roscol]
        assert_list_equal(sorted(tidycols), sorted(knowncols))

    def test_means(self):
        pdtest.assert_frame_equal(
            self.known_means,
            np.round(self.dc.means, 3),
            check_names=False
        )

    def test_medians(self):
        pdtest.assert_frame_equal(
            self.known_medians,
            np.round(self.dc.medians, 3),
            check_names=False
        )

    def test__generic_stat(self):
        pdtest.assert_frame_equal(
            self.known_genericstat,
            np.round(self.dc._generic_stat(np.min), 3),
            check_names=False
        )


class test_DataCollection_baseline(_base_DataCollecionMixin):
    def setup(self):
        self._base_setup()
        self.dc = DataCollection(make_dc_data(),
                                 paramcol='param',
                                 stationcol='loc')

        self.known_rescol = 'ros_res'
        self.known_groupby = ['loc', 'param']
        self.known_columns = ['loc', 'param', 'res', 'qual']
        self.known_bsIter = 10000
        self.known_means = pandas.DataFrame({
            ('Inflow', 'stat'): {
                'H': 3.271, 'A': 5.871, 'B': 6.967, 'C': 3.715,
                'D': 4.227, 'E': 3.894, 'F': 5.338, 'G': 3.882
            },
            ('Outflow', 'lower'): {
                'H': 1.541, 'A': 2.612, 'B': 3.627, 'C': 2.354,
                'D': 1.916, 'E': 2.571, 'F': 2.492, 'G': 1.699
            },
            ('Inflow', 'lower'): {
                'H': 1.947, 'A': 2.801, 'B': 3.894, 'C': 1.568,
                'D': 1.756, 'E': 2.436, 'F': 2.689, 'G': 2.359
            },
            ('Reference', 'upper'): {
                'H': 3.367, 'A': 3.822, 'B': 7.500, 'C': 6.529,
                'D': 5.590, 'E': 4.104, 'F': 9.527, 'G': 5.615
            },
            ('Reference', 'lower'): {
                'H': 1.692, 'A': 1.396, 'B': 2.615, 'C': 1.904,
                'D': 2.225, 'E': 1.272, 'F': 1.525, 'G': 1.907
            },
            ('Reference', 'stat'): {
                'H': 2.497, 'A': 2.562, 'B': 4.930, 'C': 3.880,
                'D': 3.838, 'E': 2.600, 'F': 4.990, 'G': 3.619
            },
            ('Inflow', 'upper'): {
                'H': 4.704, 'A': 9.258, 'B': 10.225, 'C': 6.018,
                'D': 7.488, 'E': 5.471, 'F': 8.529, 'G': 5.655
            },
            ('Outflow', 'stat'): {
                'H': 2.641, 'A': 4.797, 'B': 6.314, 'C': 3.325,
                'D': 4.452, 'E': 4.000, 'F': 3.506, 'G': 2.755
            },
            ('Outflow', 'upper'): {
                'H': 3.796, 'A': 7.098, 'B': 9.114, 'C': 4.337,
                'D': 7.329, 'E': 5.657, 'F': 4.609, 'G': 3.960
            }
        })

        self.known_medians = pandas.DataFrame({
            ('Inflow', 'stat'): {
                'H': 1.8760, 'A': 2.7040, 'B': 3.8400, 'C': 1.4580,
                'D': 1.9330, 'E': 2.8010, 'F': 2.6450, 'G': 2.1240
            },
            ('Outflow', 'lower'): {
                'H': 0.6360, 'A': 1.2340, 'B': 0.8890, 'C': 1.3180,
                'D': 0.8750, 'E': 1.3170, 'F': 1.5000, 'G': 0.8030
            },
            ('Inflow', 'lower'): {
                'H': 1.3050, 'A': 1.3290, 'B': 1.6910, 'C': 0.8940,
                'D': 0.7660, 'E': 1.3280, 'F': 1.1780, 'G': 0.7730
            },
            ('Reference', 'upper'): {
                'H': 2.8460, 'A': 2.0510, 'B': 4.4180, 'C': 2.8300,
                'D': 3.7190, 'E': 1.9170, 'F': 2.1310, 'G': 2.5130
            },
            ('Reference', 'lower'): {
                'H': 0.9760, 'A': 0.6950, 'B': 0.8320, 'C': 0.8930,
                'D': 1.1370, 'E': 0.6120, 'F': 0.6860, 'G': 1.0720
            },
            ('Reference', 'stat'): {
                'H': 1.9940, 'A': 1.3280, 'B': 2.2990, 'C': 1.868,
                'D': 2.3310, 'E': 1.2310, 'F': 1.5720, 'G': 1.9810
            },
            ('Inflow', 'upper'): {
                'H': 2.8290, 'A': 3.4850, 'B': 6.0830, 'C': 2.0080,
                'D': 2.9880, 'E': 4.1130, 'F': 4.3450, 'G': 3.8410
            },
            ('Outflow', 'stat'): {
                'H': 1.6270, 'A': 2.5100, 'B': 2.9240, 'C': 2.8590,
                'D': 1.6220, 'E': 2.3140, 'F': 2.5290, 'G': 1.5280
            },
            ('Outflow', 'upper'): {
                'H': 2.3870, 'A': 3.5540, 'B': 5.1670, 'C': 3.8340,
                'D': 1.9300, 'E': 3.7200, 'F': 4.2270, 'G': 2.3690
            }
        })

        self.known_genericstat = pandas.DataFrame({
            ('Inflow', 'stat'): {
                'H': 0.2100, 'A': 0.1780, 'B': 0.4330, 'C': 0.1160,
                'D': 0.1070, 'E': 0.2360, 'F': 0.1570, 'G': 0.0960
            },
            ('Outflow', 'lower'): {
                'H': 0.1280, 'A': 0.3440, 'B': 0.4090, 'C': 0.2190,
                'D': 0.1180, 'E': 0.1260, 'F': 0.3000, 'G': 0.1240
            },
            ('Inflow', 'lower'): {
                'H': 0.2100, 'A': 0.1780, 'B': 0.4330, 'C': 0.1160,
                'D': 0.1070, 'E': 0.2360, 'F': 0.1570, 'G': 0.0960
            },
            ('Reference', 'upper'): {
                'H': 0.4620, 'A': 0.4840, 'B': 0.4580, 'C': 0.3570,
                'D': 0.5630, 'E': 0.2970, 'F': 0.3600, 'G': 0.4000
            },
            ('Reference', 'lower'): {
                'H': 0.2210, 'A': 0.1190, 'B': 0.3070, 'C': 0.1350,
                'D': 0.2090, 'E': 0.2110, 'F': 0.0990, 'G': 0.1890
            },
            ('Reference', 'stat'): {
                'H': 0.2210, 'A': 0.1190, 'B': 0.3070, 'C': 0.1350,
                'D': 0.2090, 'E': 0.2110, 'F': 0.0990, 'G': 0.1890
            },
            ('Inflow', 'upper'): {
                'H': 0.6600, 'A': 0.2760, 'B': 1.2250, 'C': 0.4990,
                'D': 0.4130, 'E': 0.3510, 'F': 0.5930, 'G': 0.2790
            },
            ('Outflow', 'stat'): {
                'H': 0.1280, 'A': 0.3440, 'B': 0.4090, 'C': 0.2190,
                'D': 0.1180, 'E': 0.1260, 'F': 0.3000, 'G': 0.1240
            },
            ('Outflow', 'upper'): {
                'H': 0.3150, 'A': 0.5710, 'B': 0.5490, 'C': 0.4920,
                'D': 0.4790, 'E': 0.7710, 'F': 0.6370, 'G': 0.3070
            }
        })


@nottest
def make_dc_data():
    np.random.seed(0)

    dl_map = {
        'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4,
        'E': 0.1, 'F': 0.2, 'G': 0.3, 'H': 0.4,
    }

    index = pandas.MultiIndex.from_product([
        list('ABCDEFGH'),
        list('1234567'),
        ['GA', 'AL', 'OR', 'CA'],
        ['Inflow', 'Outflow', 'Reference']
    ], names=['param', 'bmp', 'state', 'loc'])

    array = np.random.lognormal(mean=0.75, sigma=1.25, size=len(index))
    data = pandas.DataFrame(data=array, index=index, columns=['res'])
    data['DL'] = data.apply(
        lambda r: dl_map.get(r.name[0]),
        axis=1
    )

    data['res'] = data.apply(
        lambda r: dl_map.get(r.name[0]) if r['res'] < r['DL'] else r['res'],
        axis=1
    )

    data['qual'] = data.apply(
        lambda r: 'ND' if r['res'] <= r['DL'] else '=',
        axis=1
    )

    return data
