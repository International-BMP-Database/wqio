import os

from nose.tools import *
import numpy as np
import numpy.testing as nptest

from wqio import testing
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex
import matplotlib.pyplot as plt
import seaborn.apionly as seaborn

import pandas
import pandas.util.testing as pdtest
from matplotlib.testing.decorators import image_comparison, cleanup

from wqio.core.features import (
    Parameter,
    Location,
    Dataset,
    DataCollection
)

from wqio import utils
import warnings


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

    def teardown(self):
        plt.close('all')


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


class _base_LocationMixin(object):
    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @nottest
    def main_setup(self):

        # basic test data
        self.tolerance = 0.05
        self.known_bsIter = 1500
        self.data = testing.getTestROSData()

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
        self.known_filtered_include = False

    def teardown(self):
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
        nptest.assert_allclose(
            self.loc.geomean_conf_interval,
            self.known_geomean_conf_interval,
            rtol=self.tolerance
        )

    def test_geostd(self):
        assert_true(hasattr(self.loc, 'geostd'))
        nptest.assert_allclose(
            self.loc.geostd,
            self.known_geostd,
            rtol=self.tolerance
        )

    def test_logmean(self):
        assert_true(hasattr(self.loc, 'logmean'))
        nptest.assert_allclose(
            self.loc.logmean,
            self.known_logmean,
            rtol=self.tolerance
        )

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
        nptest.assert_allclose(self.loc.median_conf_interval, self.known_median_conf_interval, rtol=self.tolerance*1.5)

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

    def test_pnorm(self):
        assert_true(hasattr(self.loc, 'pnorm'))
        nptest.assert_allclose(self.loc.pnorm, self.known_pnorm, rtol=self.tolerance)

    def test_plognorm(self):
        assert_true(hasattr(self.loc, 'plognorm'))
        nptest.assert_allclose(self.loc.plognorm, self.known_plognorm, rtol=self.tolerance)

    def test_shapiro(self):
        assert_true(hasattr(self.loc, 'shapiro'))
        nptest.assert_allclose(self.loc.shapiro, self.known_shapiro, rtol=self.tolerance)

    def test_shapiro_log(self):
        assert_true(hasattr(self.loc, 'shapiro_log'))
        nptest.assert_allclose(self.loc.shapiro_log, self.known_shapiro_log, rtol=self.tolerance)

    def test_lilliefors(self):
        assert_true(hasattr(self.loc, 'lilliefors'))
        nptest.assert_allclose(self.loc.lilliefors, self.known_lilliefors, rtol=self.tolerance)

    def test_lilliefors_log(self):
        assert_true(hasattr(self.loc, 'lilliefors_log'))
        nptest.assert_allclose(self.loc.lilliefors_log, self.known_lilliefors_log, rtol=self.tolerance)

    def test_anderson(self):
        assert_true(hasattr(self.loc, 'anderson'))
        nptest.assert_almost_equal(self.loc.anderson[0], self.known_anderson[0], decimal=5)
        nptest.assert_allclose(self.loc.anderson[1], self.known_anderson[1], rtol=self.tolerance)
        nptest.assert_allclose(self.loc.anderson[2], self.known_anderson[2], rtol=self.tolerance)

    def test_anderson_log(self):
        assert_true(hasattr(self.loc, 'anderson'))
        nptest.assert_almost_equal(self.loc.anderson_log[0], self.known_anderson_log[0], decimal=5)
        nptest.assert_allclose(self.loc.anderson_log[1], self.known_anderson_log[1], rtol=self.tolerance)
        nptest.assert_allclose(self.loc.anderson_log[2], self.known_anderson_log[2], rtol=self.tolerance)

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
        assert_true(isinstance(self.loc.data, np.ndarray))

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


class test_Location_ROS(_base_LocationMixin):
    def setup(self):
        self.main_setup()
        self.loc = Location(self.data, station_type='inflow', bsIter=self.known_bsIter,
                            rescol='res', qualcol='qual', useROS=True)

        # known statistics
        self.known_N = 35
        self.known_ND = 7
        self.known_fractionND = 0.2
        self.known_NUnique = 32
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
        self.known_pnorm = 0.00179254170507
        self.known_plognorm = 0.521462738514
        self.known_shapiro = [ 0.886889,  0.001789]
        self.known_shapiro_log = [ 0.972679,  0.520949]
        self.known_lilliefors = [ 0.18518 ,  0.003756]
        self.known_lilliefors_log = [ 0.091855,  0.635536]
        self.known_anderson = (1.543888, [ 0.527,  0.6  ,  0.719,  0.839,  0.998], [ 15. ,  10. ,   5. ,   2.5,   1. ])
        self.known_anderson_log = (0.30409633964188032, [ 0.527,  0.6  ,  0.719,  0.839,  0.998], [ 15. ,  10. ,   5. ,   2.5,   1. ])
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
        self.known_pnorm = 0.00323620648123
        self.known_plognorm = 0.306435495615
        self.known_shapiro = [ 0.896744,  0.003236]
        self.known_shapiro_log = [ 0.964298,  0.306435]
        self.known_lilliefors = [ 0.160353,  0.023078]
        self.known_lilliefors_log = [ 0.08148,  0.84545]
        self.known_anderson = (1.4392085, [ 0.527,  0.6  ,  0.719,  0.839,  0.998], [ 15. ,  10. ,   5. ,   2.5,   1. ])
        self.known_anderson_log = (0.3684061, [ 0.527,  0.6  ,  0.719,  0.839,  0.998], [ 15. ,  10. ,   5. ,   2.5,   1. ])
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


@nottest
def setup_location(station_type):
    data = testing.getTestROSData()
    np.random.seed(0)
    loc = Location(data, station_type=station_type, bsIter=10000,
                   rescol='res', qualcol='qual', useROS=True)
    plt.rcdefaults()
    return loc


@image_comparison(baseline_images=[
    'test_loc_boxplot_default',
    'test_loc_boxplot_patch_artists',
    'test_loc_boxplot_linscale',
    'test_loc_boxplot_no_mean',
    'test_loc_boxplot_width',
    'test_loc_boxplot_no_notch',
    'test_loc_boxplot_bacteria_geomean',
    'test_loc_boxplot_with_ylabel',
    'test_loc_boxplot_fallback_to_vert_scatter',
    'test_loc_boxplot_provided_ax',
    'test_loc_boxplot_custom_position',
    'test_loc_boxplot_with_xlabel',
], extensions=['png'])
def test_location_boxplot():
    xlims = {'left': 0, 'right': 2}
    loc = setup_location('inflow')
    loc.color = 'cornflowerblue'
    loc.plot_marker = 'o'
    fig1 = loc.boxplot()
    fig2 = loc.boxplot(patch_artist=True, xlims=xlims)
    fig3 = loc.boxplot(yscale='linear', xlims=xlims)

    loc.color = 'firebrick'
    loc.plot_marker = 'd'
    fig4 = loc.boxplot(showmean=False, xlims=xlims)
    fig5 = loc.boxplot(width=1.25, xlims=xlims)
    fig6 = loc.boxplot(notch=False, xlims=xlims)

    loc.color = 'forestgreen'
    loc.plot_marker = 's'
    fig7 = loc.boxplot(bacteria=True, xlims=xlims)
    fig8 = loc.boxplot(ylabel='Test Ylabel', xlims=xlims)
    fig9 = loc.boxplot(minpoints=np.inf, xlims=xlims)

    fig10, ax10 = plt.subplots()
    fig10 = loc.boxplot(ax=ax10, xlims=xlims)
    assert_true(isinstance(fig10, plt.Figure))
    assert_raises(ValueError, loc.boxplot, ax='junk')

    fig11 = loc.boxplot(pos=1.5, xlims=xlims)
    fig12 = loc.boxplot(xlabel='Test Xlabel', xlims=xlims)

@image_comparison(baseline_images=[
    'test_loc_probplot_default',
    'test_loc_probplot_provided_ax',
    'test_loc_probplot_yscale_linear',
    'test_loc_probplot_ppax',
    'test_loc_probplot_qqax',
    'test_loc_probplot_ylabel',
    'test_loc_probplot_clear_yticks',
    'test_loc_probplot_no_managegrid',
    'test_loc_probplot_no_rotate_xticklabels',
    'test_loc_probplot_no_set_xlims',
    'test_loc_probplot_plotopts1',
    'test_loc_probplot_plotopts2',
], extensions=['png'])
def test_location_probplot():
    loc = setup_location('inflow')
    loc.color = 'cornflowerblue'
    loc.plot_marker = 'o'

    fig1 = loc.probplot()
    fig2, ax2 = plt.subplots()
    fig2 = loc.probplot(ax=ax2)
    assert_true(isinstance(fig2, plt.Figure))
    assert_raises(ValueError, loc.probplot, ax='junk')

    fig3 = loc.probplot(yscale='linear')
    fig4 = loc.probplot(axtype='pp')
    fig5 = loc.probplot(axtype='qq')

    loc.color = 'firebrick'
    loc.plot_marker = 'd'
    fig6 = loc.probplot(ylabel='test ylabel')
    fig7 = loc.probplot(clearYLabels=True)
    fig8 = loc.probplot(managegrid=False)

    loc.color = 'forestgreen'
    loc.plot_marker = 'd'
    fig10 = loc.probplot(rotateticklabels=False)
    fig11 = loc.probplot(setxlimits=False)
    fig12 = loc.probplot(markersize=10, linestyle='--', color='blue', markerfacecolor='none', markeredgecolor='green')
    fig13 = loc.probplot(markeredgewidth=2, markerfacecolor='none', markeredgecolor='green')


@image_comparison(baseline_images=[
    'test_loc_statplot_custom_position',
    'test_loc_statplot_yscale_linear',
    'test_loc_statplot_no_notch',
    'test_loc_statplot_no_mean',
    'test_loc_statplot_custom_width',
    'test_loc_statplot_bacteria_true',
    'test_loc_statplot_ylabeled',
    'test_loc_statplot_qq',
    'test_loc_statplot_pp',
    'test_loc_statplot_patch_artist',
], extensions=['png'])
def test_location_statplot():
    loc = setup_location('inflow')
    loc.color = 'cornflowerblue'
    loc.plot_marker = 'o'

    fig1 = loc.statplot(pos=1.25)
    fig2 = loc.statplot(yscale='linear')
    fig3 = loc.statplot(notch=False)

    loc.color = 'firebrick'
    loc.plot_marker = 'd'
    fig4 = loc.statplot(showmean=False)
    fig5 = loc.statplot(width=1.5)
    fig6 = loc.statplot(bacteria=True)
    fig7 = loc.statplot(ylabel='Test Y-Label')

    loc.color = 'forestgreen'
    loc.plot_marker = 's'
    fig8 = loc.statplot(axtype='qq')
    fig9 = loc.statplot(axtype='pp')
    fig10 = loc.statplot(patch_artist=True)
    assert_true(fig10, plt.Figure)


@image_comparison(baseline_images=[
    'test_loc_vertical_scatter_default',
    'test_loc_vertical_scatter_provided_ax',
    'test_loc_vertical_scatter_pos',
    'test_loc_vertical_scatter_nojitter',
    'test_loc_vertical_scatter_alpha',
    'test_loc_vertical_scatter_ylabel',
    'test_loc_vertical_scatter_yscale_linear',
    'test_loc_vertical_scatter_not_ignoreROS',
    'test_loc_vertical_scatter_markersize',
], extensions=['png'])
def test_location_verticalScatter():
    xlims = {'left': 0, 'right': 2}
    loc = setup_location('inflow')
    loc.color = 'cornflowerblue'
    loc.plot_marker = 'o'

    fig1 = loc.verticalScatter()
    fig2, ax2 = plt.subplots()
    fig2 = loc.verticalScatter(ax=ax2, xlims=xlims)
    assert_true(isinstance(fig2, plt.Figure))
    assert_raises(ValueError, loc.verticalScatter, ax='junk', xlims=xlims)

    fig3 = loc.verticalScatter(pos=1.25, xlims=xlims)
    fig4 = loc.verticalScatter(jitter=0.0, xlims=xlims)
    fig5 = loc.verticalScatter(alpha=0.25, xlims=xlims)

    loc.color = 'firebrick'
    loc.plot_marker = 's'
    loc.verticalScatter(ylabel='Test Y-Label', xlims=xlims)
    loc.verticalScatter(yscale='linear', xlims=xlims)
    loc.verticalScatter(ignoreROS=False, xlims=xlims)
    loc.verticalScatter(markersize=8, xlims=xlims)


class test_Dataset(object):
    def setup(self):
        self.maxDiff = None

        # basic test data
        self.tolerance = 0.05
        self.known_bsIter = 750

        in_data = testing.getTestROSData()
        in_data['res'] += 3

        out_data = testing.getTestROSData()
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

        self.known_mannwhitney_stats = (9.270000e+02,   2.251523e-04)
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
        nptest.assert_allclose(self.ds.spearman_rho, self.known_spearman_rho, atol=0.0001)

    def test_spearman_p(self):
        assert_true(hasattr(self.ds, 'spearman_p'))
        nptest.assert_allclose(self.ds.spearman_p, self.known_spearman_p, atol=0.0001)

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
        nptest.assert_allclose(self.ds._spearman_stats, self.known_spearman_stats, atol=0.0001)

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

    def test_medianCIsOverlap(self):
        assert_equal(self.known_medianCIsOverlap, self.ds.medianCIsOverlap)

    def test__repr__normal(self):
        self.ds.__repr__

    def test_repr__None(self):
        self.ds.definition = None
        self.ds.__repr__


@nottest
def setup_dataset(extra_NDs=False):
    np.random.seed(0)
    in_data = testing.getTestROSData()
    in_data['res'] += 3

    out_data = testing.getTestROSData()
    out_data['res'] -= 1.5

    if extra_NDs:
        in_data.loc[[0, 1, 2], 'qual'] = 'ND'
        out_data.loc[[14, 15, 16], 'qual'] = 'ND'

    influent = Location(in_data, station_type='inflow', bsIter=10000,
                        rescol='res', qualcol='qual', useROS=False)

    effluent = Location(out_data, station_type='outflow', bsIter=10000,
                        rescol='res', qualcol='qual', useROS=False)

    ds = Dataset(influent, effluent, name='Test Dataset')
    plt.rcdefaults()
    return ds


@image_comparison(baseline_images=[
    'test_ds_boxplot_default',
    'test_ds_boxplot_patch_artists',
    'test_ds_boxplot_linscale',
    'test_ds_boxplot_no_mean',
    'test_ds_boxplot_width',
    'test_ds_boxplot_no_notch',
    'test_ds_boxplot_bacteria_geomean',
    'test_ds_boxplot_with_ylabel',
    'test_ds_boxplot_fallback_to_vert_scatter',
    'test_ds_boxplot_provided_ax',
    'test_ds_boxplot_custom_position',
    'test_ds_boxplot_custom_offset',
    'test_ds_boxplot_single_tick',
    'test_ds_boxplot_single_tick_no_name',
], extensions=['png'])
def test_dataset_boxplot():
    xlims = {'left': 0, 'right': 2}
    ds = setup_dataset()

    fig1 = ds.boxplot()
    fig2 = ds.boxplot(patch_artist=True, xlims=xlims)
    fig3 = ds.boxplot(yscale='linear', xlims=xlims)
    fig4 = ds.boxplot(showmean=False, xlims=xlims)
    fig5 = ds.boxplot(width=1.25, xlims=xlims)
    fig6 = ds.boxplot(notch=False, xlims=xlims)
    fig7 = ds.boxplot(bacteria=True, xlims=xlims)
    fig8 = ds.boxplot(ylabel='Test Ylabel', xlims=xlims)
    fig9 = ds.boxplot(minpoints=np.inf, xlims=xlims)

    fig10, ax10 = plt.subplots()
    fig10 = ds.boxplot(ax=ax10, xlims=xlims)
    assert_true(isinstance(fig10, plt.Figure))
    assert_raises(ValueError, ds.boxplot, ax='junk')

    fig11 = ds.boxplot(pos=1.5, xlims=xlims)
    fig12 = ds.boxplot(offset=0.75, xlims=xlims)
    fig13 = ds.boxplot(bothTicks=False, xlims=xlims)
    ds.name = None
    fig14 = ds.boxplot(bothTicks=False, xlims=xlims)

@image_comparison(baseline_images=[
    'test_ds_probplot_default',
    'test_ds_probplot_provided_ax',
    'test_ds_probplot_yscale_linear',
    'test_ds_probplot_ppax',
    'test_ds_probplot_qqax',
    'test_ds_probplot_ylabel',
    'test_ds_probplot_clear_yticks',
    'test_ds_probplot_no_managegrid',
    'test_ds_probplot_no_rotate_xticklabels',
    'test_ds_probplot_no_set_xlims',
], extensions=['png'])
def test_dataset_probplot():
    ds = setup_dataset()

    fig1 = ds.probplot()
    fig2, ax2 = plt.subplots()
    fig2 = ds.probplot(ax=ax2)
    assert_true(isinstance(fig2, plt.Figure))
    assert_raises(ValueError, ds.probplot, ax='junk')

    fig3 = ds.probplot(yscale='linear')
    fig4 = ds.probplot(axtype='pp')
    fig5 = ds.probplot(axtype='qq')
    fig6 = ds.probplot(ylabel='test ylabel')
    fig7 = ds.probplot(clearYLabels=True)
    fig8 = ds.probplot(managegrid=False)
    fig10 = ds.probplot(rotateticklabels=False)
    fig11 = ds.probplot(setxlimits=False)


@image_comparison(baseline_images=[
    'test_ds_statplot_custom_position',
    'test_ds_statplot_yscale_linear',
    'test_ds_statplot_no_notch',
    'test_ds_statplot_no_mean',
    'test_ds_statplot_custom_width',
    'test_ds_statplot_bacteria_true',
    'test_ds_statplot_ylabeled',
    'test_ds_statplot_qq',
    'test_ds_statplot_pp',
    'test_ds_statplot_patch_artist',
], extensions=['png'])
def test_dataset_statplot():
    ds = setup_dataset()

    fig1 = ds.statplot(pos=1.25)
    fig2 = ds.statplot(yscale='linear')
    fig3 = ds.statplot(notch=False)
    fig4 = ds.statplot(showmean=False)
    fig5 = ds.statplot(width=1.5)
    fig6 = ds.statplot(bacteria=True)
    fig7 = ds.statplot(ylabel='Test Y-Label')
    fig8 = ds.statplot(axtype='qq')
    fig9 = ds.statplot(axtype='pp')
    fig10 = ds.statplot(patch_artist=True)
    assert_true(fig10, plt.Figure)


@image_comparison(baseline_images=[
    'test_ds_scatterplot_default',
    'test_ds_scatterplot_provided_ax',
    'test_ds_scatterplot_xscale_linear',
    'test_ds_scatterplot_xyscale_linear',
    'test_ds_scatterplot_yscale_linear',
    'test_ds_scatterplot_xlabel',
    'test_ds_scatterplot_ylabel',
    'test_ds_scatterplot_no_xlabel',
    'test_ds_scatterplot_no_ylabel',
    'test_ds_scatterplot_no_legend',
    'test_ds_scatterplot_one2one',
    'test_ds_scatterplot_useROS',
], extensions=['png'])
def test_dataset_scatterplot():
    ds = setup_dataset(extra_NDs=True)

    fig1 = ds.scatterplot()
    fig2, ax2 = plt.subplots()
    fig2 = ds.scatterplot(ax=ax2)
    assert_true(isinstance(fig2, plt.Figure))
    assert_raises(ValueError, ds.scatterplot, ax='junk')

    fig3 = ds.scatterplot(xscale='linear')
    fig5 = ds.scatterplot(xscale='linear', yscale='linear')
    fig4 = ds.scatterplot(yscale='linear')
    fig6 = ds.scatterplot(xlabel='X-label')
    fig7 = ds.scatterplot(ylabel='Y-label')
    fig8 = ds.scatterplot(xlabel='')
    fig9 = ds.scatterplot(ylabel='')
    fig10 = ds.scatterplot(showlegend=False)
    fig10 = ds.scatterplot(one2one=True)
    fig11 = ds.scatterplot(useROS=True)


@image_comparison(baseline_images=[
    'test_ds__plot_NDs_both',
    'test_ds__plot_NDs_effluent',
    'test_ds__plot_NDs_influent',
    'test_ds__plot_NDs_neither',
], extensions=['png'])
def test_dataset__plot_NDs():
    ds = setup_dataset(extra_NDs=True)
    markerkwargs = dict(
        linestyle='none',
        markerfacecolor='black',
        markeredgecolor='white',
        markeredgewidth=0.5,
        markersize=6,
        zorder=10,
    )

    fig1, ax1 = plt.subplots()
    ds._plot_nds(ax1, which='both', marker='d', **markerkwargs)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 30)

    fig2, ax2 = plt.subplots()
    ds._plot_nds(ax2, which='effluent', marker='<', **markerkwargs)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 30)

    fig3, ax3 = plt.subplots()
    ds._plot_nds(ax3, which='influent', marker='v', **markerkwargs)
    ax3.set_xlim(0, 30)
    ax3.set_ylim(0, 30)

    fig4, ax4 = plt.subplots()
    ds._plot_nds(ax4, which='neither', marker='o', **markerkwargs)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(0, 30)


@image_comparison(baseline_images=[
    'test_ds_joint_hist',
    'test_ds_joint_kde',
    'test_ds_joint_rug',
    'test_ds_joint_kde_rug_hist',
], extensions=['png'])
def test_dataset_jointplot():
    ds = setup_dataset(extra_NDs=True)

    def do_jointplots(ds, hist=False, kde=False, rug=False):
        jg = ds.jointplot(hist=hist, kde=kde, rug=rug)
        assert_true(isinstance(jg, seaborn.JointGrid))
        return jg.fig

    fig1 = do_jointplots(ds, hist=True)
    fig2 = do_jointplots(ds, kde=True)
    fig3 = do_jointplots(ds, rug=True)
    fig4 = do_jointplots(ds, hist=True, kde=True, rug=True)


@nottest
def make_dc_data(ndval='ND', rescol='res', qualcol='qual'):
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
    data = pandas.DataFrame(data=array, index=index, columns=[rescol])
    data['DL'] = data.apply(
        lambda r: dl_map.get(r.name[0]),
        axis=1
    )

    data[rescol] = data.apply(
        lambda r: dl_map.get(r.name[0]) if r[rescol] < r['DL'] else r[rescol],
        axis=1
    )

    data[qualcol] = data.apply(
        lambda r: ndval if r[rescol] <= r['DL'] else '=',
        axis=1
    )

    return data


class _base_DataCollecionMixin(object):
    known_rescol = 'ros_res'
    known_raw_rescol = 'res'
    known_roscol = 'ros_res'
    known_qualcol = 'qual'
    known_stationcol = 'loc'
    known_paramcol = 'param'
    known_ndval = 'ND'

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
        assert_list_equal(self.dc.ndval, [self.known_ndval])

    def test_bsIter(self):
        assert_equal(self.dc.bsIter, self.known_bsIter)

    def test_groupby(self):
        assert_equal(self.dc.groupcols, self.known_groupcols)

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
        np.random.seed(0)
        pdtest.assert_frame_equal(
            np.round(self.dc.means, 3),
            self.known_means,
            check_names=False
        )

    def test_medians(self):
        np.random.seed(0)
        pdtest.assert_frame_equal(
            np.round(self.dc.medians, 3),
            self.known_medians,
            check_names=False
        )

    def test__generic_stat(self):
        np.random.seed(0)
        pdtest.assert_frame_equal(
            np.round(self.dc._generic_stat(np.min, bootstrap=False), 3),
            self.known_genericstat,
            check_names=False
        )


class test_DataCollection_baseline(_base_DataCollecionMixin):
    def setup(self):
        self.data = make_dc_data(ndval=self.known_ndval, rescol=self.known_raw_rescol,
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
                'F': 4.600, 'A': 7.089, 'B': 9.163, 'E': 5.612
            },
            ('Inflow', 'stat'): {
                'H': 3.252, 'C': 3.674, 'D': 4.241, 'G': 3.908,
                'F': 5.355, 'A': 5.882, 'B': 6.969, 'E': 3.877
            },
            ('Reference', 'lower'): {
                'H': 1.708, 'C': 1.817, 'D': 2.200, 'G': 1.826,
                'F': 1.477, 'A': 1.397, 'B': 2.575, 'E': 1.234
            },
            ('Reference', 'upper'): {
                'H': 3.411, 'C': 6.405, 'D': 5.602, 'G': 5.626,
                'F': 9.489, 'A': 3.870, 'B': 7.393, 'E': 4.047
            },
            ('Inflow', 'lower'): {
                'H': 1.907, 'C': 1.644, 'D': 1.786, 'G': 2.262,
                'F': 2.539, 'A': 2.767, 'B': 3.933, 'E': 2.422
            },
            ('Inflow', 'upper'): {
                'H': 4.694, 'C': 6.186, 'D':  7.300, 'G': 5.583,
                'F': 8.444, 'A': 9.462, 'B': 10.391, 'E': 5.547
            },
            ('Reference', 'stat'): {
                'H': 2.515, 'C': 3.892, 'D': 3.819, 'G': 3.619,
                'F': 4.988, 'A': 2.555, 'B': 4.919, 'E': 2.587
            },
            ('Outflow', 'stat'): {
                'H': 2.633, 'C': 3.327, 'D': 4.451, 'G': 2.758,
                'F': 3.515, 'A': 4.808, 'B': 6.297, 'E': 4.011
            },
            ('Outflow', 'lower'): {
                'H': 1.568, 'C': 2.346, 'D': 1.863, 'G': 1.643,
                'F': 2.477, 'A': 2.714, 'B': 3.608, 'E': 2.475
            }
        })

        self.known_medians = pandas.DataFrame({
            ('Outflow', 'upper'): {
                'G': 2.421, 'E': 3.512, 'H': 2.387, 'F': 4.227,
                'B': 5.167, 'C': 3.736, 'D': 1.930, 'A': 3.554
            },
            ('Inflow', 'lower'): {
                'G': 0.735, 'E': 1.311, 'H': 1.305, 'F': 1.178,
                'B': 1.691, 'C': 0.924, 'D': 0.766, 'A': 1.329
            },
            ('Outflow', 'lower'): {
                'G': 0.817, 'E': 1.317, 'H': 0.662, 'F': 1.500,
                'B': 0.889, 'C': 1.288, 'D': 0.877, 'A': 1.234
            },
            ('Reference', 'lower'): {
                'G': 1.137, 'E': 0.612, 'H': 0.976, 'F': 0.776,
                'B': 0.833, 'C': 0.938, 'D': 1.190, 'A': 0.691
            },
            ('Reference', 'stat'): {
                'G': 2.008, 'E': 1.211, 'H': 1.940, 'F': 1.473,
                'B': 2.118, 'C': 1.833, 'D': 2.187, 'A': 1.293
            },
            ('Inflow', 'upper'): {
                'G': 3.841, 'E': 4.113, 'H': 2.829, 'F': 4.345,
                'B': 6.327, 'C': 2.012, 'D': 2.915, 'A': 3.485
            },
            ('Reference', 'upper'): {
                'G': 2.513, 'E': 1.883, 'H': 2.846, 'F': 2.131,
                'B': 4.418, 'C': 2.893, 'D': 3.719, 'A': 2.051
            },
            ('Outflow', 'stat'): {
                'G': 1.559, 'E': 2.101, 'H': 1.604, 'F': 2.495,
                'B': 2.485, 'C': 3.075, 'D': 1.613, 'A': 2.500
            },
            ('Inflow', 'stat'): {
                'G': 1.844, 'E': 2.883, 'H': 1.729, 'F': 2.630,
                'B': 3.511, 'C': 1.410, 'D': 1.873, 'A': 2.830
            }
        })

        self.known_genericstat = pandas.DataFrame({
            ('Reference', self.known_roscol): {
                'D': 0.209, 'A': 0.119, 'B': 0.307, 'H': 0.221,
                'G': 0.189, 'F': 0.099, 'E': 0.211, 'C': 0.135,
            },
            ('Inflow', self.known_roscol): {
                'D': 0.107, 'A': 0.178, 'B': 0.433, 'H': 0.210,
                'G': 0.096, 'F': 0.157, 'E': 0.236, 'C': 0.116,
            },
            ('Outflow', self.known_roscol): {
                'D': 0.118, 'A': 0.344, 'B': 0.409, 'H': 0.128,
                'G': 0.124, 'F': 0.300, 'E': 0.126, 'C': 0.219,
            }
        })


class test_DataCollection_customNDval(test_DataCollection_baseline):
    known_raw_rescol = 'conc'
    known_roscol = 'ros_conc'
    known_rescol = 'ros_conc'
    known_qualcol = 'anote'
    known_stationcol = 'loc'
    known_paramcol = 'param'

