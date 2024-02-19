import numpy
import pytest
import seaborn
from matplotlib import pyplot

from wqio.features import Dataset, Location
from wqio.tests import helpers

BASELINE_IMAGES = "_baseline_images/features_tests"
TOLERANCE = helpers.get_img_tolerance()


@helpers.seed
def setup_location(station_type):
    data = helpers.getTestROSData()
    loc = Location(
        data,
        station_type=station_type,
        bsiter=10000,
        rescol="res",
        qualcol="qual",
        useros=True,
    )
    pyplot.rcdefaults()
    return loc


@pytest.fixture
def inflow_loc_blue():
    loc = setup_location("inflow")
    loc.color = "cornflowerblue"
    loc.plot_marker = "o"
    return loc


@pytest.fixture
def inflow_loc_red():
    loc = setup_location("inflow")
    loc.color = "firebrick"
    loc.plot_marker = "d"
    return loc


@pytest.fixture
def inflow_loc_green():
    loc = setup_location("inflow")
    loc.color = "forestgreen"
    loc.plot_marker = "d"
    return loc


@pytest.fixture
def inflow_xlims():
    return {"left": 0, "right": 2}


@pytest.fixture
def inflow_xlims_vs():
    return {"left": -0.5, "right": 0.5}


@pytest.fixture
def outflow_loc():
    loc = setup_location("outflow")
    return loc


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_default(inflow_loc_blue, inflow_xlims):
    fig1 = inflow_loc_blue.boxplot()
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_patch_artists(inflow_loc_blue, inflow_xlims):
    fig2 = inflow_loc_blue.boxplot(patch_artist=True, xlims=inflow_xlims)
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_linscale(inflow_loc_blue, inflow_xlims):
    fig3 = inflow_loc_blue.boxplot(yscale="linear", xlims=inflow_xlims)
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_no_mean(inflow_loc_red, inflow_xlims):
    fig4 = inflow_loc_red.boxplot(showmean=False, xlims=inflow_xlims)
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_width(inflow_loc_red, inflow_xlims):
    fig5 = inflow_loc_red.boxplot(width=1.25, xlims=inflow_xlims)
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_no_notch(inflow_loc_red, inflow_xlims):
    fig6 = inflow_loc_red.boxplot(shownotches=False, xlims=inflow_xlims)
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_bacteria_geomean(inflow_loc_green, inflow_xlims):
    fig7 = inflow_loc_green.boxplot(bacteria=True, xlims=inflow_xlims)
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_with_ylabel(inflow_loc_green, inflow_xlims):
    fig8 = inflow_loc_green.boxplot(ylabel="Test Ylabel", xlims=inflow_xlims)
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_provided_ax(inflow_loc_green, inflow_xlims):
    fig10, ax10 = pyplot.subplots()
    fig10 = inflow_loc_green.boxplot(ax=ax10, xlims=inflow_xlims)
    assert isinstance(fig10, pyplot.Figure)
    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_custom_position(inflow_loc_green, inflow_xlims):
    fig11 = inflow_loc_green.boxplot(pos=1.5, xlims=inflow_xlims)
    return fig11


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_boxplot_with_xlabel(inflow_loc_green, inflow_xlims):
    fig12 = inflow_loc_green.boxplot(xlabel="Test Xlabel", xlims=inflow_xlims)
    return fig12


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_default(inflow_loc_blue):
    fig1 = inflow_loc_blue.probplot()
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_provided_ax(inflow_loc_blue):
    fig2, ax2 = pyplot.subplots()
    fig2 = inflow_loc_blue.probplot(ax=ax2)
    assert isinstance(fig2, pyplot.Figure)
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_yscale_linear(inflow_loc_blue):
    fig3 = inflow_loc_blue.probplot(yscale="linear")
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_ppax(inflow_loc_blue):
    fig4 = inflow_loc_blue.probplot(axtype="pp")
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_qqax(inflow_loc_blue):
    fig5 = inflow_loc_blue.probplot(axtype="qq")
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_ylabel(inflow_loc_red):
    fig6 = inflow_loc_red.probplot(ylabel="test ylabel")
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_clear_yticks(inflow_loc_red):
    fig7 = inflow_loc_red.probplot(clearYLabels=True)
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot(inflow_loc_red):
    fig8 = inflow_loc_red.probplot()
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_no_rotate_xticklabels(inflow_loc_green):
    fig10 = inflow_loc_green.probplot(rotateticklabels=False)
    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_plotopts1(inflow_loc_green):
    fig12 = inflow_loc_green.probplot(
        markersize=10,
        linestyle="--",
        color="blue",
        markerfacecolor="none",
        markeredgecolor="green",
    )
    return fig12


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_probplot_plotopts2(inflow_loc_green):
    fig13 = inflow_loc_green.probplot(
        markeredgewidth=2, markerfacecolor="none", markeredgecolor="green"
    )
    return fig13


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_custom_position(inflow_loc_blue):
    fig1 = inflow_loc_blue.statplot(pos=1.25)
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_yscale_linear(inflow_loc_blue):
    fig2 = inflow_loc_blue.statplot(yscale="linear")
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_no_notch(inflow_loc_blue):
    fig3 = inflow_loc_blue.statplot(shownotches=False)
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_no_mean(inflow_loc_red):
    fig4 = inflow_loc_red.statplot(showmean=False)
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_custom_width(inflow_loc_red):
    fig5 = inflow_loc_red.statplot(width=1.5)
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_bacteria_true(inflow_loc_red):
    fig6 = inflow_loc_red.statplot(bacteria=True)
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_ylabeled(inflow_loc_red):
    fig7 = inflow_loc_red.statplot(ylabel="Test Y-Label")
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_qq(inflow_loc_green):
    fig8 = inflow_loc_green.statplot(axtype="qq")
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_pp(inflow_loc_green):
    fig9 = inflow_loc_green.statplot(axtype="pp")
    return fig9


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_statplot_patch_artist(inflow_loc_green):
    fig10 = inflow_loc_green.statplot(patch_artist=True)
    assert isinstance(fig10, pyplot.Figure)
    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_default(inflow_loc_blue):
    fig = inflow_loc_blue.verticalScatter()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_provided_ax(inflow_loc_blue):
    fig, ax2 = pyplot.subplots()
    fig = inflow_loc_blue.verticalScatter(ax=ax2)
    assert isinstance(fig, pyplot.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_pos(inflow_loc_blue):
    fig = inflow_loc_blue.verticalScatter(pos=1.25)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_ylabel(inflow_loc_red):
    fig = inflow_loc_red.verticalScatter(ylabel="Test Y-Label")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_yscale_linear(inflow_loc_red):
    fig = inflow_loc_red.verticalScatter(yscale="linear")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_not_ignoreROS(inflow_loc_red):
    fig = inflow_loc_red.verticalScatter(ignoreROS=False)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_loc_vertical_scatter_markersize(inflow_loc_red):
    fig = inflow_loc_red.verticalScatter(markersize=8)
    return fig


@helpers.seed
def setup_dataset(extra_NDs=False):
    in_data = helpers.getTestROSData()
    in_data["res"] += 3

    out_data = helpers.getTestROSData()
    out_data["res"] -= 1.5

    if extra_NDs:
        in_data.loc[[0, 1, 2], "qual"] = "ND"
        out_data.loc[[14, 15, 16], "qual"] = "ND"

    influent = Location(
        in_data,
        station_type="inflow",
        bsiter=10000,
        rescol="res",
        qualcol="qual",
        useros=False,
    )

    effluent = Location(
        out_data,
        station_type="outflow",
        bsiter=10000,
        rescol="res",
        qualcol="qual",
        useros=False,
    )

    ds = Dataset(influent, effluent, name="Test Dataset")
    pyplot.rcdefaults()
    return ds


@pytest.fixture
def ds_basic():
    return setup_dataset(extra_NDs=False)


@pytest.fixture
def ds_NDs():
    return setup_dataset(extra_NDs=True)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_default(ds_basic, inflow_xlims):
    fig1 = ds_basic.boxplot()
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_patch_artists(ds_basic, inflow_xlims):
    fig2 = ds_basic.boxplot(patch_artist=True, xlims=inflow_xlims)
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_linscale(ds_basic, inflow_xlims):
    fig3 = ds_basic.boxplot(yscale="linear", xlims=inflow_xlims)
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_no_mean(ds_basic, inflow_xlims):
    fig4 = ds_basic.boxplot(showmean=False, xlims=inflow_xlims)
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_width(ds_basic, inflow_xlims):
    fig5 = ds_basic.boxplot(width=1.25, xlims=inflow_xlims)
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_no_notch(ds_basic, inflow_xlims):
    fig6 = ds_basic.boxplot(shownotches=False, xlims=inflow_xlims)
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_bacteria_geomean(ds_basic, inflow_xlims):
    fig7 = ds_basic.boxplot(bacteria=True, xlims=inflow_xlims)
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_with_ylabel(ds_basic, inflow_xlims):
    fig8 = ds_basic.boxplot(ylabel="Test Ylabel", xlims=inflow_xlims)
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_provided_ax(ds_basic, inflow_xlims):
    fig10, ax10 = pyplot.subplots()
    fig10 = ds_basic.boxplot(ax=ax10, xlims=inflow_xlims)
    assert isinstance(fig10, pyplot.Figure)

    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_custom_position(ds_basic, inflow_xlims):
    fig11 = ds_basic.boxplot(pos=1.5, xlims=inflow_xlims)
    return fig11


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_custom_offset(ds_basic, inflow_xlims):
    fig12 = ds_basic.boxplot(offset=0.75, xlims=inflow_xlims)
    return fig12


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_single_tick(ds_basic, inflow_xlims):
    fig13 = ds_basic.boxplot(bothTicks=False, xlims=inflow_xlims)
    return fig13


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_boxplot_single_tick_no_name(ds_basic, inflow_xlims):
    ds_basic.name = None
    fig14 = ds_basic.boxplot(bothTicks=False, xlims=inflow_xlims)
    return fig14


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_default(ds_basic):
    fig1 = ds_basic.probplot()
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_provided_ax(ds_basic):
    fig2, ax2 = pyplot.subplots()
    fig2 = ds_basic.probplot(ax=ax2)
    assert isinstance(fig2, pyplot.Figure)
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_yscale_linear(ds_basic):
    fig3 = ds_basic.probplot(yscale="linear")
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_ppax(ds_basic):
    fig4 = ds_basic.probplot(axtype="pp")
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_qqax(ds_basic):
    fig5 = ds_basic.probplot(axtype="qq")
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_ylabel(ds_basic):
    fig6 = ds_basic.probplot(ylabel="test ylabel")
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_clear_yticks(ds_basic):
    fig7 = ds_basic.probplot(clearYLabels=True)
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot(ds_basic):
    fig8 = ds_basic.probplot()
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_probplot_no_rotate_xticklabels(ds_basic):
    fig10 = ds_basic.probplot(rotateticklabels=False)
    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_custom_position(ds_basic):
    fig1 = ds_basic.statplot(pos=1.25)
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_yscale_linear(ds_basic):
    fig2 = ds_basic.statplot(yscale="linear")
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_no_notch(ds_basic):
    fig3 = ds_basic.statplot(shownotches=False)
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_no_mean(ds_basic):
    fig4 = ds_basic.statplot(showmean=False)
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_custom_width(ds_basic):
    fig5 = ds_basic.statplot(width=1.5)
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_bacteria_true(ds_basic):
    fig6 = ds_basic.statplot(bacteria=True)
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_ylabeled(ds_basic):
    fig7 = ds_basic.statplot(ylabel="Test Y-Label")
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_qq(ds_basic):
    fig8 = ds_basic.statplot(axtype="qq")
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_pp(ds_basic):
    fig9 = ds_basic.statplot(axtype="pp")
    return fig9


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_statplot_patch_artist(ds_basic):
    fig10 = ds_basic.statplot(patch_artist=True)
    assert fig10, pyplot.Figure
    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_default(ds_NDs):
    fig1 = ds_NDs.scatterplot()
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_best_fit(ds_NDs):
    fig = ds_NDs.scatterplot(bestfit=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_best_fit_through_origin(ds_NDs):
    with numpy.errstate(divide="ignore", invalid="ignore"):
        fig = ds_NDs.scatterplot(bestfit=True, fitopts=dict(through_origin=True))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_provided_ax(ds_NDs):
    fig2, ax2 = pyplot.subplots()
    fig2 = ds_NDs.scatterplot(ax=ax2)
    assert isinstance(fig2, pyplot.Figure)

    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_xscale_linear(ds_NDs):
    fig3 = ds_NDs.scatterplot(xscale="linear")
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_xyscale_linear(ds_NDs):
    fig5 = ds_NDs.scatterplot(xscale="linear", yscale="linear")
    return fig5


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_yscale_linear(ds_NDs):
    fig4 = ds_NDs.scatterplot(yscale="linear")
    return fig4


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_xlabel(ds_NDs):
    fig6 = ds_NDs.scatterplot(xlabel="X-label")
    return fig6


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_ylabel(ds_NDs):
    fig7 = ds_NDs.scatterplot(ylabel="Y-label")
    return fig7


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_no_xlabel(ds_NDs):
    fig8 = ds_NDs.scatterplot(xlabel="")
    return fig8


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_no_ylabel(ds_NDs):
    fig9 = ds_NDs.scatterplot(ylabel="")
    return fig9


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_no_legend(ds_NDs):
    fig10 = ds_NDs.scatterplot(showlegend=False)
    return fig10


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds_scatterplot_one2one(ds_NDs):
    fig10 = ds_NDs.scatterplot(one2one=True)
    return fig10


def test_ds_scatterplot_useros(ds_NDs):
    with helpers.raises(ValueError):
        ds_NDs.scatterplot(useros=True)


@pytest.fixture
def markerkwargs():
    opts = dict(
        linestyle="none",
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=6,
        zorder=10,
    )
    return opts


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds__plot_NDs_both(ds_NDs, markerkwargs):
    fig1, ax1 = pyplot.subplots()
    ds_NDs._plot_nds(ax1, which="both", marker="d", **markerkwargs)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 30)
    return fig1


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds__plot_NDs_effluent(ds_NDs, markerkwargs):
    fig2, ax2 = pyplot.subplots()
    ds_NDs._plot_nds(ax2, which="effluent", marker="<", **markerkwargs)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 30)
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds__plot_NDs_influent(ds_NDs, markerkwargs):
    fig3, ax3 = pyplot.subplots()
    ds_NDs._plot_nds(ax3, which="influent", marker="v", **markerkwargs)
    ax3.set_xlim(0, 30)
    ax3.set_ylim(0, 30)
    return fig3


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=TOLERANCE)
def test_ds__plot_NDs_neither(ds_NDs, markerkwargs):
    fig4, ax4 = pyplot.subplots()
    ds_NDs._plot_nds(ax4, which="neither", marker="o", **markerkwargs)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(0, 30)
    return fig4


def _do_jointplots(ds, hist=False, kde=False, rug=False):
    jg = ds.jointplot(hist=hist, kde=kde, rug=rug)
    assert isinstance(jg, seaborn.JointGrid)
    return jg.figure


def test_ds_joint_hist_smoke(ds_NDs):
    with seaborn.axes_style("ticks"):
        fig1 = _do_jointplots(ds_NDs, hist=True)
    return fig1


def test_ds_joint_kde_smoke(ds_NDs):
    with seaborn.axes_style("ticks"):
        fig2 = _do_jointplots(ds_NDs, kde=True)
    return fig2


def test_ds_joint_rug_smoke(ds_NDs):
    with seaborn.axes_style("ticks"):
        fig3 = _do_jointplots(ds_NDs, rug=True)
        return fig3


def test_ds_joint_kde_rug_hist_smoke(ds_NDs):
    with seaborn.axes_style("ticks"):
        fig4 = _do_jointplots(ds_NDs, hist=True, kde=True)
    return fig4
