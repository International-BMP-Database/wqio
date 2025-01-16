from functools import partial

import numpy
import probscale
import seaborn
from matplotlib import pyplot, ticker
from pandas.api.types import CategoricalDtype

from wqio import utils, validate


def rotate_tick_labels(ax, rotation, which, rotation_mode="anchor", ha="right"):
    """Rotates the ticklabels of a matplotlib Axes

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object that will be modified.
    rotation : float
        The amount of rotation, in degrees, to be applied to the labels.
    which : string
        The axis whose ticklabels will be rotated. Valid values are 'x',
        'y', or 'both'.
    rotation_mode : string, optional
        The rotation point for the ticklabels. Highly recommended to use
        the default value ('anchor').
    ha : string
        The horizontal alignment of the ticks. Again, recommended to use
        the default ('right').

    Returns
    -------
    None

    """

    if which == "both":
        rotate_tick_labels(ax, rotation, "x", rotation_mode=rotation_mode, ha=ha)
        rotate_tick_labels(ax, rotation, "y", rotation_mode=rotation_mode, ha=ha)
    else:
        if which == "x":
            axis = ax.xaxis

        elif which == "y":
            axis = ax.yaxis

        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


def log_formatter(use_1x=True, threshold=5):
    def _formatter(tick, pos=None, use_1x=True, threshold=3):
        """Formats log axes as `1 x 10^N` when N > 4 or N < -4."""

        if 10**threshold >= tick > 1:
            _tick = f"{int(tick):,d}"
        elif tick > 10**threshold or tick < 10 ** (-1 * threshold):
            int_log = int(numpy.log10(tick))
            _tick = rf"$1 \times 10 ^ {int_log:d}$" if use_1x else rf"$10 ^ {int_log:d}$"
        else:
            _tick = utils.sig_figs(tick, n=1)

        return str(_tick)

    func = partial(_formatter, use_1x=use_1x, threshold=threshold)
    return ticker.FuncFormatter(func)


def gridlines(ax, xlabel=None, ylabel=None, xscale=None, yscale=None, xminor=True, yminor=True):
    """Standard formatting for gridlines on a matplotlib Axes

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object that will be modified.
    xlabel, ylabel : string, optional
        The labels of the x- and y-axis.
    xscale, yscale : string, optional
        The scale of each axis. Can be 'linear', 'log', or 'prob'.
    xminor, yminor : bool, optional
        Toggles the grid on minor ticks. Has no effect if minor ticks
        are not present.

    Returns
    -------
    None

    """

    # set the scales
    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)

    # set the labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # major grids
    ax.yaxis.grid(True, which="major", ls="-", alpha=0.35)
    ax.xaxis.grid(True, which="major", ls="-", alpha=0.35)

    # minor grids
    if xminor:
        ax.xaxis.grid(True, which="minor", ls="-", alpha=0.17)

    if yminor:
        ax.yaxis.grid(True, which="minor", ls="-", alpha=0.17)


def one2one_line(ax, set_limits=True, set_aspect=True, **kwargs):
    label = kwargs.pop("label", "1:1 Line")
    axis_limits = [
        numpy.min([ax.get_xlim(), ax.get_ylim()]),
        numpy.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    if set_limits:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
    if set_aspect:
        ax.set_aspect("equal")

    return ax.plot(axis_limits, axis_limits, label=label, **kwargs)


def jointplot(
    x=None,
    y=None,
    data=None,
    xlabel=None,
    ylabel=None,
    color=None,
    zeromin=True,
    one2one=True,
):
    """Plots the joint distribution of two variables via seaborn

    Parameters
    ----------
    x, y : array-like or string
        Sequences of values or column names found within ``data``.
    data : pandas DataFrame or None, optional
        An optional DataFrame containing the data.
    xlabel, ylabel : string, optional
        Overrides the default x- and y-axis labels.
    color : matplotlib color, optional
        Color used for the plot elements.
    zeromin : bool, optional
        When True (default), force lower axes limits to 0.
    one2one : bool, optional
        When True (default), plots the 1:1 line on the axis and sets
        the x- and y-axis limits to be equal.

    Returns
    -------
    jg : seaborn.JointGrid

    """
    jg = seaborn.jointplot(x=x, y=y, color=color, data=data, marginal_kws=dict(rug=True, kde=True))

    if xlabel is None:
        xlabel = jg.ax_joint.get_xlabel()

    if ylabel is None:
        ylabel = jg.ax_joint.get_ylabel()

    jg.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

    if zeromin:
        jg.ax_joint.set_xlim(left=0)
        jg.ax_joint.set_ylim(bottom=0)

    if one2one:
        ax_limit_min = numpy.min([jg.ax_joint.get_xlim(), jg.ax_joint.get_ylim()])
        ax_limit_max = numpy.max([jg.ax_joint.get_xlim(), jg.ax_joint.get_ylim()])
        jg.ax_joint.set_xlim(left=ax_limit_min, right=ax_limit_max)
        jg.ax_joint.set_ylim(bottom=ax_limit_min, top=ax_limit_max)
        jg.ax_joint.plot(
            [ax_limit_min, ax_limit_max],
            [ax_limit_min, ax_limit_max],
            marker="None",
            linestyle="-",
            linewidth=1.75,
            color=color or "k",
            alpha=0.45,
            label="1:1 line",
        )

        jg.ax_joint.legend(frameon=False, loc="upper left")

    return jg


def whiskers_and_fliers(x, q1=None, q3=None, flierfactor=1.5, transformout=None):
    """Computes extent of whiskers and fliers on optionally transformed
    data for box and whisker plots.

    Parameters
    ----------
    x : array-like
        Sequence of optionally transformed data.
    q1, q3 : floats, optional
        First and third quartiles of the optionally transformed data.
    flierfactor : float, optional (default=1.5)
        multiplier by which outliers are compared to the IQR
    transformout : callable, optional
        Function to un-transform the results back into the original
        space of the data.

    Returns
    -------
    whiskers_and_fliers : dict
        Dictionary of whisker and fliers values.

    Examples
    --------
    >>> x = numpy.random.lognormal(size=37)
    >>> whisk_fly = whiskers_and_fliers(numpy.log(x), transformout=numpy.exp)

    See also
    --------
    wqio.utils.figutils.boxplot

    """
    wnf = {}
    if transformout is None:

        def transformout(x):
            return x

    if q1 is None:
        q1 = numpy.percentile(x, 25)

    if q3 is None:
        q3 = numpy.percentile(x, 75)

    iqr = q3 - q1
    # get low extreme
    loval = q1 - (flierfactor * iqr)
    whislo = numpy.compress(x >= loval, x)
    whislo = q1 if len(whislo) == 0 or numpy.min(whislo) > q1 else numpy.min(whislo)

    # get high extreme
    hival = q3 + (flierfactor * iqr)
    whishi = numpy.compress(x <= hival, x)
    whishi = q3 if len(whishi) == 0 or numpy.max(whishi) < q3 else numpy.max(whishi)

    wnf["fliers"] = numpy.hstack(
        [
            transformout(numpy.compress(x < whislo, x)),
            transformout(numpy.compress(x > whishi, x)),
        ]
    )
    wnf["whishi"] = transformout(whishi)
    wnf["whislo"] = transformout(whislo)

    return wnf


def boxplot(
    boxplot_stats,
    ax=None,
    position=None,
    width=0.8,
    shownotches=True,
    color="b",
    marker="o",
    patch_artist=True,
    showmean=False,
):
    """
    Draws a boxplot on an axes

    Parameters
    ----------
    boxplot_stats : list of dicts
        List of matplotlib boxplot-compatible statistics to be plotted
    ax : matplotlib Axes, optional
        The axis on which the boxplot will be drawn.
    position : int or list of int, optional
        Location on the x-axis where the boxplot will be drawn.
    width : float, optional (default = 0.8)
        Width of the boxplots.
    shownotches : bool, optional (default = True)
        Toggles notched boxplots where the notches show a confidence
        interval around the mediand.
    color : string, optional (default = 'b')
        Matplotlib color used to plot the outliers, median or box, and
        the optional mean.
    marker : str, optional (default = 'o')
        Matplotlib marker used for the the outliers and optional mean.
    patch_artist : bool, optional (default = True)
        Toggles drawing the boxes as a patch filled in with ``color``
        and a black median or as a black where the median is drawn
        in the ``color``.
    showmean : bool, optional (default = False)
        Toggles inclusion of the means in the boxplots.

    Returns
    -------
    bp : dictionary of matplotlib artists
        The graphical elements of the boxplot.

    """

    fig, ax = validate.axes(ax)

    if position is None:
        position = numpy.arange(len(boxplot_stats)) + 1
    elif numpy.isscalar(position):
        position = [position]

    meanprops = dict(marker=marker, markersize=6, markerfacecolor=color, markeredgecolor="Black")

    flierprops = dict(
        marker=marker,
        markersize=4,
        zorder=4,
        markerfacecolor="none",
        markeredgecolor=color,
        alpha=1,
    )

    whiskerprops = dict(linestyle="-", color="k", linewidth=0.75, zorder=4)

    if patch_artist:
        medianprops = dict(linewidth=1.00, color="k", linestyle="-", zorder=5)

        boxprops = dict(edgecolor="k", facecolor=color, linewidth=0.75, zorder=4, alpha=0.5)
    else:
        medianprops = dict(linewidth=1.00, color=color, linestyle="-", zorder=3)

        boxprops = dict(color="k", linewidth=0.75, zorder=4)

    bp = ax.bxp(
        boxplot_stats,
        positions=position,
        widths=width,
        showmeans=showmean,
        meanprops=meanprops,
        flierprops=flierprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        boxprops=boxprops,
        shownotches=shownotches,
        showcaps=False,
        manage_ticks=False,
        patch_artist=patch_artist,
    )

    return bp


def probplot(
    data,
    ax=None,
    axtype="prob",
    yscale="log",
    xlabel=None,
    ylabel=None,
    bestfit=False,
    scatter_kws=None,
    line_kws=None,
    return_results=False,
):
    """Probability, percentile, and quantile plots.

    Parameters
    ----------
    data : sequence or array-like
        1-dimensional data to be plotted
    ax : optional matplotlib axes object or None (default).
        The Axes on which to plot. If None is provided, one will be
        created.
    axtype : string (default = 'pp')
        Type of plot to be created. Options are:
            - 'prob': probabilty plot
            - 'pp': percentile plot
            - 'qq': quantile plot
    yscale : string (default = 'log')
        Scale for the y-axis. Use 'log' for logarithmic (default) or
        'linear'.
    xlabel, ylabel : string or None (default)
        Axis labels for the plot.
    bestfit : bool, optional (default is False)
        Specifies whether a best-fit line should be added to the
        plot.
    scatter_kws, line_kws : dictionary
        Dictionary of keyword arguments passed directly to `pyplot.plot`
        when drawing the scatter points and best-fit line, respectively.
    return_results : bool (default = False)
        If True a dictionary of results of is returned along with the
        figure. Keys are:
            q - array of quantiles
            x, y - arrays of data passed to function
            xhat, yhat - arrays of modeled data plotted in best-fit line
            res - a statsmodels Result object.

    Returns
    -------
    fig : matplotlib.Figure
    result : dictionary of linear fit results.

    """
    output = probscale.viz.probplot(
        data,
        ax=ax,
        plottype=axtype,
        probax="x",
        datalabel=ylabel,
        problabel=xlabel,
        datascale=yscale,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        bestfit=bestfit,
        return_best_fit_results=return_results,
    )
    return output


def _connect_spines(left_ax, right_ax, left_y, right_y, linestyle="solid", **line_kwds):
    """Connects the y-spines between two Axes

    Parameters
    ----------
    left_ax, right_ax : matplotlib Axes objects
        The Axes that need to be connected.
    left_y, right_y : float
        Values on the spines that wil be connected.
    linestyle : string, optional (default = 'solid')
        The line style to use. Valid values are 'solid', 'dashed',
        'dashdot', 'dotted'.
    **line_kwds : keyword arguments
        Additional options for style the line.

    Returns
    -------
    connector : BboxConnector
        The weird mpl-line-like-thingy that connects the spines.

    """

    import matplotlib.transforms as mtrans
    import mpl_toolkits.axes_grid1.inset_locator as inset

    left_trans = mtrans.blended_transform_factory(left_ax.transData, left_ax.transAxes)
    right_trans = mtrans.blended_transform_factory(right_ax.transData, right_ax.transAxes)

    left_data_trans = left_ax.transScale + left_ax.transLimits
    right_data_trans = right_ax.transScale + right_ax.transLimits

    left_pos = left_data_trans.transform((0, left_y))[1]
    right_pos = right_data_trans.transform((0, right_y))[1]

    bbox = mtrans.Bbox.from_extents(0, left_pos, 0, right_pos)
    right_bbox = mtrans.TransformedBbox(bbox, right_trans)
    left_bbox = mtrans.TransformedBbox(bbox, left_trans)

    # deal with the linestyle
    connector = inset.BboxConnector(
        left_bbox, right_bbox, loc1=3, loc2=2, linestyle=linestyle, **line_kwds
    )
    connector.set_clip_on(False)
    left_ax.add_artist(connector)

    return connector


def parallel_coordinates(dataframe, hue, cols=None, palette=None, showlegend=True, **subplot_kws):
    """Produce a parallel coordinates plot from a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data to be plotted.
    hue : string
        The column used to the determine assign the lines' colors.
    cols : list of strings, optional
        The non-hue columns to include. If None, all other columns are
        used.
    palette : string, optional
        Name of the seaborn color palette to use.
    showlegend : bool (default = True)
        Toggles including a legend on the plot.
    **subplot_kws : keyword arguments
        Options passed directly to pyplot.subplots()

    Returns
    -------
    fig : matplotlib Figure

    """

    # get the (non-hue) columns to plot
    if cols is None:
        cols = dataframe.columns.tolist()
        cols.remove(hue)

    # subset the data, putting the hue column last
    # python 3.5: data = dataframe[[*cols, hue]]
    data = dataframe[[*cols, hue]]

    # these plots look ridiculous in anything other than 'ticks'
    with seaborn.axes_style("ticks"):
        fig, axes = pyplot.subplots(ncols=len(cols), **subplot_kws)
        hue_vals = dataframe[hue].unique()
        colors = seaborn.color_palette(palette=palette, n_colors=len(hue_vals))
        color_dict = {}
        lines = []
        for h, c in zip(hue_vals, colors):
            lines.append(pyplot.Line2D([0], [0], linestyle="-", color=c, label=h))
            color_dict[h] = c

        for col, ax in zip(cols, axes):
            data_limits = [(0, dataframe[col].min()), (0, dataframe[col].max())]
            ax.set_xticks([0])
            ax.update_datalim(data_limits)
            ax.set_xticklabels([col])
            ax.autoscale(axis="y")
            ax.tick_params(axis="y", direction="inout")
            ax.tick_params(axis="x", direction="in")

        for row in data.values:
            for n, (ax1, ax2) in enumerate(zip(axes[:-1], axes[1:])):
                _connect_spines(ax1, ax2, row[n], row[n + 1], color=color_dict[row[-1]])

    if showlegend:
        fig.legend(lines, hue_vals)

    fig.subplots_adjust(wspace=0)
    seaborn.despine(fig=fig, bottom=True, trim=True)
    return fig


def categorical_histogram(df, valuecol, bins, classifier=None, **factoropts):
    """Plot a faceted, categorical histogram.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of storm information such as precipitation depth,
        duration, presence of outflow, flow volume, etc.
    valuecol : str
        The name of the column that should be categorized and plotted.
    bins : array-like
        The right-edges of the histogram bins.
    classifier : callable, optional
        Function-like object that classifies the values in ``valuecol``.
        Should accept a float scalar and return a string.
    factoropts : keyword arguments, optional
        Options passed directly to seaborn.factorplot

    Returns
    -------
    fig : seaborn.FacetGrid

    See also
    --------
    seaborn.factorplot

    """

    def format_col(colname):
        return colname.replace("_", " ").title()

    def process_column(colname):
        if colname is not None:
            return format_col(colname)

    if classifier is None:
        classifier = partial(utils.classifier, bins=bins, units="mm")

    cats = utils.unique_categories(classifier, bins)
    cat_type = CategoricalDtype(cats, ordered=True)
    aspect = factoropts.pop("aspect", 1.6)

    display_col = format_col(valuecol)

    processed_opts = dict(
        row=process_column(factoropts.pop("row", None)),
        col=process_column(factoropts.pop("col", None)),
        hue=process_column(factoropts.pop("hue", None)),
        kind="count",
        aspect=aspect,
        sharex=True,
    )

    final_opts = {**factoropts, **processed_opts}

    fig = (
        df.assign(display=df[valuecol].apply(classifier).astype(cat_type))
        .drop([valuecol], axis=1)
        .rename(columns={"display": valuecol})
        .rename(columns=lambda c: format_col(c))
        .pipe((seaborn.catplot, "data"), x=display_col, **final_opts)
        .set_ylabels("Occurences")
    )

    return fig
