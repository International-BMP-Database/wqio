import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import seaborn.apionly as seaborn

from . import numutils


def _check_ax(ax):
    """ Checks if a value if an Axes. If None, a new one is created.

    Parameters
    ----------
    ax : matplotlib.Axes or None
        The value to be tested.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    """

    if ax is None:
        fig, ax = plt.subplots()
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    return fig, ax


def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='right'):
    """ Rotates the ticklabels of a matplotlib Axes

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

    if which =='both':
        rotateTickLabels(ax, rotation, 'x', rotation_mode=rotation_mode, ha=ha)
        rotateTickLabels(ax, rotation, 'y', rotation_mode=rotation_mode, ha=ha)
    else:
        if which == 'x':
            axis = ax.xaxis

        elif which == 'y':
            axis = ax.yaxis

        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


def setProbLimits(ax, N, which):
    """ Sets the limits of a probabilty axis based the number of point.

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object that will be modified.
    N : int
        Maximum number of points for the series plotted on the Axes.
    which : string
        The axis whose ticklabels will be rotated. Valid values are 'x',
        'y', or 'both'.

    Returns
    -------
    None

    """

    minval = 10 ** (-1 *np.ceil(np.log10(N) - 2))
    if which in ['x', 'both']:
        ax.set_xlim(left=minval, right=100-minval)
    elif which in ['y', 'both']:
        ax.set_ylim(bottom=minval, top=100-minval)


def gridlines(ax, xlabel=None, ylabel=None, xscale=None, yscale=None,
              xminor=True, yminor=True):
    """ Standard formatting for gridlines on a matplotlib Axes

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

    formatter = mticker.FuncFormatter(logLabelFormatter)

    # set the scales
    if xscale is not None:
        ax.set_xscale(xscale)
        if xscale == 'log':
            ax.xaxis.set_major_formatter(formatter)

    if yscale is not None:
        ax.set_yscale(yscale)
        if yscale == 'log':
            ax.yaxis.set_major_formatter(formatter)

    # set the labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # major grids
    ax.yaxis.grid(True, which='major', ls='-', alpha=0.35)
    ax.xaxis.grid(True, which='major', ls='-', alpha=0.35)

    # minor grids
    if xminor:
        ax.xaxis.grid(True, which='minor', ls='-', alpha=0.17)

    if yminor:
        ax.yaxis.grid(True, which='minor', ls='-', alpha=0.17)


def jointplot(x=None, y=None, data=None, xlabel=None, ylabel=None,
              color=None, zeromin=True, one2one=True):
    """ Plots the joint distribution of two variables via seaborn

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
    jg = seaborn.jointplot(x=x, y=y, color=color, data=data,
                           marginal_kws=dict(rug=True, kde=True))

    if xlabel is None:
        xlabel = jg.ax_joint.get_xlabel()

    if ylabel is None:
        ylabel = jg.ax_joint.get_ylabel()

    jg.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

    if zeromin:
        jg.ax_joint.set_xlim(left=0)
        jg.ax_joint.set_ylim(bottom=0)

    if one2one:
        ax_limit_max = np.max([jg.ax_joint.get_xlim(), jg.ax_joint.get_ylim()])
        jg.ax_joint.set_xlim(left=0, right=ax_limit_max)
        jg.ax_joint.set_ylim(bottom=0, top=ax_limit_max)
        jg.ax_joint.plot([0, ax_limit_max], [0, ax_limit_max], marker='None',
                         linestyle='-', linewidth=1.75, color=color or 'k',
                         alpha=0.45, label='1:1 line')

        jg.ax_joint.legend(frameon=False, loc='upper left')

    return jg


def scatterHistogram(*args, **kwargs):
    """ No longer implemented
    """
    raise NotImplementedError('This gone. Use jointplot instead.')


def boxplot(ax, data, position, median=None, CIs=None,
            mean=None, meanmarker='o', meancolor='b'):
    """ Draws a boxplot on an axes

    Parameters
    ----------
    ax : matplotlib Axes
        The axis on which the boxplot will be drawn.
    data : array-like
        The data to be summarized.
    position : int
        Location on the x-axis where the boxplot will be drawn.
    median : float, optional
        Custom value for the median.
    CIs : array-like, length = 2
        Custom confidence intervals around the median for notched plots.
    mean : float, optional
        Custom value for the mean of the data (e.g., geomeans for
        bacteria data).
    meanmarker : string, optional (default = 'o')
        matplotlib symbol used to plot the mean.
    meancolor : string, optional (default = 'b')
        matplotlib color used to plot the mean.

    Returns
    -------
    bp : dictionary of matplotlib artists
        The graphical elements of the boxplot.

    """

    if median is not None and np.isscalar(median):
        median = [median]

    if CIs is not None:
        if not isinstance(CIs, np.ndarray):
            CIs = np.array(CIs)

        if len(CIs.shape) == 1:
            CIs = CIs.reshape(1,2)

        CIs = CIs.tolist()

    # plot the data
    bp = ax.boxplot(data, notch=1, positions=[position],
                      widths=0.6, usermedians=median, conf_intervals=CIs)

    # plot the mean value as a big marker
    if mean is not None:
        # (geomean for bacteria data)
        ax.plot(position, mean, marker=meanmarker, markersize=4,
                  markerfacecolor=meancolor, markeredgecolor='k')

    return bp


def formatBoxplot(bp, color='b', marker='o', markersize=4, linestyle='-',
                  showcaps=False, patch_artist=False):
    """ Easily format a boxplots for stylistic consistency

    Parameters
    -----------
    bp : dictionary of matplotlib artists
        Graphical elements of the boxplot as returned by `plt.boxplot`
        or `figutils.boxplot`.
    color : string or RGB tuple
        Any valid matplotlib color string or RGB-spec for the median
        lines and outlier markers.
    marker : string
        Any valid matplotlib marker string for the outliers.
    markersize : int
        Size (in points) of the outlier marker.
    linestyle : string
        Any valid matplotlib linestyle string for the medians
    showcaps : bool (default = False)
        Toggles the drawing of caps (short horizontal lines) on the
        whiskers.
    patch_artist : bool (default = False)
        Toggles the use of patch artist instead of a line artists for
        the boxes.

    Returns
    -------
    None

    """

    if patch_artist:
        # format the box itself
        for box in bp['boxes']:
            plt.setp(box, edgecolor='k', facecolor=color, linewidth=0.75, zorder=4, alpha=0.5)

        # format the medians
        for med in bp['medians']:
            plt.setp(med, linewidth=1.00, color='k', linestyle=linestyle, zorder=5)

    else:
        # format the box itself
        for box in bp['boxes']:
            plt.setp(box, color='k', linewidth=0.75, zorder=4)

        # format the medians
        for med in bp['medians']:
            plt.setp(med, linewidth=1.00, color=color, linestyle=linestyle, zorder=3)

    # format the whiskers
    for whi in bp['whiskers']:
        plt.setp(whi, linestyle='-', color='k', linewidth=0.75, zorder=4)

    # format the caps on top of the whiskers
    for cap in bp['caps']:
        plt.setp(cap, color='k', linewidth=0.75, zorder=4, alpha=int(showcaps))

    # format the outliers
    for fli in bp['fliers']:
        plt.setp(fli, marker=marker, markersize=markersize, zorder=4,
                 markerfacecolor='none', markeredgecolor=color, alpha=1)


def probplot(data, ax=None, axtype='prob', yscale='log',
             xlabel=None, ylabel=None, bestfit=False,
             scatter_kws=None, line_kws=None, return_results=False):
    """ Probability, percentile, and quantile plots.

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
        Dictionary of keyword arguments passed directly to `plt.plot`
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

    scatter_kws = {} if scatter_kws is None else scatter_kws
    line_kws = {} if line_kws is None else line_kws

    fig, ax = _check_ax(ax)
    if axtype not in ['pp', 'qq', 'prob']:
        raise ValueError("invalid axtype: {}".format(axtype))

    qntls, ranked = stats.probplot(data, fit=False)
    if axtype == 'qq':
        xvalues = qntls
    else:
        xvalues = stats.norm.cdf(qntls) * 100

    # plot the final ROS data versus the Z-scores
    linestyle = scatter_kws.pop('linestyle', 'none')
    marker = scatter_kws.pop('marker', 'o')
    ax.plot(xvalues, ranked, linestyle=linestyle, marker=marker, **scatter_kws)

    ax.set_yscale(yscale)
    if yscale == 'log':
        fitlogs = 'y'
    else:
        fitlogs = None

    if axtype == 'prob':
        fitprobs = 'x'
        ax.set_xscale('prob')
    else:
        fitprobs = None
        #left, right = ax.get_xlim()
        #ax.set_xlim(left=left*0.96, right=right*1.04)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if bestfit:
        xhat, yhat, modelres = numutils.fit_line(xvalues, ranked, fitprobs=fitprobs,
                                                 fitlogs=fitlogs)
        ax.plot(xhat, yhat, **line_kws)
    else:
        xhat, yhat, modelres = (None, None, None)

    if return_results:
        return fig, dict(q=qntls, x=xvalues, y=ranked, xhat=xhat, yhat=yhat, res=modelres)
    else:
        return fig


def logLabelFormatter(tick, pos=None):
    """ Formats log axes as `1 x 10^N` when N > 4 or N < -4. """

    if 10**3 >= tick > 1:
        tick = int(tick)
    elif tick > 10**3 or tick < 10**-3:
        tick = r'$1 \times 10 ^ {%d}$' % int(np.log10(tick))

    return tick


def alt_logLabelFormatter(tick, pos=None):
    """ Formats log axes as `10^N` when N > 4 or N < -4. """

    if 10**3 >= tick > 1:
        tick = int(tick)
    elif tick > 10**3 or tick < 10**-3:
        tick = r'$10 ^ {%d}$' % int(np.log10(tick))

    return tick


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """ Offset the "center" of a colormap.

    Useful for data with a negative min and positive max and you want
    the middle of the colormap's dynamic range to be at zero.

    Parameters
    ----------
    cmap : The matplotlib colormap
        The colormap be altered
    start : float, optional
        Offset from lowest point in the colormap's range. Defaults to
        0.0 (no lower ofset). Should be between 0.0 and `midpoint`.
    midpoint : float, optional
        The new center of the colormap. Defaults to 0.5 (no shift).
        Should be between `start` and `stop`. In general, this should be
        1 - vmax/(vmax + abs(vmin)) For example if your data range from
        -15.0 to +5.0 and you want the center of the colormap at 0.0,
        `midpoint` should be set to  1 - 5/(5 + 15)) or 0.75.
    stop : float, optional
        Offset from highets point in the colormap's range. Defaults to
        1.0 (no upper ofset). Should be between `midpoint` and 1.0.
    name : string, optional
        Just a name for the colormap. Not particularly important.

    Returns
    -------
    newcmp : matplotlib.colors.LinearSegmentedColormap
        The modified color map.

    """

    if midpoint <= start or midpoint >= stop:
        raise ValueError("`midpoint` must be between `start` and `stop`")

    if start < 0 or start >= midpoint:
        raise ValueError("`start` must be between 0.0 and `midpoint`")

    if stop <= midpoint or stop > 1:
        raise ValueError("`stop` must be between `midpoint` and 1.0")

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 256)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 128, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def _connect_spines(left_ax, right_ax, left_y, right_y, linestyle='solid', **line_kwds):
    """ Connects the y-spines between two Axes

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
    connector = inset.BboxConnector(left_bbox, right_bbox, loc1=3, loc2=2,
                                    linestyle=linestyle, **line_kwds)
    connector.set_clip_on(False)
    left_ax.add_line(connector)

    return connector


def parallel_coordinates(dataframe, hue, cols=None, palette=None, showlegend=True, **subplot_kws):
    """ Produce a parallel coordinates plot from a dataframe.

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
        Options passed directly to plt.subplots()

    Returns
    -------
    fig : matplotlib Figure

    """

    # get the columsn to plot
    if cols is None:
        cols = dataframe.select(lambda c: c != hue, axis=1).columns.tolist()

    # subset the data
    final_cols = copy.copy(cols)
    final_cols.append(hue)
    data = dataframe[final_cols]

    # these plots look ridiculous in anything other than 'ticks'
    with seaborn.axes_style('ticks'):
        fig, axes = plt.subplots(ncols=len(cols), **subplot_kws)
        hue_vals = dataframe[hue].unique()
        colors = seaborn.color_palette(palette=palette, n_colors=len(hue_vals))
        color_dict = {}
        lines = []
        for h, c in zip(hue_vals, colors):
            lines.append(plt.Line2D([0,], [0], linestyle='-', color=c, label=h))
            color_dict[h] = c

        for col, ax in zip(cols, axes):
            data_limits =[(0, dataframe[col].min()), (0, dataframe[col].max())]
            ax.set_xticks([0])
            ax.update_datalim(data_limits)
            ax.set_xticklabels([col])
            ax.autoscale(axis='y')
            ax.tick_params(axis='y', direction='inout')
            ax.tick_params(axis='x', direction='in')

        for row in data.values:
            for n, (ax1, ax2) in enumerate(zip(axes[:-1], axes[1:])):
                line = _connect_spines(ax1, ax2, row[n], row[n+1], color=color_dict[row[-1]])

    if showlegend:
        fig.legend(lines, hue_vals)

    fig.subplots_adjust(wspace=0)
    seaborn.despine(fig=fig, bottom=True, trim=True)
    return fig
