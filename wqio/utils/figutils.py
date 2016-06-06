import numpy
import matplotlib
from matplotlib import pyplot


@numpy.deprecate
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
    ax.yaxis.grid(True, which='major', ls='-', alpha=0.35)
    ax.xaxis.grid(True, which='major', ls='-', alpha=0.35)

    # minor grids
    if xminor:
        ax.xaxis.grid(True, which='minor', ls='-', alpha=0.17)

    if yminor:
        ax.yaxis.grid(True, which='minor', ls='-', alpha=0.17)


@numpy.deprecate
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
    reg_index = numpy.linspace(start, stop, 256)

    # shifted index to match the data
    shift_index = numpy.hstack([
        numpy.linspace(0.0, midpoint, 128, endpoint=False),
        numpy.linspace(midpoint, 1.0, 128, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    pyplot.register_cmap(cmap=newcmap)

    return newcmap
