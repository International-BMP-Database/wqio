from __future__ import print_function, division

import sys

from six import StringIO
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy import stats
import statsmodels.api as sm

from wqio import testing


def addSecondColumnLevel(levelval, levelname, olddf):
    """ Add a second level to the column-index if a dataframe.

    Parameters
    ----------
    levelval : int or string
        Constant value to be assigned to the second level.
    levelname : string
        The name of the second level.
    olddf : pandas.DataFrame
        The original dataframe to be modified.

    Returns
    -------
    newdf : pandas.DataFrame
        The mutated dataframe with a MultiIndex in the columns.

    Example
    -------
    >>> df = pandas.DataFrame(columns=['res', 'qual'], index=range(3))
    >>> df.columns
    Index(['res', 'qual'], dtype='object')
    >>> df2 = utils.addSecondColumnLevel('Infl', 'location', df)
    >>> df2.columns
    MultiIndex(levels=[['Infl'], ['qual', 'res']],
               labels=[[0, 0], [1, 0]],
               names=['loc', 'quantity'])

    """

    if isinstance(olddf.columns, pandas.MultiIndex):
        raise ValueError('Dataframe already has MultiIndex on columns')

    origlevel = 'quantity'
    if olddf.columns.names[0] is not None:
        origlevel = olddf.columns.names[0]

    # define the index
    colarray = [[levelval]*len(olddf.columns), olddf.columns]
    colindex = pandas.MultiIndex.from_arrays(colarray)

    # copy the dataframe and redefine the columns
    newdf = olddf.copy()
    newdf.columns = colindex
    newdf.columns.names = [levelname, origlevel]
    return newdf


def getUniqueDataframeIndexVal(df, indexlevel):
    """ Confirms that a given level of a dataframe's index only has
    one unique value. Useful for confirming consistent units. Raises
    error if level is not a single value. Returns unique value of the
    index level.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe whose index will be inspected.
    indexlevel : int or string
        Level of the dataframe's index to be

    Returns
    -------
    uniqueval
        The unique value of the index.

    """

    index = np.unique(df.index.get_level_values(indexlevel).tolist())
    if index.shape != (1,):
        raise ValueError('index level "%s" is not unique!' % indexlevel)

    return index[0]


def _boxplot_legend(ax, notch=False, shrink=2.5, fontsize=10, showmean=False):
    """ Drawas an annotated a boxplot to serve as a legend.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes on which the legend will be drawn.
    notch : bool, optional (default = False)
        Whether or not to include notches (confidence intervals) around
        the median.
    shrink : float, optional (default = 2.5)
        Some measure of how far away the annotation arrows should be
        from the elements to which they point.
    fontsize : int, optional (default = 10)
        Fontsize of the annotation in points.
    showmean : bool, optonal (default = False)
        Toggles the display of the mean on the on the plot.

    Returns
    -------
    None

    Notes
    -----
    Operates on an existing Axes object.

    """

    # load static, randomly generated data
    x = np.array([[
        1.7117, 2.5470, 3.2817, 2.3303, 2.7066, 4.2024, 2.7184, 2.9790,
        2.7782, 1.9440, 3.9939, 4.3938, 6.1780, 3.2937, 3.6596, 2.3589,
        1.5408, 3.7236, 2.9327, 4.2844, 3.5441, 3.9499, 2.0023, 3.7872,
        3.4989, 2.2898, 2.7913, 3.2796, 2.3650, 3.5436, 3.3459, 3.8699,
        3.7448, 2.0149, 2.1290, 4.2193, 4.3932, 1.6687, 5.1053, 2.3849,
        1.6996, 3.1484, 3.4078, 2.0051, 0.88211, 2.038, 3.3291, 2.3526,
        1.4030, 2.7147
    ]])

    # plot the boxplot
    bpLeg = ax.boxplot(x, notch=notch, positions=[1], widths=0.5, bootstrap=10000)

    # plot the mean
    if showmean:
        ax.plot(1, x.mean(), marker='o', mec='k', mfc='r', zorder=100)

    # format stuff (colors and linewidths and whatnot)
    plt.setp(bpLeg['whiskers'], color='k', linestyle='-', zorder=10)
    plt.setp(bpLeg['boxes'], color='k', zorder=10)
    plt.setp(bpLeg['medians'], color='r', zorder=5, linewidth=1.25)
    plt.setp(bpLeg['fliers'], marker='o', mfc='none', mec='r', ms=4, zorder=10)
    plt.setp(bpLeg['caps'], linewidth=0)

    # positions of the boxplot elements
    if notch:
        x05, y05 = 1.00, bpLeg['caps'][0].get_ydata()[0]
        x25, y25 = bpLeg['boxes'][0].get_xdata()[1], bpLeg['boxes'][0].get_ydata()[0]
        x50, y50 = bpLeg['medians'][0].get_xdata()[1], bpLeg['medians'][0].get_ydata()[0]
        x75, y75 = bpLeg['boxes'][0].get_xdata()[1], bpLeg['boxes'][0].get_ydata()[5]
        x95, y95 = 1.00, bpLeg['caps'][1].get_ydata()[0]
        xEx, yEx = 1.00, bpLeg['fliers'][0].get_ydata()[0]
        xCIL, yCIL = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[2]
        xCIU, yCIU = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[4]

    else:
        x05, y05 = bpLeg['caps'][0].get_xdata()[1], bpLeg['caps'][1].get_ydata()[0]
        x25, y25 = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[0]
        x50, y50 = bpLeg['medians'][0].get_xdata()[1], bpLeg['medians'][0].get_ydata()[0]
        x75, y75 = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[2]
        x95, y95 = bpLeg['caps'][1].get_xdata()[1], bpLeg['caps'][0].get_ydata()[0]
        xEx, yEx = 0.95, bpLeg['fliers'][0].get_ydata()[0]

    # axes formatting
    ax.set_xlim([0, 2])
    ax.set_yticks(range(9))
    ax.set_yticklabels([])
    ax.set_xticklabels([], fontsize=5)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # annotation formats
    ap = dict(arrowstyle="->", shrinkB=shrink)

    # text for the labels
    note1 = r'{1) Interquartile range: $\mathrm{IQR} = \mathrm{Q3} - \mathrm{Q1}$}'
    note2 = '{2) Geometric means are plotted only for bacteria data.\nOtherwise, arithmetic means are shown.'
    legText = {
        '5th': r'{Min. data $\ge \mathrm{Q1} - 1.5 \times \mathrm{IQR}$}',
        '25th': r'{$25^{\mathrm{th}}\:\mathrm{percentile},\:\mathrm{Q}1$}',
        '50th': r'{$50^{\mathrm{th}} \mathrm{percentile}$, median}',
        '75th': r'{$75^{\mathrm{th}}\:\mathrm{percentile},\:\mathrm{Q3}$}',
        '95th': r'{Max. data $\le \mathrm{Q3} + 1.5 \times \mathrm{IQR}$}',
        'Out': r'{Outlier $>\mathrm{Q3} + 1.5 \times \mathrm{IQR}$} (note 1)',
        'CIL': r'{Lower 95\% CI about the median}',
        'CIU': r'{Upper 95\% CI about the median}',
        'notes': 'Notes:\n%s\n%s' % (note1, note2),
        'mean': r'Mean (note 2)'
    }

    # add notes
    ax.annotate(legText['notes'], (0.05, 0.01), xycoords='data', fontsize=fontsize)

    # label the mean
    if showmean:
        ax.annotate(legText['mean'], (1, x.mean()), xycoords='data',
                    xytext=(1.15, 5.75), textcoords='data', va='center',
                    arrowprops=ap, fontsize=fontsize)

    # label the lower whisker
    ax.annotate(legText['5th'], (x05, y05), xycoords='data',
                xytext=(1.25, y05), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label 1st quartile
    ax.annotate(legText['25th'], (x25, y25), xycoords='data',
                xytext=(1.55, y25*0.75), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label the median
    ax.annotate(legText['50th'], (x50, y50), xycoords='data',
                xytext=(1.45, y50), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label the 3rd quartile
    ax.annotate(legText['75th'], (x75, y75), xycoords='data',
                xytext=(1.55, y75*1.25), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label the upper whisker
    ax.annotate(legText['95th'], (x95, y95), xycoords='data',
                xytext=(0.05, 7.00), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label an outlier
    ax.annotate(legText['Out'], (xEx, yEx), xycoords='data',
                xytext=(1.00, 7.50), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label confidence intervals around the mean
    if notch:
        ax.annotate(legText['CIL'], (xCIL, yCIL), xycoords='data',
                    xytext=(0.05, 0.75*yCIL), textcoords='data', va='center',
                    arrowprops=ap, fontsize=fontsize)

        ax.annotate(legText['CIU'], (xCIU, yCIU), xycoords='data',
                    xytext=(0.05, 1.25*yCIU), textcoords='data', va='center',
                    arrowprops=ap, fontsize=fontsize)

    # legend and grid formats
    ax.set_frame_on(False)
    ax.yaxis.grid(False, which='major')
    ax.xaxis.grid(False, which='major')


def makeBoxplotLegend(filename='bmp/tex/boxplotlegend', figsize=4, **kwargs):
    """ Creates an explanatory diagram for boxplots.

    Parameters
    ----------
    filename : string
        File name and path (without extension) where the figure will be
        saved.
    figsize : float or int
        Size of the figure in inches.
    kwargs : keyword arguments
        Keyword arguments passed directly to `_boxplot_legend`

    Returns
    -------
    None

    """

    # setup the figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # call the helper function that does the heavy lifting
    _boxplot_legend(ax, **kwargs)

    # optimize the figure's layout
    fig.tight_layout()

    # save and close
    fig.savefig(filename + '.pdf', transparent=True, dpi=300)
    fig.savefig(filename + '.png', transparent=True, dpi=300)
    plt.close(fig)


def nested_getattr(baseobject, attribute):
    """  Returns the value of an attribute of an object that is nested
    several layers deep.

    Parameters
    ----------
    baseobject : this seriously can be anything
        The top-level object
    attribute : string
        Any string of what you want.

    Returns
    -------
    output : object
        No telling what type it is. It depends on what you ask for.

    Examples
    --------
    >>> nested_getattr(dataset, 'influent.mean')

    """

    for attr in attribute.split('.'):
        baseobject = getattr(baseobject, attr)
    return baseobject


def stringify(value, fmt, attribute=None):
    """ Weird wrapper to format attributes of objects as strings

    Parameters
    ----------
    value : object
        The item or top-level container of the item to be formatted.
    fmt : str
        A valid old-style(e.g., %0.3f) python format string
    attribute : string, optional
        Bottom-level attribute of ``value`` to be formatted.

    Returns
    -------
    formatted_value : string
        The formatted version of ``value.<attribute>``. If
        ``value.<attribute>`` is None, "--" is returned instead.

    Examples
    --------
    >>> stringify(None, '%s')
    '--'
    >>> stringify(1.2, '%0.3f')
    '1.200'
    >>> stringify(dataset, '%d', 'influent.N')
    '4'

    """

    if attribute is not None and value is not None:
        quantity = nested_getattr(value, attribute)
    else:
        quantity = value

    if quantity is None:
        return '--'
    else:
        return fmt % quantity


def redefineIndexLevel(dataframe, levelname, value, criteria=None, dropold=True):
    """ Redefine a index values in a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to be modified.
    levelname : string
        The name of the index level that needs to be modified. The catch
        here is that this value needs to be valid after calling
        `dataframe.reset_index()`. In otherwords, if you have a 3-level
        column index and you want to modify the "Units" level of the
        index, you should actually pass `("Units", "", "")`. Annoying,
        but that's life right now.
    value : string or int
        The replacement value for the index level.
    critera : function/lambda expression or None
        This should return True/False in a manner consitent with the
        `.select()` method of a pandas dataframe. See that docstring
        for more info. If None, the redifinition will apply to the whole
        dataframe.
    dropold : optional bool (defaul is True)
        Toggles the replacement (True) or addition (False) of the data
        of the redefined BMPs into the the `data` dataframe.

    Returns
    -------
    appended : pandas.DataFrame
        Dataframe with the modified index.

    """

    if criteria is not None:
        selection = dataframe.select(criteria)
    else:
        selection = dataframe.copy()

    if dropold:
        dataframe = dataframe.drop(selection.index)

    selection.reset_index(inplace=True)
    selection[levelname] = value
    selection = selection.set_index(dataframe.index.names)

    return dataframe.append(selection).sort_index()


def whiskers_and_fliers(x, q1=None, q3=None, transformout=None):
    """ Computes extent of whiskers and fliers on optionally transformed
    data for box and whisker plots.

    Parameters
    ----------
    x : array-like
        Sequence of optionally transformed data.
    q1, q3 : floats, optional
        First and third quartiles of the optionally transformed data.
    transformout : callable, optional
        Function to un-transform the results back into the original
        space of the data.

    Returns
    -------
    whiskers_and_fliers : dict
        Dictionary of whisker and fliers values.

    Examples
    --------
    >>> x = np.random.lognormal(size=37)
    >>> whisk_fly = whiskers_and_fliers(np.log(x), transformout=np.exp)

    See also
    --------
    wqio.utils.figutils.boxplot

    """
    wnf = {}
    if transformout is None:
        transformout = lambda x: x

    if q1 is None:
        q1 = np.percentile(x, 25)

    if q3 is None:
        q1 = np.percentile(x, 75)

    iqr = q3 - q1
    # get low extreme
    loval = q1 - (1.5 * iqr)
    whislo = np.compress(x >= loval, x)
    if len(whislo) == 0 or np.min(whislo) > q1:
        whislo = q1
    else:
        whislo = np.min(whislo)

    # get high extreme
    hival = q3 + (1.5 * iqr)
    whishi = np.compress(x <= hival, x)
    if len(whishi) == 0 or np.max(whishi) < q3:
        whishi = q3
    else:
        whishi = np.max(whishi)

    wnf['fliers'] = np.hstack([
        transformout(np.compress(x < whislo, x)),
        transformout(np.compress(x > whishi, x))
    ])
    wnf['whishi'] = transformout(whishi)
    wnf['whislo'] = transformout(whislo)

    return wnf


class ProgressBar:
    def __init__(self, sequence, width=50, labels=None, labelfxn=None):
        '''Progress bar for notebookes:

        Basic Usage:
        >>> X = range(1000)
        >>> pbar = utils.ProgressBar(X)
        >>> for n, x in enumerate(X, 1):
        >>>     # do stuff with x
        >>>     pbar.animate(n)


        '''
        self.sequence = sequence
        self.iterations = len(sequence)
        self.labels = labels
        self.labelfxn = labelfxn
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = width
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        if self.labels is None and self.labelfxn is None:
            self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)
        elif elapsed_iter <= self.iterations:
            if self.labels is None:
                label = self.labelfxn(self.sequence[elapsed_iter-1])
            else:
                label = self.labels[elapsed_iter-1]

            self.prog_bar += '  %d of %s (%s)' % (elapsed_iter, self.iterations, label)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
