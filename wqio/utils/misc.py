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


class ProgressBar: # pragma: no cover
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
