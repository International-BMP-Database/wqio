from io import StringIO
from copy import copy

import numpy
import pandas


def head_tail(df, N=5):
    return pandas.concat([df.head(N), df.tail(N)])


def add_column_level(df, levelvalue, levelname):
    """ Adds a second level to the column-index if a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe to be modified.
    levelvalue : int or string
        Constant value to be assigned to the second level.
    levelname : string
        The name of the second level.

    Returns
    -------
    newdf : pandas.DataFrame
        The mutated dataframe with a MultiIndex in the columns.

    Example
    -------
    >>> df = pandas.DataFrame(columns=['res', 'qual'], index=range(3))
    >>> df.columns
    Index(['res', 'qual'], dtype='object')
    >>> df2 = utils.add_column_level(df, 'Infl', 'location')
    >>> df2.columns
    MultiIndex(levels=[['Infl'], ['qual', 'res']],
               labels=[[0, 0], [1, 0]],
               names=['loc', 'quantity'])

    """
    if isinstance(df.columns, pandas.MultiIndex):
        raise ValueError('Dataframe already has MultiIndex on columns')

    origlevel = 'quantity'
    if df.columns.names[0] is not None:
        origlevel = df.columns.names[0]

    # define the index
    colarray = [[levelvalue] * len(df.columns), df.columns]
    colindex = pandas.MultiIndex.from_arrays(colarray)

    # copy the dataframe and redefine the columns
    newdf = df.copy()
    newdf.columns = colindex
    newdf.columns.names = [levelname, origlevel]

    return newdf


def swap_column_levels(df, level_1, level_2, sort=True):
    """ Swaps columns levels in a dataframe with multi-level columns

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with a multi-level column index
    level_1, level_2 : int or string
        The position or label of the columns levels to be swapped.

    Returns
    -------
    swapped : pandas.DataFrame

    Examples
    --------
    >>> import numpy
    >>> import pandas
    >>> import wqio
    >>> columns = pandas.MultiIndex.from_product(
    ...     [['A', 'B',], ['res', 'cen'], ['mg/L']],
    ...     names=['loc', 'value', 'units']
    ... )
    >>> data = numpy.arange(len(columns) * 3).reshape((3, len(columns)))
    >>> df = pandas.DataFrame(data, columns=columns)
    >>> print(df)
    loc      A         B
    value  res  cen  res  cen
    units mg/L mg/L mg/L mg/L
    0        0    1    2    3
    1        4    5    6    7
    2        8    9   10   11

    >>> print(wqio.utils.swap_column_levels(df, 'units', 'loc'))
    units mg/L
    value  cen     res
    loc      A   B   A   B
    0        1   3   0   2
    1        5   7   4   6
    2        9  11   8  10

    """
    df2 = df.copy()
    df2.columns = df2.columns.swaplevel(level_1, level_2)
    if sort:
        df2 = df2.sort_index(axis='columns')

    return df2


def flatten_columns(df, sep='_'):
    """ Completely flattens a multi-level column index

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame with multi-level columns that will be flattened.
    sep : string, optional
        The string that will be used to delimit each level of the
        column index in the flattened names.

    Returns
    -------
    flattened : pandas.DataFrame

    """
    df = df.copy()
    df.columns = [sep.join(_) for _ in df.columns]
    return df


def redefine_index_level(df, levelname, value, criteria=None, dropold=True):
    """ Redefine a index values in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be modified.
    levelname : string
        The name of the index level that needs to be modified. The catch
        here is that this value needs to be valid after calling
        `df.reset_index()`. In otherwords, if you have a 3-level
        column index and you want to modify the "Units" level of the
        index, you should actually pass `("Units", "", "")`. Annoying,
        but that's life right now.
    value : string or int
        The replacement value for the index level.
    criteria : function/lambda expression or None
        This should return True/False in a manner consitent with the
        `.select()` method of a pandas dataframe. See that docstring
        for more info. If None, the redifinition will apply to the whole
        df.
    dropold : optional bool (defaul is True)
        Toggles the replacement (True) or addition (False) of the data
        of the redefined BMPs into the the `data` df.

    Returns
    -------
    appended : pandas.DataFrame
        Dataframe with the modified index.

    """

    if criteria is None:
        def criteria(*args, **kwargs):
            return True

    redefined = (
        df.loc[df.index.map(criteria), :]
          .reset_index()
          .assign(**{levelname: value})
          .set_index(df.index.names)
    )

    if dropold:
        df = df.loc[df.index.map(lambda r: not criteria(r)), :]

    return pandas.concat([df, redefined]).sort_index()


def categorize_columns(df, *columns):
    newdf = df.copy()
    for c in columns:
        if newdf[c].dtype != object:
            raise ValueError("column {} is not an object type".format(c))
        newdf[c] = newdf[c].astype('category')

    return newdf


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


def classifier(value, bins, units=None):
    """
    An example classifier function for `storm_histogram`

    Parameters
    ----------
    value : float
        Valueue to classify.
    bins : array-like
        Finite right-edges of the classification bins.
    units : string, optional
        Units of measure to be appended to the returned categories.

    Returns
    -------
    category : string

    Examples
    --------
    >>> bins = [5, 10, 15, 20, 25]
    >>> _classifier(3, bins)
    "<5"
    >>> _classifier(12, bins, units='feet')
    "10 - 15 feet"
    >>> _classifier(48, bins, units='mm')
    ">25 mm"

    """

    units = units or ''

    if numpy.isnan(value):
        return numpy.nan

    # below the lower edge
    elif value <= min(bins):
        output = '<{}'.format(min(bins))

    # above the upper edge
    elif value > max(bins):
        output = '>{}'.format(max(bins))

    # everything else with the range of bins
    else:
        for left, right in zip(bins[:-1], bins[1:]):
            if left < value <= right:
                output = '{} - {}'.format(left, right)
                break

    # add the units (or don't)
    return '{} {}'.format(output, units or '').strip()


def unique_categories(classifier, bins):
    """
    Computs all of the unique category returned by a classifier.

    Parameters
    ----------
    classifier : callable
        A function or class that we called categorizes continuous
        (floating point) values.
    bins : array-like
        Finite right-edges of the classification bins.

    Returns
    -------
    categories : list
        List, sequentially ordered with the possible input values, of
        all of the possible categories returned from ``classifier``.

    Examples
    --------
    >>> from pycvc import viz
    >>> bins = [5, 10, 15]
    >>> viz._unique_categories(_viz._classifier, bins)
    ['<5', '5 - 10', '10 - 15', '>15']

    """
    bins = numpy.asarray(bins)
    midpoints = 0.5 * (bins[:-1] + bins[1:])
    all_bins = [min(bins) * 0.5] + list(midpoints) + [max(bins) * 2]
    return [classifier(value) for value in all_bins]


def pop_many(some_dict, *args):
    """ Pop several key-values out of a dictionary and return a copy

    Parameters
    ----------
    some_dict : dictionary
    *keys : hashables
        All of the keys you would like removed from *some_dict*

    Returns
    -------
    popped : dictionary

    Example
    -------
    >>> from wqio import utils
    >>> x = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    >>> utils.pop_many(x, 'A', 'D')
    {'B': 2, 'C': 3}

    """

    popped = copy(some_dict)
    for v in args:
        _ = popped.pop(v)
    return popped


def selector(default, *cond_results):
    """ Thin wrapper around numpy.select with a more convenient API (maybe).

    Parameters
    ----------
    default : scalar
        The default value to be returned when none of the conditions are met.abs
    *cond_results : tuple
        Tuples of conditions (bool arrays) and result values (scalar)

    Returns
    -------
    selected : numpy.array

    Example
    -------
    >>> from wqio import utils
    >>> import numpy
    >>> x = numpy.arange(10)
    >>> utils.selector('Z', (x <= 2, 'A'), (x < 6, 'B'), (x <= 7, 'C'))
    array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'Z', 'Z'],
          dtype='<U1')
    """
    conditions, results = zip(*cond_results)
    return numpy.select(conditions, results, default)


def non_filter(*args, **kwargs):
    return True


def no_op(value):
    return value
