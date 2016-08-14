from io import StringIO

import numpy
import pandas


@numpy.deprecate(new_name='add_column_level')
def addSecondColumnLevel(levelval, levelname, df):
    """ Add a second level to the column-index if a dataframe.

    Parameters
    ----------
    levelval : int or string
        Constant value to be assigned to the second level.
    levelname : string
        The name of the second level.
    df : pandas.DataFrame
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

    return add_column_level(df, levelval, levelname)


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
    colarray = [[levelvalue]*len(df.columns), df.columns]
    colindex = pandas.MultiIndex.from_arrays(colarray)

    # copy the dataframe and redefine the columns
    newdf = df.copy()
    newdf.columns = colindex
    newdf.columns.names = [levelname, origlevel]

    return newdf


def redefineIndexLevel(df, levelname, value, criteria=None, dropold=True):
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
    critera : function/lambda expression or None
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

    if criteria is not None:
        selection = df.select(criteria)
    else:
        selection = df.copy()

    if dropold:
        df = df.drop(selection.index)

    selection.reset_index(inplace=True)
    selection[levelname] = value
    selection = selection.set_index(df.index.names)

    return df.append(selection).sort_index()


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
