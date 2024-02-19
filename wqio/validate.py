import os

import numpy
import pandas
from matplotlib import pyplot


def dataset(fname):
    valid_names = ["bmpdata", "cvc", "nsqd"]
    fname, ext = os.path.splitext(fname)
    if fname.lower() not in valid_names:
        msg = "filename '{}' not one of {}"
        raise ValueError(msg.format(fname, valid_names))

    return fname.lower() + ".zip"


def timestamp(datelike):
    """Converts datetime-like objects to pandas.Timestamp.
    Pretty miuch a direct pass through, but give a slighly
    more informative error message.

    Parameters
    ----------
    datelike : datetime.datetime, string, or Timestamp
        The value to be coerced into the pandas.Timestamp.

    Returns
    -------
    tstamp : pandas.Timestamp
        Coerced value.

    """

    try:
        tstamp = pandas.Timestamp(datelike)
    except ValueError:
        msg = f"{datelike} could not be coerced into a pandas.Timestamp"
        raise ValueError(msg)

    return tstamp


def axes(ax, fallback="new"):
    """Checks if a value if an Axes. If None, a new one is created or
    the 'current' one is found.

    Parameters
    ----------
    ax : matplotlib.Axes or None
        The value to be tested.
    fallback : str, optional
        If ax is None. ``fallback='new'`` will create a new axes and
        figure. The other option is ``fallback='current'``, where the
        "current" axes are return via ``pyplot.gca()``

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    """

    if ax is None:
        if fallback == "new":
            fig, ax = pyplot.subplots()
        elif fallback == "current":
            ax = pyplot.gca()
            fig = ax.figure
        else:
            raise ValueError("fallback must be either 'new' or 'current'")

    elif isinstance(ax, pyplot.Axes):
        fig = ax.figure

    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    return fig, ax


def single_value_in_index(df, index_level):
    """Confirms that a given level of a dataframe's index only has
    one unique value. Useful for confirming consistent units. Raises
    error if level is not a single value. Returns unique value of the
    index level.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe whose index will be inspected.
    index_level : int or string
        Level of the dataframe's index that should only have one unique
        value.

    Returns
    -------
    uniqueval
        The unique value of the index.

    """

    index = numpy.unique(df.index.get_level_values(index_level).tolist())
    if index.shape != (1,):
        raise ValueError(f'index level "{index_level}" is not unique.')

    return index[0]


def at_least_empty_list(value):
    if isinstance(value, numpy.ndarray):
        value = value.tolist()
    elif numpy.isscalar(value) and value != "":
        value = [value]
    elif not value:
        value = []

    return value


def at_least_empty_dict(value, **kwargs):
    if value is None or value == "":
        value = {}
    elif not isinstance(value, dict):
        raise ValueError(f"{value} cannot be a dictionary")
    else:
        value = value.copy()

    value.update(**kwargs)

    return value


def fit_arguments(arg, argname):
    valid_args = ["x", "y", "both", None]
    if arg not in valid_args:
        msg = "Valid value for {} ({}). Must be on of {}"
        raise ValueError(msg.format(argname, arg, valid_args))

    return arg
