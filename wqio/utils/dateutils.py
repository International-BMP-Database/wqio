import warnings

import numpy
import pandas

from wqio import validate


def getSeason(date):
    """ Defines the season from a given date.

    Parameters
    ----------
    date : datetime.datetime object or similar
        Any object that represents a date and has `.month` and `.day`
        attributes

    Returns
    -------
    season : str

    Notes
    -----
    Assumes that all seasons changed on the 22nd (e.g., all winters
    start on Decemeber 22). This isn't strictly true, but it's good
    enough for now.

    """

    date = validate.timestamp(date)
    if (date.month == 12 and date.day >= 22) or \
            (date.month in [1, 2]) or \
            (date.month == 3 and date.day < 22):
        return 'winter'
    elif (date.month == 3 and date.day >= 22) or \
            (date.month in [4, 5]) or \
            (date.month == 6 and date.day < 22):
        return 'spring'
    elif (date.month == 6 and date.day >= 22) or \
            (date.month in [7, 8]) or \
            (date.month == 9 and date.day < 22):
        return 'summer'
    elif (date.month == 9 and date.day >= 22) or \
            (date.month in [10, 11]) or \
            (date.month == 12 and date.day < 22):
        return 'autumn'
    else:  # pragma: no cover
        raise ValueError('could not assign season to  {}'.format(date))


def makeTimestamp(row, datecol='sampledate', timecol='sampletime',
                  issuewarnings=False):
    """ Makes a pandas.Timestamp from separate date/time columns

    Parameters
    ----------
    row : dict-like (ideallty a row in a dataframe)
    datecol : string (default = 'sampledate')
        Name of the column containing the dates
    timecol : string (default = 'sampletime')
        Name of the column containing the times
    issuewarnings : bool (default = False)
        Toggles UserWarnings when the fallback date (1901-01-01) or
        fallback time (00:00) must be used.

    Returns
    -------
    tstamp : pandas.Timestamp

    """

    fallback_datetime = pandas.Timestamp('1901-01-01 00:00')

    if row[datecol] is None or pandas.isnull(row[datecol]):
        fb_date = True
    else:
        fb_date = False
        try:
            date = pandas.Timestamp(row[datecol]).date()
        except ValueError:
            fb_date = True

    if fb_date:
        date = fallback_datetime.date()
        if issuewarnings:
            warnings.warn("Using fallback date from {}".format(row[datecol]))

    if row[timecol] is None or pandas.isnull(row[timecol]):
        fb_time = True
    else:
        fb_time = False
        try:
            time = pandas.Timestamp(row[timecol]).time()
        except ValueError:
            fb_time = True

    if fb_time:
        time = fallback_datetime.time()
        if issuewarnings:
            warnings.warn("Using fallback time from {}".format(row[timecol]))

    dtstring = '{} {}'.format(date, time)
    tstamp = pandas.Timestamp(dtstring)

    return tstamp


def getWaterYear(date):
    """ Returns the water year of a given date

    Parameters
    ----------
    date : datetime-like
        A datetime or Timestamp object

    Returns
    -------
    wateryear : string
        The water year of `date`

    Example
    -------
    >>> import datetime
    >>> import wqio
    >>> x = datetime.datetime(2005, 11, 2)
    >>> print(wqio.utils.getWaterYear(x))
        '2005/2006'

    """

    year = date.year
    yearstring = '{}/{}'
    if date.month >= 10:
        return yearstring.format(year, year + 1)
    else:
        return yearstring.format(year - 1, year)
