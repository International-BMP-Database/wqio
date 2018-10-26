import itertools
from textwrap import dedent
from collections import namedtuple

import numpy
from scipy import stats
import statsmodels.api as sm
from probscale.algo import _estimate_from_fit

from wqio import validate


TheilStats = namedtuple(
    'TheilStats',
    ('slope', 'intercept', 'low_slope', 'high_slope')
)


def sigFigs(x, n, expthresh=5, tex=False, pval=False, forceint=False):
    """ Formats a number with the correct number of sig figs.

    Parameters
    ----------
    x : int or float
        The number you want to format.
    n : int
        The number of sig figs it should have.
    expthresh : int, optional (default = 5)
        The absolute value of the order of magnitude at which numbers
        are formatted in exponential notation.
    tex : bool, optional (default is False)
        Toggles the scientific formatting of the number either for
        terminal output (False) or LaTeX documents (True).
    pval : bool, optional (default is False)
        Useful for formatting p-values from hypothesis tests. When True
        and x < 0.001, will return "<0.001".
    forceint : bool, optional (default is False)
        If true, simply returns int(x)

    Returns
    -------
    formatted : str
        The formatted number as a string

    Examples
    --------
    >>> print(sigFigs(1247.15, 3))
    1,250

    >>> print(sigFigs(1247.15, 7))
    1,247.150

    """

    # check on the number provided
    if x is not None and not numpy.isinf(x) and not numpy.isnan(x):

        # check on the sigFigs
        if n < 1:
            raise ValueError("number of sig figs must be greater than zero!")

        # return a string value unaltered
        if isinstance(x, str):
            out = x

        elif pval and x < 0.001:
            out = "<0.001"
            if tex:
                out = '${}$'.format(out)

        elif forceint:
            out = '{:,.0f}'.format(x)

        # logic to do all of the rounding
        elif x != 0.0:
            order = numpy.floor(numpy.log10(numpy.abs(x)))

            if -1.0 * expthresh <= order <= expthresh:
                decimal_places = int(n - 1 - order)

                if decimal_places <= 0:
                    out = '{0:,.0f}'.format(round(x, decimal_places))

                else:
                    fmt = '{0:,.%df}' % decimal_places
                    out = fmt.format(x)

            else:
                decimal_places = n - 1
                if tex:
                    fmt = r'$%%0.%df \times 10 ^ {%d}$' % (decimal_places, order)
                    out = fmt % round(x / 10 ** order, decimal_places)
                else:
                    fmt = '{0:.%de}' % decimal_places
                    out = fmt.format(x)

        else:
            out = str(round(x, n))

    # with NAs and INFs, just return 'NA'
    else:
        out = 'NA'

    return out


def formatResult(result, qualifier, sigfigs=3):
    """ Formats a results with its qualifier

    Parameters
    ----------
    results : float
        The concentration or particulate strength
    qualifier : string
        The result's qualifier
    sigfigs : int
        The number of significant digits to which `result` should be
        formatted

    Returns
    -------
    formatted : string

    Example
    -------
    >>> formatResult(1.23, '<', sigfigs=4)
    '<1.230'

    """

    return '{}{}'.format(qualifier, sigFigs(result, sigfigs))


def process_p_vals(pval):
    """ Processes p-values into nice strings to reporting. When the
    p-values are less than 0.001, "<0.001" is returned. Otherwise, a
    string with three decimal places is returned.

    Parameters
    ----------
    pval : float
        The p-value to be formatted. Must be between 0 and 1. Otherwise
        an error is raised.

    Returns
    -------
    processed : string

    """
    if pval is None:
        out = 'NA'
    elif 0 < pval < 0.001:
        out = formatResult(0.001, '<', sigfigs=1)
    elif pval > 1 or pval < 0:
        raise ValueError('p-values must be between 0 and 1 (not {})'.format(pval))
    else:
        out = '%0.3f' % pval

    return out


def translate_p_vals(pval):
    """ Translates ambiguous p-values into more meaningful emoji.

    Parameters
    ----------
    pval : float

    Returns
    -------
    emoji : string

    """

    if pval is None:
        emoji = r'ಠ_ಠ'
    elif pval <= 0.01:
        emoji = r'¯\(ツ)/¯'
    elif 0.01 < pval <= 0.05:
        emoji = r'¯\_(ツ)_/¯'
    elif 0.05 < pval <= 0.1:
        emoji = r'¯\__(ツ)__/¯'
    elif 0.1 < pval <= 1:
        emoji = r'(╯°□°)╯︵ ┻━┻'
    else:
        raise ValueError('p-values must be between 0 and 1 (not {})'.format(pval))

    return emoji


def anderson_darling(data):
    """
    Compute the Anderson-Darling Statistic and p-value

    Parmeters
    ---------
    data : array-like

    Returns
    -------
    ADResult : namedtuple
        A collection of results. Keys are:

          * ``statistic``
          * ``critical_values``
          * ``significance_level``
          * ``pvalue``


    See Also
    --------
    scipy.stats.anderson

    """
    AD = stats.anderson(data)
    field_names = list(AD._fields) + ['pvalue']

    p = _anderson_darling_p_vals(AD, len(data))
    values = AD._asdict()
    values['pvalue'] = p

    ADResult = namedtuple('ADResult', field_names)
    return ADResult(**values)


def processAndersonDarlingResults(ad_results):
    """ Return a nice string of Anderson-Darling test results

    Parameters
    ----------
    ad_result : tuple or namedtuple
        The packed output from scipy.stats.anderson

    Returns
    -------
    result : str
        A string representation of the confidence in the result.

    """

    AD, crit, sig = ad_results
    try:
        ci = 100 - sig[AD > crit][-1]
        return '%0.1f%%' % (ci,)
    except IndexError:
        ci = 100 - sig[0]
        return '<%0.1f%%' % (ci,)


def _anderson_darling_p_vals(ad_results, n_points):
    """
    Computes the p-value of the Anderson-Darling Result

    Parameters
    ----------
    ad_result : tuple or namedtuple
        The packed output from scipy.stats.anderson
    n_points : int
        The number of points in the sample.

    Returns
    -------
    p : float

    References
    ----------
    R.B. D'Augostino and M.A. Stephens, Eds., 1986, Goodness-of-Fit
    Techniques, Marcel Dekker

    """

    AD, crit, sig = ad_results
    AD_star = AD * (1 + 0.75 / n_points + 2.25 / n_points**2)
    if AD_star >= 0.6:
        p = numpy.exp(1.2397 - (5.709 * AD_star) + (0.0186 * AD_star**2))
    elif 0.34 <= AD_star < 0.6:
        p = numpy.exp(0.9177 - (4.279 * AD_star) - (1.38 * AD_star**2))
    elif 0.2 < AD_star < 0.34:
        p = 1 - numpy.exp(-8.318 + (42.796 * AD_star) - (59.938 * AD_star**2))
    else:
        p = 1 - numpy.exp(-13.436 + (101.14 * AD_star) - (223.73 * AD_star**2))

    return p


def normalize_units(df, unitsmap, targetunit, paramcol='parameter',
                    rescol='res', unitcol='units', napolicy='ignore'):
    """
    Normalize units of measure in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe contained results and units of measure data.
    unitsmap : dictionary
        Dictionary where keys are the units present in the df and
        values are the conversion factors required to standardize
        results to a common unit.
    targetunit : string or dict
        The desired final units of measure. Must be present in
        ``unitsmap``. If a string, all rows in df will be
        assigned ``targetunit``. If a dictionary, the units will be
        mapped from the dictionary using the values of ``paramcol`` as
        keys.
    paramcol, rescol, unitcol : string, optional
        Labels for the parameter, results, and units columns in the
        df.
    napolicy : string, optional
        Determines how mull/missing values are stored. By default,
        this is set to "ignore". Use "raise" to throw a ``ValueError``
        if when unit conversion or target units data cannot be found.

    Returns
    -------
    normalized : pandas.DataFrame
        Dataframe with normalized units of measure.

    """

    # determine the preferred units in the wqdata
    target = df[paramcol].map(targetunit)

    # factors to normialize to standard units
    normalization = df[unitcol].map(unitsmap)

    # factor to convert to preferred units
    conversion = target.map(unitsmap)

    if napolicy == 'raise':
        msg = ''
        if target.isnull().any():
            nulls = df[target.isnull()][paramcol].unique()
            msg += "Some target units could not be mapped to the {} column ({})\n".format(
                paramcol, nulls
            )

        if normalization.isnull().any():
            nulls = df[normalization.isnull()][unitcol].unique()
            msg += "Some normalization factors could not be mapped to the {} column ({})\n".format(
                unitcol, nulls
            )

        if conversion.isnull().any():
            nulls = target[conversion.isnull()]
            msg += "Some conversion factors could not be mapped to the target units ({})".format(
                nulls
            )

        if len(msg) > 0:
            raise ValueError(msg)

    # convert results
    normalized = df.assign(**{
        rescol: df[rescol] * normalization / conversion,
        unitcol: target,
    })

    return normalized


def pH2concentration(pH, *args):
    """ Converts pH values to proton concentrations in mg/L

    Parameters
    ----------
    pH : int or float
        Number between 0 and 14 representing the pH

    Returns
    -------
    conc : float
        Concentration of protons in mg/L

    """

    # check that we recieved a valid input:
    if pH < 0 or pH > 14:
        raise ValueError('pH = %f but must be between 0 and 14' % pH)

    # avogadro's number (items/mole)
    avogadro = 6.0221413e+23

    # mass of a proton (kg)
    proton_mass = 1.672621777e-27

    # grams per kilogram
    kg2g = 1000

    # milligrams per gram
    g2mg = 1000

    return 10 ** (-1 * pH) * avogadro * proton_mass * kg2g * g2mg


def compute_theilslope(y, x=None, alpha=0.95, percentile=50):
    """ Adapted from stats.mstats.theilslopes
    https://goo.gl/nxPF54
    {}
    """.format(dedent(stats.mstats.theilslopes.__doc__))

    # We copy both x and y so we can use _find_repeats.
    y = numpy.array(y).flatten()
    if x is None:
        x = numpy.arange(len(y), dtype=float)
    else:
        x = numpy.array(x, dtype=float).flatten()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths (%s != %s)" %
                             (len(y), len(x)))

    # Compute sorted slopes only when deltax > 0
    deltax = x[:, numpy.newaxis] - x
    deltay = y[:, numpy.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    slopes.sort()
    outslope = numpy.percentile(slopes, percentile)
    outinter = numpy.percentile(y, percentile) - outslope * numpy.median(x)
    # Now compute confidence intervals
    if alpha > 0.5:
        alpha = 1. - alpha

    z = stats.distributions.norm.ppf(alpha / 2.)

    # This implements (2.6) from Sen (1968)
    _, nxreps = stats.mstats.find_repeats(x)
    _, nyreps = stats.mstats.find_repeats(y)
    nt = len(slopes)       # N in Sen (1968)
    ny = len(y)            # n in Sen (1968)

    # Equation 2.6 in Sen (1968):
    sigsq = 1 / 18. * (ny * (ny - 1) * (2 * ny + 5) -
                       numpy.sum([k * (k - 1) * (2 * k + 5) for k in nxreps]) -
                       numpy.sum([k * (k - 1) * (2 * k + 5) for k in nyreps]))

    # Find the confidence interval indices in `slopes`
    sigma = numpy.sqrt(sigsq)
    Ru = min(int(numpy.round((nt - z * sigma) / 2.)), len(slopes) - 1)
    Rl = max(int(numpy.round((nt + z * sigma) / 2.)) - 1, 0)
    delta = slopes[[Rl, Ru]]

    return TheilStats(outslope, outinter, delta[0], delta[1])


def fit_line(x, y, xhat=None, fitprobs=None, fitlogs=None, dist=None, through_origin=False):
    """ Fits a line to x-y data in various forms (raw, log, prob scales)

    Parameters
    ----------
    x, y : array-like
        Independent and dependent data, respectively.
    xhat : array-like or None, optional
        The values at which yhat should should be estimated. If
        not provided, falls back to the sorted values of ``x``.
    fitprobs, fitlogs : str, options.
        Defines how data should be transformed. Valid values are
        'x', 'y', or 'both'. If using ``fitprobs``, variables should
        be expressed as a percentage, i.e.,
        Probablility transform = lambda x: ``dist``.ppf(x / 100.).
        Log transform = lambda x: numpy.log(x).
        Take care to not pass the same value to both ``fitlogs`` and
        ``figprobs`` as both transforms will be applied.
    dist : scipy.stats distribution or None, optional
        A fully-spec'd scipy.stats distribution such that ``dist.ppf``
        can be called. If not provided, defaults to scipt.stats.norm.
    through_origin : bool, optional (default = False)
        If True, the y-intercept of the best-fit line will be
        constrained to 0 during the regression.

    Returns
    -------
    xhat, yhat : numpy arrays
        Linear model estimates of ``x`` and ``y``.
    results : a statmodels result object
        The object returned by statsmodels.OLS.fit()

    """

    fitprobs = validate.fit_arguments(fitprobs, "fitprobs")
    fitlogs = validate.fit_arguments(fitlogs, "fitlogs")

    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if xhat is None:
        xhat = numpy.array([numpy.min(x), numpy.max(x)])

    if dist is None:
        dist = stats.norm

    if fitprobs in ['x', 'both']:
        x = dist.ppf(x / 100.)
        xhat = dist.ppf(numpy.array(xhat) / 100.)

    if fitprobs in ['y', 'both']:
        y = dist.ppf(y / 100.)

    if fitlogs in ['x', 'both']:
        x = numpy.log(x)

    if fitlogs in ['y', 'both']:
        y = numpy.log(y)

    x = sm.add_constant(x)

    # fix the y-intercept at 0 if requested
    # note that this is a nuance of the statsmodels package.
    # `x[:, 0] = 5` doesn't fix the intercept at 5, for instance.
    if through_origin:
        x[:, 0] = 0

    results = sm.OLS(y, x).fit()
    yhat = _estimate_from_fit(xhat, results.params[1], results.params[0],
                              xlog=fitlogs in ['x', 'both'],
                              ylog=fitlogs in ['y', 'both'])

    if fitprobs in ['y', 'both']:
        yhat = 100. * dist.cdf(yhat)
    if fitprobs in ['x', 'both']:
        xhat = 100. * dist.cdf(xhat)

    return xhat, yhat, results


def checkIntervalOverlap(interval1, interval2, oneway=False):
    """ Checks if two numeric intervals overlaps.

    Parameters
    ----------
    interval1, interval2 : array like
        len = 2 sequences to compare
    oneway : bool (default = False)
        If True, only checks that ``interval1`` falls at least partially
        inside ``interval2``, but not the other way around

    Returns
    -------
    overlap : bool
        True if an overlap exists. Otherwise, False.

    """

    test1 = numpy.min(interval2) <= numpy.max(interval1) <= numpy.max(interval2)
    test2 = numpy.min(interval2) <= numpy.min(interval1) <= numpy.max(interval2)

    # TODO: try test1 or test2 or (not oneway and test3)

    if oneway:
        return test1 or test2
    else:
        test3 = checkIntervalOverlap(interval2, interval1, oneway=True)
        return test1 or test2 or test3


def winsorize_dataframe(df, **limits):
    """ Winsorizes columns in a dataframe

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data to be modified (a copy is created).
    **limits : optional kwargs (floats, or two-tuples of floats)
        Optional key-value pairs of column names and (two-tuples of)
        floats that are the windsor limits to be applied to the
        respective column. Values should be between 0 and 1.

    Returns
    -------
    winsored_df : pandas.DataFrame
        The modified dataframe.

    See also
    --------
    scipy.stats.mstats.winsorize

    """
    newcols = {
        colname: stats.mstats.winsorize(df[colname], limits=limit)
        for colname, limit in limits.items()
    }

    return df.assign(**newcols)


def remove_outliers(x, factor=1.5):
    """ Removes outliers from an array based on a scaling of the
    interquartile range (IQR).

    Parameters
    ----------
    x : array-like
    factor : float
        The scaling factor for the IQR

    Returns
    -------
    cleaned : numpy.array
        Copy of *x* with the outliers removed

    Example
    -------
    >>> import numpy
    >>> from wqio import utils
    >>> numpy.random.seed(0)
    >>> x = numpy.random.normal(0, 4, size=37) # shape is (37,)
    >>> utils.remove_outliers(x).shape
    (35,)

    """
    mean = numpy.mean(x)
    pctl25 = numpy.percentile(x, 25)
    pctl75 = numpy.percentile(x, 75)
    IQR = pctl75 - pctl25
    return x[(x >= pctl25 - (factor * IQR)) & (x <= pctl75 + (factor * IQR))]


def _comp_stat_generator(df, groupcols, pivotcol, rescol, statfxn,
                         statname=None, **statopts):
    """ Generator of records containing results of comparitive
    statistical functions.

    Parameters
    ----------
    df : pandas.DataFrame
    groupcols : list of string
        The columns on which the data will be groups. This should *not*
        include the column that defines the separate datasets that will
        be comparied.
    pivotcol : string
        The column that distinquishes the individual samples that are
        being compared.
    rescol : string
        The column that containes the values to compare.
    statfxn : callable
        Any function that takes to array-like datasets and returns a
        ``namedtuple`` with elements "statistic" and "pvalue". If the
        "pvalue" portion is not relevant, simply wrap the function such
        that it returns None or similar for it.
    statname : string, optional
        The name of the primary statistic being returned.
    **statopts : kwargs
        Additional options to be fed to ``statfxn``.

    Returns
    -------
    comp_stats : pandas.DataFrame

    """

    groupcols = validate.at_least_empty_list(groupcols)
    if statname is None:
        statname = 'stat'

    groups = df.groupby(by=groupcols)
    for name, g in groups:
        stations = g[pivotcol].unique()
        name = validate.at_least_empty_list(name)
        for _x, _y in itertools.permutations(stations, 2):
            row = dict(zip(groupcols, name))

            station_columns = [pivotcol + '_1', pivotcol + '_2']
            row.update(dict(zip(station_columns, [_x, _y])))

            x = g.loc[g[pivotcol] == _x, rescol].values
            y = g.loc[g[pivotcol] == _y, rescol].values

            stat = statfxn(x, y, **statopts)
            row.update({
                statname: stat[0],
                'pvalue': stat.pvalue,
            })
            yield row


def _paired_stat_generator(df, groupcols, pivotcol, rescol, statfxn,
                           statname=None, **statopts):
    """ Generator of records containing results of comparitive
    statistical functions specifically for paired data.

    Parameters
    ----------
    df : pandas.DataFrame
    groupcols : list of string
        The columns on which the data will be groups. This should *not*
        include the column that defines the separate datasets that will
        be comparied.
    pivotcol : string
        The column that distinquishes the individual samples that are
        being compared.
    rescol : string
        The column that containes the values to compare.
    statfxn : callable
        Any function that takes to array-like datasets and returns a
        ``namedtuple`` with elements "statistic" and "pvalue". If the
        "pvalue" portion is not relevant, simply wrap the function such
        that it returns None or similar for it.
    statname : string, optional
        The name of the primary statistic being returned.
    **statopts : kwargs
        Additional options to be fed to ``statfxn``.

    Returns
    -------
    comp_stats : pandsa.DataFrame

    """

    groupcols = validate.at_least_empty_list(groupcols)
    if statname is None:
        statname = 'stat'

    groups = df.groupby(level=groupcols)[rescol]
    for name, g in groups:
        stations = g.columns.tolist()
        name = validate.at_least_empty_list(name)
        for _x, _y in itertools.permutations(stations, 2):
            row = dict(zip(groupcols, name))

            station_columns = [pivotcol + '_1', pivotcol + '_2']
            row.update(dict(zip(station_columns, [_x, _y])))

            _df = g[[_x, _y]].dropna()

            x = _df[_x].values
            y = _df[_y].values

            stat = statfxn(x, y, **statopts)
            row.update({
                statname: stat[0],
                'pvalue': stat.pvalue,
            })
            yield row
