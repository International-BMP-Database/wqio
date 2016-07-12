import numpy
from scipy import stats
import statsmodels.api as sm


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
           1250
    >>> print(sigFigs(1247.15, 7))
           1247.150

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
    >>> wqio.formatResult(1.23, '<', sigfigs=4)
    "<1.230"

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
        raise ValueError('p-values cannot be greater than 1 and less than 0')
    else:
        out = '%0.3f' % pval

    return out


def processAndersonDarlingResults(ad_results):
    """ Return a nice string of Anderson-Darling test results

    Parameters
    ----------
    ad_result : tuple or namedtuple
        The packed output from scipt.stats.anderson

    Returns
    -------
    result : str
        A string representation of the confidence in the result.

    """

    a2, crit, sig = ad_results
    try:
        ci = 100 - sig[a2 < crit][-1]
        return '%0.1f%%' % (ci,)
    except IndexError:
        ci = 100 - sig[0]
        return '<%0.1f%%' % (ci,)


def normalize_units(df, units_map, targetunit, paramcol='parameter',
                    rescol='res', unitcol='units', debug=False):
    """
    Normalize units of measure in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe contained results and units of measure data.
    units_map : dictionary
        Dictionary where keys are the units present in the df and
        values are the conversion factors required to standardize
        results to a common unit.
    targetunit : string or dict
        The desired final units of measure. Must be present in
        ``units_map``. If a string, all rows in df will be
        assigned ``targetunit``. If a dictionary, the units will be
        mapped from the dictionary using the values of ``paramcol`` as
        keys.
    paramcol, rescol, unitcol : string
        Labels for the parameter, results, and units columns in the
        df.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with normalized units of measure.

    Notes
    -----
    Input dfs are modified in-placed and returned. If you need
    to preserve the original df, pass a copy to this function.

    See also
    --------
    normalize_units2

    """


    # determine the preferred units in the wqdata
    target = df[paramcol].map(targetunit.get)

    # factors to normialize to standard units
    normalization = df[unitcol].map(units_map.get)

    # factor to convert to preferred units
    conversion = target.map(units_map.get)

    # convert results
    df[rescol] = df[rescol] * normalization / conversion

    # reassign units
    df[unitcol] = target

    return df


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

    return 10**(-1*pH) * avogadro * proton_mass * kg2g * g2mg


def fit_line(x, y, xhat=None, fitprobs=None, fitlogs=None, dist=None):
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

    Returns
    -------
    xhat, yhat : numpy arrays
        Linear model estimates of ``x`` and ``y``.
    results : a statmodels result object
        The object returned by statsmodels.OLS.fit()

    """

    def _check_fit_arg(arg, argname):
        valid_args = ['x', 'y', 'both', None]
        if arg not in valid_args:
            msg = 'Valid value for {} ({}). Must be on of {}'
            raise ValueError(msg.format(argname, arg, valid_args))

    _check_fit_arg(fitprobs, "fitprobs")
    _check_fit_arg(fitlogs, "fitlogs")

    if xhat is None:
        xhat = numpy.array([numpy.min(x), numpy.max(x)])

    if dist is None:
        dist = stats.norm

    if fitprobs in ['x', 'both']:
        x = dist.ppf(x/100.)
        xhat = dist.ppf(numpy.array(xhat)/100.)

    if fitprobs in ['y', 'both']:
        y  = dist.ppf(y/100.)

    if fitlogs in ['x', 'both']:
        x = numpy.log(x)
    if fitlogs in ['y', 'both']:
        y = numpy.log(y)

    x = sm.add_constant(x)
    model = model = sm.OLS(y, x)
    results = model.fit()


    yhat = estimateFromLineParams(xhat, results.params[1],
                                        results.params[0],
                                        xlog=fitlogs in ['x', 'both'],
                                        ylog=fitlogs in ['y', 'both'])

    if fitprobs in ['y', 'both']:
        yhat = 100.* dist.cdf(yhat)
    if fitprobs in ['x', 'both']:
        xhat = 100.* dist.cdf(xhat)

    return xhat, yhat, results


def estimateFromLineParams(xdata, slope, intercept, xlog=False, ylog=False):
    """ Estimate the dependent of a linear fit given x-data and linear
    parameters.

    Parameters
    ----------
    xdata : numpy array or pandas Series/DataFrame
        The input independent variable of the fit
    slope : float
        Slope of the best-fit line
    intercept : float
        y-intercept of the best-fit line
    xlog, ylog : bool (default = False)
        Toggles whether or not the logs of the x- or y- data should be
        used to perform the regression.

    Returns
    -------
    yhat : same type as xdata
        Estimate of the dependent variable.

    """

    x = numpy.array(xdata)
    if ylog:
        if xlog:
            yhat = numpy.exp(intercept) * x  ** slope
        else:
            yhat = numpy.exp(intercept) * numpy.exp(slope) ** x

    else:
        if xlog:
            yhat = slope * numpy.log(x) + intercept

        else:
            yhat = slope * x + intercept

    return yhat


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
    test2 =  numpy.min(interval2) <= numpy.min(interval1) <= numpy.max(interval2)

    # TODO: try test1 or test2 or (not oneway and test3)

    if oneway:
        return test1 or test2
    else:
        test3 = checkIntervalOverlap(interval2, interval1, oneway=True)
        return test1 or test2 or test3


def winsorize_dataframe(dataframe, **limits):
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
    df = dataframe.copy()
    for colname, limit in limits.items():
        df[colname] = stats.mstats.winsorize(df[colname], limits=limit)

    return df
