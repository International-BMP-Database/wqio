import itertools
from collections import namedtuple
from collections.abc import Callable
from textwrap import dedent

import numpy
import pandas
import statsmodels.api as sm
from probscale.algo import _estimate_from_fit
from scipy import stats
from scipy.stats._hypotests import TukeyHSDResult

from wqio import validate
from wqio.utils import misc

TheilStats = namedtuple("TheilStats", ("slope", "intercept", "low_slope", "high_slope"))
DunnResult = namedtuple("DunnResult", ("rank_stats", "results", "scores"))


def sig_figs(x, n, expthresh=5, tex=False, pval=False, forceint=False):
    """Formats a number with the correct number of sig figs.

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
    >>> print(sig_figs(1247.15, 3))
    1,250

    >>> print(sig_figs(1247.15, 7))
    1,247.150

    """
    # return a string value unaltered
    if isinstance(x, str):
        out = x

    # check on the number provided
    elif x is not None and not numpy.isinf(x) and not numpy.isnan(x):
        # check on the sig_figs
        if n < 1:
            raise ValueError("number of sig figs must be greater than zero!")

        elif pval and x < 0.001:
            out = "<0.001"
            if tex:
                out = f"${out}$"

        elif forceint:
            out = f"{x:,.0f}"

        # logic to do all of the rounding
        elif x != 0.0:
            order = int(numpy.floor(numpy.log10(numpy.abs(x))))

            if -1.0 * expthresh <= order <= expthresh:
                decimal_places = int(n - 1 - order)

                if decimal_places <= 0:
                    out = f"{round(x, decimal_places):,.0f}"

                else:
                    fmt = f"{{0:,.{decimal_places:d}f}}"
                    out = fmt.format(x)

            else:
                decimal_places = int(n - 1)
                if tex:
                    fmt = rf"$%0.{decimal_places:d}f \times 10 ^ {{{order:d}}}$"
                    out = fmt % round(x / 10**order, decimal_places)
                else:
                    fmt = f"{{0:.{decimal_places:d}e}}"
                    out = fmt.format(x)

        else:
            out = str(round(x, n))

    # with NAs and INFs, just return 'NA'
    else:
        out = "NA"

    return out


def format_result(result, qualifier, sigfigs=3):
    """Formats a results with its qualifier

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
    >>> format_result(1.23, '<', sigfigs=4)
    '<1.230'

    """

    return f"{qualifier}{sig_figs(result, sigfigs)}"


def process_p_vals(pval):
    """Processes p-values into nice strings to reporting. When the
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
        out = "NA"
    elif 0 < pval < 0.001:
        out = format_result(0.001, "<", sigfigs=1)
    elif pval > 1 or pval < 0:
        raise ValueError(f"p-values must be between 0 and 1 (not {pval})")
    else:
        out = f"{pval:0.3f}"

    return out


def translate_p_vals(pval, as_emoji=True):
    """Translates ambiguous p-values into more meaningful emoji.

    Parameters
    ----------
    pval : float
    as_emoji : bool (default = True)
        Represent the p-vals as emoji. Otherwise, words will be used.

    Returns
    -------
    interpreted : string

    """

    if pval is None:
        interpreted = r"ಠ_ಠ" if as_emoji else "NA"
    elif pval <= 0.01:
        interpreted = r"¯\(ツ)/¯" if as_emoji else "maybe"
    elif 0.01 < pval <= 0.05:
        interpreted = r"¯\_(ツ)_/¯" if as_emoji else "maybe (weak)"
    elif 0.05 < pval <= 0.1:
        interpreted = r"¯\__(ツ)__/¯" if as_emoji else "maybe (very weak)"
    elif 0.1 < pval <= 1:
        interpreted = r"(╯°□°)╯︵ ┻━┻" if as_emoji else "nope"
    else:
        raise ValueError(f"p-values must be between 0 and 1 (not {pval})")

    return interpreted


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
    # field_names = list(AD._fields) + ["pvalue"]

    p = _anderson_darling_p_vals(AD, len(data))
    values = AD._asdict()
    values["pvalue"] = p

    ADResult = namedtuple("ADResult", values)
    return ADResult(**values)


def process_AD_result(ad_result):
    """Return a nice string of Anderson-Darling test results

    Parameters
    ----------
    ad_result : tuple or namedtuple
        The packed output from scipy.stats.anderson

    Returns
    -------
    result : str
        A string representation of the confidence in the result.

    """

    AD, crit, sig = ad_result
    try:
        ci = 100 - sig[crit < AD][-1]
        return f"{ci:0.1f}%"
    except IndexError:
        ci = 100 - sig[0]
        return f"<{ci:0.1f}%"


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


def normalize_units(
    df,
    unitsmap,
    targetunit,
    paramcol="parameter",
    rescol="res",
    unitcol="units",
    napolicy="ignore",
):
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

    if napolicy == "raise":
        msg = ""
        if target.isnull().any():
            nulls = df[target.isnull()][paramcol].unique()
            msg += f"Some target units could not be mapped to the {paramcol} column ({nulls})\n"

        if normalization.isnull().any():
            nulls = df[normalization.isnull()][unitcol].unique()
            msg += (
                "Some normalization factors could not "
                f"be mapped to the {unitcol} column ({nulls})\n"
            )

        if conversion.isnull().any():
            nulls = target[conversion.isnull()]
            msg += f"Some conversion factors could not be mapped to the target units ({nulls})"

        if len(msg) > 0:
            raise ValueError(msg)

    # convert results
    normalized = df.assign(**{rescol: df[rescol] * normalization / conversion, unitcol: target})

    return normalized


def pH_to_concentration(pH, *args):
    """Converts pH values to proton concentrations in mg/L

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
        raise ValueError(f"pH = {pH} but must be between 0 and 14")

    # avogadro's number (items/mole)
    avogadro = 6.0221413e23

    # mass of a proton (kg)
    proton_mass = 1.672621777e-27

    # grams per kilogram
    kg2g = 1000

    # milligrams per gram
    g2mg = 1000

    return 10 ** (-1 * pH) * avogadro * proton_mass * kg2g * g2mg


def compute_theilslope(y, x=None, alpha=0.95, percentile=50):
    f""" Adapted from stats.mstats.theilslopes so that we can tweak the
    `percentile` parameter.
    https://goo.gl/nxPF54
    {dedent(stats.mstats.theilslopes.__doc__)}
    """

    # We copy both x and y so we can use _find_repeats.
    y = numpy.array(y).flatten()
    if x is None:
        x = numpy.arange(len(y), dtype=float)
    else:
        x = numpy.array(x, dtype=float).flatten()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ({len(y)} != {len(x)})")

    # Compute sorted slopes only when deltax > 0
    deltax = x[:, numpy.newaxis] - x
    deltay = y[:, numpy.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    slopes.sort()
    outslope = numpy.percentile(slopes, percentile)
    outinter = numpy.percentile(y, percentile) - outslope * numpy.median(x)
    # Now compute confidence intervals
    if alpha > 0.5:
        alpha = 1.0 - alpha

    z = stats.distributions.norm.ppf(alpha / 2.0)

    # This implements (2.6) from Sen (1968)
    _, nxreps = stats.mstats.find_repeats(x)
    _, nyreps = stats.mstats.find_repeats(y)
    nt = len(slopes)  # N in Sen (1968)
    ny = len(y)  # n in Sen (1968)

    # Equation 2.6 in Sen (1968):
    sigsq = (
        1
        / 18.0
        * (
            ny * (ny - 1) * (2 * ny + 5)
            - numpy.sum([k * (k - 1) * (2 * k + 5) for k in nxreps])
            - numpy.sum([k * (k - 1) * (2 * k + 5) for k in nyreps])
        )
    )

    # Find the confidence interval indices in `slopes`
    sigma = numpy.sqrt(sigsq)
    Ru = min(int(numpy.round((nt - z * sigma) / 2.0)), len(slopes) - 1)
    Rl = max(int(numpy.round((nt + z * sigma) / 2.0)) - 1, 0)
    delta = slopes[[Rl, Ru]]

    return TheilStats(outslope, outinter, delta[0], delta[1])


def fit_line(x, y, xhat=None, fitprobs=None, fitlogs=None, dist=None, through_origin=False):
    """Fits a line to x-y data in various forms (raw, log, prob scales)

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

    if fitprobs in ["x", "both"]:
        x = dist.ppf(x / 100.0)
        xhat = dist.ppf(numpy.array(xhat) / 100.0)

    if fitprobs in ["y", "both"]:
        y = dist.ppf(y / 100.0)

    if fitlogs in ["x", "both"]:
        x = numpy.log(x)

    if fitlogs in ["y", "both"]:
        y = numpy.log(y)

    x = sm.add_constant(x)

    # fix the y-intercept at 0 if requested
    # note that this is a nuance of the statsmodels package.
    # `x[:, 0] = 5` doesn't fix the intercept at 5, for instance.
    if through_origin:
        x[:, 0] = 0

    results = sm.OLS(y, x).fit()
    yhat = _estimate_from_fit(
        xhat,
        results.params[1],
        results.params[0],
        xlog=fitlogs in ["x", "both"],
        ylog=fitlogs in ["y", "both"],
    )

    if fitprobs in ["y", "both"]:
        yhat = 100.0 * dist.cdf(yhat)
    if fitprobs in ["x", "both"]:
        xhat = 100.0 * dist.cdf(xhat)

    return xhat, yhat, results


def check_interval_overlap(interval1, interval2, oneway=False, axis=None):
    """Checks if two numeric intervals overlaps.

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

    first_check = numpy.bitwise_or(
        numpy.bitwise_and(
            numpy.min(interval2, axis=axis) <= numpy.max(interval1, axis=axis),
            numpy.max(interval1, axis=axis) <= numpy.max(interval2, axis=axis),
        ),
        numpy.bitwise_and(
            numpy.min(interval2, axis=axis) <= numpy.min(interval1, axis=axis),
            numpy.min(interval1, axis=axis) <= numpy.max(interval2, axis=axis),
        ),
    )

    if oneway:
        return first_check
    else:
        return first_check | check_interval_overlap(interval2, interval1, oneway=True, axis=axis)


def winsorize_dataframe(df, **limits):
    """Winsorizes columns in a dataframe

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
    """Removes outliers from an array based on a scaling of the
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
    pctl25 = numpy.percentile(x, 25)
    pctl75 = numpy.percentile(x, 75)
    IQR = pctl75 - pctl25
    return x[(x >= pctl25 - (factor * IQR)) & (x <= pctl75 + (factor * IQR))]


def _comp_stat_generator(
    df: pandas.DataFrame,
    groupcols: list[str],
    pivotcol: str,
    rescol: str,
    statfxn: Callable,
    statname: str = "stat",
    pbarfxn: Callable = misc.no_op,
    **statopts,
):
    """Generator of records containing results of comparitive
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
        statname = "stat"

    groups = df.groupby(by=groupcols)
    for name, g in pbarfxn(groups):
        stations = g[pivotcol].unique()
        name = validate.at_least_empty_list(name)
        for _x, _y in pbarfxn(itertools.permutations(stations, 2)):
            row = dict(zip(groupcols, name))

            station_columns = [pivotcol + "_1", pivotcol + "_2"]
            row.update(dict(zip(station_columns, [_x, _y])))

            x = g.loc[g[pivotcol] == _x, rescol].values
            y = g.loc[g[pivotcol] == _y, rescol].values

            stat = statfxn(x, y, **statopts)
            row.update({statname: stat[0], "pvalue": stat.pvalue})
            yield row


def _group_comp_stat_generator(
    df: pandas.DataFrame,
    groupcols: list[str],
    pivotcol: str,
    rescol: str,
    statfxn: Callable,
    pbarfxn: Callable = misc.no_op,
    statname: str = "stat",
    control: str | None = None,
    **statopts,
):
    groupcols = validate.at_least_empty_list(groupcols)

    for names, main_group in pbarfxn(df.groupby(by=groupcols)):
        subsets = {
            pivot: subgroup[rescol].to_numpy()
            for pivot, subgroup in main_group.groupby(by=pivotcol)
        }

        _pivots = list(subsets.keys())
        _samples = list(subsets.values())

        if control is not None:
            control_sample = _samples.pop(_pivots.index(control))
            result = statfxn(*_samples, control=control_sample, **statopts)

        else:
            result = statfxn(*_samples, **statopts)

        row = {**dict(zip(groupcols, names)), statname: result.statistic, "pvalue": result.pvalue}
        yield row


def _paired_stat_generator(
    df: pandas.DataFrame,
    groupcols: list[str],
    pivotcol: str,
    rescol: str,
    statfxn: Callable,
    statname: str = "stat",
    pbarfxn: Callable = misc.no_op,
    **statopts,
):
    """Generator of records containing results of comparitive
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
        statname = "stat"

    groups = df.groupby(level=groupcols)[rescol]
    for name, g in pbarfxn(groups):
        stations = g.columns.tolist()
        name = validate.at_least_empty_list(name)
        for _x, _y in pbarfxn(itertools.permutations(stations, 2)):
            row = dict(zip(groupcols, name))

            station_columns = [pivotcol + "_1", pivotcol + "_2"]
            row.update(dict(zip(station_columns, [_x, _y])))

            _df = g[[_x, _y]].dropna()

            x = _df[_x].values
            y = _df[_y].values

            stat = statfxn(x, y, **statopts)
            row.update({statname: stat[0], "pvalue": stat.pvalue})
            yield row


def _tukey_res_to_df(
    names: list[str], hsd_res: list[TukeyHSDResult], group_prefix: str
) -> pandas.DataFrame:
    """Converts Scipy's TukeyHSDResult to a dataframe

    Parameters
    ----------
    names : list of str
        Name of the groups present in the Tukey HSD Results
    hsd_res : list of TukeyHSDResult
        List of Tukey results to be converted to a dateframe
    group_prefix : str (default = "Loc")
        Prefix that describes the nature of the groups

    Returns
    -------
    hsd_df : pandas.DataFrame

    """
    rows = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i != j:
                ci_bands = hsd_res.confidence_interval()
                row = {
                    f"{group_prefix} 1": names[n1],
                    f"{group_prefix} 2": names[n2],
                    "HSD Stat": hsd_res.statistic[i, j],
                    "p-value": hsd_res.pvalue[i, j],
                    "CI-Low": ci_bands.low[i, j],
                    "CI-High": ci_bands.high[i, j],
                }

                rows.append(row)

    df = pandas.DataFrame(rows).set_index([f"{group_prefix} 1", f"{group_prefix} 2"])
    return df


def tukey_hsd(
    df: pandas.DataFrame,
    rescol: str,
    compcol: str,
    paramcol: str,
    *othergroups: str,
):
    """
    Run the Tukey HSD Test on a dataframe based on groupings

    Parameters
    ----------
    df : pandas.DataFrame
    rescol : str
        Name of the column that contains the values of interest
    compcol: str
        Name of the column that defines the groups to be compared
        (i.e., treatment vs control)
    paramcol: str
        Name of the column that contains the measured parameter
    *othergroups : str
        Names of any other columsn that need to considered when
        defining the groups to be considered.

    Returns
    -------
    hsd_df : pandas.DataFrame

    """
    groupcols = [paramcol, *othergroups]
    scores = []
    for name, g in df.groupby(by=groupcols):
        locs = {loc: subg[rescol].values for loc, subg in g.groupby(compcol) if subg.shape[0] > 1}
        subset_names = {n: loc for n, loc in enumerate(locs)}
        res = stats.tukey_hsd(*[v for v in locs.values()])
        df_res = _tukey_res_to_df(subset_names, res, group_prefix=compcol)

        keys = {g: n for g, n in zip(groupcols, name)}
        scores.append(
            df_res.assign(
                is_diff=lambda df: df["p-value"].lt(0.05).astype(int),
                sign_of_diff=lambda df: numpy.sign(df["HSD Stat"]).astype(int),
                score=lambda df: df["is_diff"] * df["sign_of_diff"],
                **keys,
            ).set_index(groupcols, append=True)
        )

    return pandas.concat(scores, ignore_index=False, axis="index")


def process_tukey_hsd_scores(
    hsd_df: pandas.DataFrame, compcol: str, paramcol: str
) -> pandas.DataFrame:
    """
    Converts a Tukey HSD Results dataframe into scores that describe
    the value of groups' magnitude relative to each other.

    Generally speaking:
    * -7 to -5 -> significantly lower
    * -5 to -3 -> moderately lower
    * -3 to -1 -> slightly lower
    * -1 to +1 -> neutral
    * +1 to +3 -> slightly higher
    * +3 to +5 -> moderately higher
    * +5 to +7 -> significantly higher

    Parameters
    ----------
    hsd_df : pandas.DataFrame
        Dataframe dumped by `tukey_hsd`
    group_prefix : str (default = "Loc")
        Prefix that describes the nature of the groups

    Returns
    -------
    scores : pandas.DataFrame

    """
    return (
        hsd_df["score"]
        .unstack(level=f"{compcol} 2")
        .fillna(0)
        .groupby(level=paramcol, as_index=False, group_keys=False)
        .apply(lambda g: g.sum(axis="columns"))
        .unstack(level=f"{compcol} 1")
    )


def rank_stats(
    df: pandas.DataFrame,
    rescol: str,
    compcol: str,
    *othergroups: str,
) -> pandas.DataFrame:
    """
    Compute basic rank statistics (count, mean, sum) of a dataframe
    based on logical groupings. Assumes a single parameter is in the
    dataframe

    Parameters
    ----------
    df : pandas.DataFrame
    rescol : str
        Name of the column that contains the values of interest
    compcol: str
        Name of the column that defines the groups to be compared
        (i.e., treatment vs control)

    *othergroups : str
        Names of any other columsn that need to considered when
        defining the groups to be considered.

    """

    return (
        df.assign(rank=lambda df: df[rescol].rank())
        .groupby([compcol, *othergroups])["rank"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )


def _dunn_result(rs: pandas.DataFrame, group_prefix: str):
    threshold = stats.norm.isf(0.001667 / 2)
    N = rs["count"].sum()
    results = (
        rs.join(rs, how="cross", lsuffix="_1", rsuffix="_2")
        .assign(
            diff=lambda df: df["mean_1"] - df["mean_2"],
            scale=lambda df: ((N * (N + 1) / 12) * (1 / df["count_1"] + 1 / df["count_2"])) ** 0.5,
            dunn_z=lambda df: df["diff"] / df["scale"],
            score=lambda df: numpy.where(
                numpy.abs(df["dunn_z"]) <= threshold, 0, numpy.sign(df["dunn_z"])
            ).astype(int),
        )
        .loc[:, [f"{group_prefix}_1", f"{group_prefix}_2", "dunn_z", "score"]]
    )

    return results


def _dunn_scores(dr: pandas.DataFrame, group_prefix: str) -> pandas.DataFrame:
    scores = dr.pivot(index=f"{group_prefix}_2", columns=f"{group_prefix}_1", values="score").sum()
    return scores


def dunn_test(df: pandas.DataFrame, rescol: str, compcol: str, *othergroups: str) -> DunnResult:
    rs = rank_stats(df, rescol, compcol, *othergroups)
    results = _dunn_result(rs, compcol)
    scores = _dunn_scores(results, compcol)
    return DunnResult(rs, results, scores)
