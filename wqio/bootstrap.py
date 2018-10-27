import warnings
from collections import namedtuple

import numpy
import scipy.stats as stats
from probscale.algo import _estimate_from_fit


fitestimate = namedtuple('BootstrappedFitEstimate',
                         ['xhat', 'yhat', 'lower', 'upper', 'xlog', 'ylog'])


__all__ = ['BCA', 'percentile']


def _acceleration(data):
    """ Compute the acceleration statistic.

    Parameters
    ----------
    data : array-like
        Sequence of values whose acceleration will be computed.

    Returns
    -------
    acc : float
        The acceleration statistic.

    """

    # intermediate values
    sumcube_resids = ((data.mean() - data)**3).sum()

    # dodge the ZeroDivision error
    sumsqr_resids = max(((data.mean() - data)**2).sum(), 1e-12)

    # compute and return the acceleration
    return sumcube_resids / (6 * sumsqr_resids**1.5)


def _make_boot_index(elements, niter):
    """ Generate an array of bootstrap sample sets

    Parameters
    ----------
    elements : int
        The number of rows in the original dataset.
    niter : int
        Number of iteration for the bootstrapping.

    Returns
    -------
    index : numpy array
        A collection of random *indices* that can be used to randomly
        sample a dataset ``niter`` times.

    """
    return numpy.random.randint(low=0, high=elements, size=(niter, elements))


def BCA(data, statfxn, niter=10000, alpha=0.05):
    """
    Estimates confidence intervals around a statistic using the
    Bias-Corrected and Accelerated method.

    Parameters
    ----------
    data : array-like
        Input sequence of values
    statfxn : callable
        A reducing function that returns a single value when passed
        ``data``.
    niter : int, optional (default = 10000)
        The number of iterations for which the data will be resampled
        and the statistic recomputed.
    alpha : float, optional (default = 0.05)
        The desired confidence interval subtracted from 1. For example,
        if you want the 95% CI, use ``alpha = 0.05``.

    Returns
    -------
    CI : numpy array
        Confidence intervals around the statistic.

    Examples
    --------
    >>> import numpy
    >>> import wqio
    >>> numpy.random.seed(0)
    >>> # generate fake data and bootstrap it
    >>> data = numpy.random.normal(loc=2, scale=1.75, size=37)
    >>> wqio.bootstrap.BCA(data, numpy.mean, niter=1000, alpha=0.05)
    array([1.89528037, 3.16395883])

    """

    data = numpy.asarray(data)

    index = _make_boot_index(data.shape[0], niter)
    boot_stats = statfxn(data[index], axis=-1)
    primary_result = statfxn(data)
    boot_result = boot_stats.mean()

    # number of results below the premlinary estimate
    NumBelow = numpy.sum(boot_stats < primary_result)
    if NumBelow == 0:
        NumBelow = 0.00001

    # compute the acceleration
    a_hat = _acceleration(data)

    if NumBelow == niter:
        warnings.warn("All results below primary_result", UserWarning)
        CI = percentile(data, statfxn, niter, alpha=alpha)

    # z-stats on the % of `NumBelow` and the confidence limits
    else:
        z0 = stats.norm.ppf(NumBelow / niter)
        z = stats.norm.ppf([0.5 * alpha, 1 - (0.5 * alpha)])

        # refine the confidence limits (alphas)
        zTotal = (z0 + (z0 + z) / (1 - a_hat * (z0 + z)))
        new_alpha = stats.norm.cdf(zTotal) * 100.0

        # confidence intervals from the new alphas
        CI = numpy.percentile(boot_stats, new_alpha)

        # fall back to the standard percentile method if the results
        # don't make any sense
        if boot_result < CI[0] or CI[1] < boot_result:
            warnings.warn("secondary result outside of CI", UserWarning)
            CI = percentile(data, statfxn, niter, alpha=alpha)

    return CI


def percentile(data, statfxn, niter=10000, alpha=0.05):
    """
    Estimates confidence intervals around a statistic using the
    percentile method.

    Parameters
    ----------
    data : array-like
        Input sequence of values
    statfxn : callable
        A reducing function that returns a single value when passed
        ``data``.
    niter : int, optional (default = 10000)
        The number of iterations for which the data will be resampled
        and the statistic recomputed.
    alpha : float, optional (default = 0.05)
        The desired confidence interval subtracted from 1. For example,
        if you want the 95% CI, use ``alpha = 0.05``.

    Returns
    -------
    CI : numpy array
        Confidence intervals around the statistic.

    Examples
    --------
    >>> import numpy
    >>> import wqio
    >>> numpy.random.seed(0)
    >>> # generate fake data and bootstrap it
    >>> data = numpy.random.normal(loc=2, scale=1.75, size=37)
    >>> wqio.bootstrap.percentile(data, numpy.median, niter=1000, alpha=0.05)
    array([2.20960993, 3.33181602])
    """

    index = _make_boot_index(data.shape[0], niter)
    boot_stats = statfxn(data[index], axis=-1)

    # compute the `alpha/2` and `1-alpha/2` percentiles of `boot_stats`
    CI = numpy.percentile(boot_stats, [alpha * 50, 100 - (alpha * 50)], axis=0)

    return CI


def fit(x, y, fitfxn, niter=10000, alpha=0.05, xlog=False, ylog=False,
        **kwargs):
    """
    Perform a percentile bootstrap estimate on a linear regression.

    Parameters
    ----------
    x, y : array-like
        Predictor and response variables, respectively.
    fitfxn : callable
        The function that performs the linear regression. It must return a
        sequence (length > 3) where each element is a scalar and the first and
        second elements are the slope and intercept (in that order). The first
        two parameters fed to the function must be ``x`` and ``y`` (in that
        order).
    niter : int, optional (default = 10000)
        Number of bootstrap iterations to perform.
    alpha : float, optional (default = 0.05)
        The confidence level to estimate the response variable. For example,
        when alpha = 0.05, the 97.5th and 2.5th perctile estimates will be
        returned.
    xlog, ylog : bool, optional (default = False)
        Toggles performing the fit and estimates in arithmetic or logarithmic
        space.

    Additional Parameters
    ---------------------
    Any additional keyword arguments will be directly passed to ``fitfxn``
    after ``x`` and ``y``.

    Returns
    -------
    BootstrappedFitEstimate : namedtuple of numpy.arrays
        Names include:

          xhat : the synthetic predictor variable
          yhat : the estimated response variable
          lower, upper : the confidence intervals around the estimated respone

    Examples
    --------
    >>> import numpy
    >>> from matplotlib import pyplot
    >>> from wqio import bootstrap
    >>> N = 10
    >>> x = numpy.arange(1, N + 1, dtype=float)
    >>> y = numpy.array([4.527, 3.519, 9.653, 8.036, 10.805,
    ...                  14.329, 13.508, 11.822, 13.281, 10.410])
    >>> bsfit = bootstrap.fit(x, y, numpy.polyfit, niter=2000,
    ...                       xlog=False, ylog=False,
    ...                       deg=1, full=False)
    >>> fig, ax = pyplot.subplots()
    >>> _pnts = ax.plot(x, y, 'k.', zorder=2)
    >>> _line = ax.plot(bsfit.xhat, bsfit.yhat, 'b-', zorder=1)
    >>> _ci = ax.fill_between(bsfit.xhat, bsfit.lower, bsfit.upper,
    ...                       zorder=0, alpha=0.5)

    """

    # get the data in order according to the *predictor* variable only
    sort_index = numpy.argsort(x)
    x = x[sort_index]
    y = y[sort_index]

    # apply logs if necessary
    if xlog:
        x = numpy.log(x)
    if ylog:
        y = numpy.log(y)

    # compute fit on original data
    main_params = fitfxn(x, y, **kwargs)

    # raw loop to estimate the bootstrapped fit parameters
    bs_params = numpy.array([
        fitfxn(x[ii], y[ii], **kwargs)
        for ii in _make_boot_index(len(x), niter)
    ])

    # un-log, if necesssary
    if xlog:
        x = numpy.exp(x)

    # compute estimate from original data fit
    yhat = _estimate_from_fit(x, main_params[0], main_params[1],
                              xlog=xlog, ylog=ylog)

    # full array of estimates
    bs_estimates = _estimate_from_fit(
        x[:, None], bs_params[:, 0], bs_params[:, 1],
        xlog=xlog, ylog=ylog
    )

    # both alpha alphas
    percentiles = 100 * numpy.array([alpha * 0.5, 1 - alpha * 0.5])

    # lower, upper bounds
    bounds = numpy.percentile(bs_estimates, percentiles, axis=1)

    return fitestimate(x, yhat, bounds[0], bounds[1], xlog, ylog)
