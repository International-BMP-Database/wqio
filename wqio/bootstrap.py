import warnings

import numpy
import scipy.stats as stats


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

    data = numpy.asarray(data)

    # intermediate values
    SSD = numpy.sum((data.mean() - data)**3)
    SCD = numpy.sum((data.mean() - data)**2)

    # dodge the ZeroDivision error
    if SCD == 0:
        SCD = 1e-12

    # compute and return the acceleration
    acc = SSD / (6 * SCD**1.5)
    return acc


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
    array([ 1.89528037,  3.16395883])

    """

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
        CI = stats.scoreatpercentile(boot_stats, new_alpha)

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
    (2.583930072904967, array([ 2.20960993,  3.33181602]))
    """

    index = _make_boot_index(data.shape[0], niter)
    boot_stats = statfxn(data[index], axis=-1)

    # compute the `alpha/2` and `1-alpha/2` percentiles of `boot_stats`
    CI = numpy.percentile(boot_stats, [alpha * 50, 100 - (alpha * 50)], axis=0)

    return CI
