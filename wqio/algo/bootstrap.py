import numpy as np
import scipy.stats as stats
import scipy.stats.distributions as dist
import scipy.optimize as opt


__all__ = ['Stat', 'Fit']

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

    data = np.asarray(data)

    # intermediate values
    SSD = np.sum((data.mean() - data)**3)
    SCD = np.sum((data.mean() - data)**2)

    # dodge the ZeroDivision error
    if SCD == 0:
        SCD = 1e-12

    # comput and return the acceleration
    acc = SSD / (6 * SCD**1.5)
    return acc


class Stat(object):
    """ Class for using bootstrap techniques to estimate a statistic
    and its confidence intervals.

    Parameters
    ----------
    inputdata : array-like
        The data that we're describing.
    statfxn : callable, optional (default = numpy.median)
        Function that takes `inputdata` as the sole argument and return
        a single float value.
    alpha : float, optional (default = 0.05)
        The uncertainty level e.g., for 95% confidence intervals,
        `alpha = 0.05`.
    NIter : int, optional (default = 5000)
        The number of interation to use in the bootstrapping routine.
    use_prelim : bool, optional (default = True)
        When True, the statistic returned is computed from the original
        dataset. When False, the statistic is computed as the arithmetic
        mean of the bootstrapped array.

    """

    def __init__(self, inputdata, statfxn=np.median, alpha=0.05, NIter=5000,
                 use_prelim=True):
        self.data = np.array(inputdata, dtype=np.float64)
        self.statfxn = statfxn
        self.alpha = alpha
        self.NIter = NIter
        self.use_prelim = use_prelim

        self._primary_result = None
        self._boot_array = None
        self._boot_stats = None
        self._secondary_result = None

    @property
    def boot_array(self):
        if self._boot_array is None:
            self._boot_array = self._make_bootstrap_array()
        return self._boot_array

    @property
    def primary_result(self):
        if self._primary_result is None:
            self._primary_result = self.statfxn(self.data)
        return self._primary_result

    @property
    def boot_stats(self):
        if self._boot_stats is None:
            self._boot_stats = self.statfxn(self.boot_array, axis=1)
        return self._boot_stats

    @property
    def secondary_result(self):
        if self._secondary_result is None:
            self._secondary_result = self.boot_stats.mean()
        return self._secondary_result

    @property
    def final_result(self):
        if self.use_prelim:
            return self.primary_result
        else:
            return self.secondary_result

    def _make_bootstrap_array(self):
        """ Generate an array of bootstrap sample sets

        Parameters
        ----------
        None

        Returns
        -------
        bootArray : numpy array of floats
            A collection of random samples pulled from the the dataset.

        """

        # stack the data together if we're
        # bootstrapping a curve fig
        if hasattr(self, 'outputdata'):
            data = np.vstack([self.data, self.outputdata]).T
        else:
            data = self.data

        # number of data points
        N = data.shape[0]

        # shape of the output array
        shape = [self.NIter]
        shape.extend(data.shape)

        # create the output array
        bootArray = np.empty(shape, dtype=np.float64)

        # stuff a random samples (with replacement)
        # into the each row of the output array
        for n in range(self.NIter):
            index = np.random.random_integers(low=0, high=N-1, size=N)
            bootArray[n] = data[index]

        # we're done
        return bootArray

    def _eval_BCA(self, primary_result, boot_stats):
        """ Evaluate the Bias-Corrected and Accelerated method of
        aquiring confidence intervals around a statistic.

        Parameters
        ----------
        primary_result :float
            An estimate of the statistic computed from the full dataset.
        boot_stats : array-like
            Estimates of the statistic computed from iteratively
            resampling the dataset with replacement.

        Returns
        -------
        result : float
            Refined(?) estimate of the statistic.
        CI : numpy array
            Confidence intervals of statistic.

        """

        # number of results below the premlinary estimate
        NumBelow = np.sum(boot_stats < primary_result)
        if NumBelow == 0:
            NumBelow = 0.00001

        # compute the acceleration
        a_hat = _acceleration(self.data)

        # z-stats on the % of `NumBelow` and the confidence limits
        if NumBelow != self.NIter:
            z0 = dist.norm.ppf(NumBelow / self.NIter)
            z = dist.norm.ppf([0.5 * self.alpha, 1 - (0.5 * self.alpha)])

            # refine the confidence limits (alphas)
            zTotal = (z0 + (z0 + z) / (1 - a_hat*(z0 + z)))
            alpha = dist.norm.cdf(zTotal) * 100.0

            # confidence intervals from the new alphas
            CI = stats.scoreatpercentile(boot_stats, alpha)

            # fall back to the standard percentile method if the results
            # don't make any sense
            # TODO: fallback to percentile method should raise a warning
            if self.secondary_result < CI[0] or CI[1] < self.secondary_result:
                CI = self._eval_percentile(boot_stats)
        else:
            CI = self._eval_percentile(boot_stats)

        return CI

    def _eval_percentile(self, boot_stats):
        """ Evaluate the percentile method of aquiring confidence
        intervals around a statistic.

        Parameters
        ----------
        boot_stats : array-like
            Estimates of the statistic computed from iteratively
            resampling the dataset with replacement.

        Returns
        -------
        result : float
            Refined(?) estimate of the statistic.
        CI : numpy array
            Confidence intervals of statistic.

        """

        # compute the `alpha/2` and `1-alpha/2` percentiles of `boot_stats`
        CI = np.percentile(boot_stats, [self.alpha*50, 100-self.alpha*50], axis=0)

        return CI

    def BCA(self):
        """ BCA method of aquiring confidence intervals. """
        CI = self._eval_BCA(self.primary_result, self.boot_stats)
        return self.final_result, CI

    def percentile(self):
        """ Percentile method of aquiring confidence intervals. """
        CI = self._eval_percentile(self.boot_stats)
        return self.final_result, CI


class Fit(Stat):
    """ Class for using bootstrap techniques to estimate curve-fitting
    parameters and their confidence intervals.

    Parameters
    ----------
    inputdata, outputdata : array-like
        Input (NxM) and repsponse data (Nx1) that we're modeling.
    curvefitfxn : callable
        Function that performs the describes the type of curve we are
        try to fit. For example ``lambda x, m, b: m*x + b)``, where
        ``x`` is our input data and ``m``, ``b`` are the variables
        we're estimating.
    statfxn : callable, optional (default = numpy.median)
        Function that takes ``curvefitfxn``, ``inputdata``, and
        ``outputdata`` as arguments and returns the model parameters
        and the covariance value.
    alpha : float, optional (default = 0.05)
        The uncertainty level e.g., for 95% confidence intervals,
        `alpha = 0.05`.
    NIter : int, optional (default = 5000)
        The number of interation to use in the bootstrapping routine.

    """

    def __init__(self, inputdata, outputdata, curvefitfxn,
                 statfxn=opt.curve_fit, alpha=0.05, NIter=5000,
                 use_prelim=True):
        self.data = np.array(inputdata, dtype=np.float64)
        self.outputdata = np.array(outputdata, dtype=np.float64)
        self.curvefitfxn = curvefitfxn
        self.statfxn = statfxn
        self.alpha = alpha
        self.NIter = NIter
        self._primary_result = None
        self._boot_array = None
        self._boot_stats = None
        self._secondary_result = None
        self.use_prelim = use_prelim


    @property
    def primary_result(self):
        if self._primary_result is None:
            self._primary_result, self.pcov = self.statfxn(self.curvefitfxn,
                                                          self.data,
                                                          self.outputdata)

        return self._primary_result

    @property
    def boot_stats(self):
        if self._boot_stats is None:
            # setup bootstrap stats array
            self._boot_stats = np.empty([self.NIter, self.primary_result.shape[0]])

            # fill in the results
            for r, boot in enumerate(self.boot_array):
                fitparams, covariance = self.statfxn(self.curvefitfxn,
                                                     boot[:, 0], boot[:, 1])
                self._boot_stats[r] = fitparams
        return self._boot_stats

    @property
    def secondary_result(self):
        if self._secondary_result is None:
            self._secondary_result = self.boot_stats.mean()
        return self._secondary_result

    @property
    def final_result(self):
        if self.use_prelim:
            return self.primary_result
        else:
            return self.secondary_result


    def BCA(self):
        """ BCA method of aquiring confidence intervals. """
        # empty arrays CIs for each fit parameter
        CI = np.empty([self.primary_result.shape[0], 2])

        # use BCA to estimate each parameter
        for n, param in enumerate(self.primary_result):
            ci = self._eval_BCA(param, self.boot_stats[:, n])
            #results[n] = res
            CI[n] = ci

        return self.final_result, CI

    def percentile(self):
        """ Percentile method of aquiring confidence intervals. """
        # empty arrays of CIs for each fit parameter
        CI = np.empty([self.boot_stats.shape[1], 2])

        # use percentiles to estimate each parameter
        for n, bstat in enumerate(self.boot_stats.T):
            ci = self._eval_percentile(bstat)
            CI[n] = ci

        return self.final_result, CI
