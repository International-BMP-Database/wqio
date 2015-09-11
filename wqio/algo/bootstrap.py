import numpy as np
import scipy.stats as stats
import scipy.stats.distributions as dist
import scipy.optimize as opt


__all__ = ['Stat', 'Fit']


class _bootstrapMixin(object):
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

    Attributes
    ----------
    data : numpy array
        Array of the data that we're describing
    statfxn : function or lambda
        Function that takes `inputdata` as the sole argument and return
        a single float value.
    alpha : float
        The uncertainty level of the confidence intervals
    NIter : int
        The number of interation used in the bootstrapping routine
    prelim_result : float
        Estimate of the statistic based on the original dataset

    """

    def _acceleration(self):
        """ Compute the acceleration statistic.

        Parameters
        ----------
        None

        Returns
        -------
        acc : float
            The acceleration statistic.

        """

        # intermediate values
        SSD = np.sum((self.data.mean() - self.data)**3)
        SCD = np.sum((self.data.mean() - self.data)**2)

        # dodge the ZeroDivision error
        if SCD == 0:
            SCD = 1e-12

        # comput and return the acceleration
        acc = SSD / (6 * SCD**1.5)
        return acc

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

    def _eval_BCA(self, prelim_result, boot_stats):
        """ Evaluate the Bias-Corrected and Accelerated method of
        aquiring confidence intervals around a statistic.

        Parameters
        ----------
        prelim_result :float
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
        NumBelow = np.sum(boot_stats < prelim_result)
        if NumBelow == 0:
            NumBelow = 0.00001

        # compute the acceleration
        a_hat = self._acceleration()

        # z-stats on the % of `NumBelow` and the confidence limits
        if NumBelow != self.NIter:
            z0 = dist.norm.ppf(float(NumBelow)/self.NIter)
            z1 = dist.norm.ppf(self.alpha/2.0)
            z2 = dist.norm.ppf(1-self.alpha/2.0)

            # refine the confidence limits (alphas)
            z1Total = (z0 + (z0 + z1)) / (1 - a_hat*(z0+z1))
            z2Total = (z0 + (z0 + z2)) / (1 - a_hat*(z0+z2))
            alpha1 = dist.norm.cdf(z1Total)*100.0
            alpha2 = dist.norm.cdf(z2Total)*100.0

            # take the mean of the `boot_stats`
            result = boot_stats.mean()

            # confidence intervals from the new alphas
            CI = np.array([stats.scoreatpercentile(boot_stats, alpha1),
                          stats.scoreatpercentile(boot_stats, alpha2)])

            # fall back to the standard percentile method if the results
            # don't make any sense
            # TODO: fallback to percentile method should raise a warning
            if result < CI[0] or CI[1] < result:
                result, CI = self._eval_percentile(boot_stats)
        else:
            result, CI = self._eval_percentile(boot_stats)

        return result, CI

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
        CI = np.array([
            np.percentile(boot_stats, self.alpha*50, axis=0),
            np.percentile(boot_stats, 100-self.alpha*50, axis=0),
        ])

        # median of `boot_stats`
        result = np.percentile(boot_stats, 50, axis=0)

        return result, CI


class Stat(_bootstrapMixin):
    def __init__(self, inputdata, statfxn=np.median, alpha=0.05, NIter=5000):
        self.data = inputdata
        self.statfxn = statfxn
        self.alpha = alpha
        self.NIter = NIter
        self._boot_array = self._make_bootstrap_array()
        self._setup()

    def _setup(self):
        '''
        Utility method to setup the _bootstrapMixin object's attributes of the
            preliminary results and the boot strapped
        '''
        self.prelim_result = self.statfxn(self.data)
        self._boot_stats = self.statfxn(self._boot_array, axis=1)

    def BCA(self):
        """ BCA method of aquiring confidence intervals. """
        return self._eval_BCA(self.prelim_result, self._boot_stats)

    def percentile(self):
        """ Percentile method of aquiring confidence intervals. """
        return self._eval_percentile(self._boot_stats)


class Fit(_bootstrapMixin):
    def __init__(self, inputdata, outputdata, curvefitfxn,
                 statfxn=opt.curve_fit, alpha=0.05, NIter=5000):
        self.data = np.array(inputdata, dtype=np.float64)
        self.outputdata = np.array(outputdata, dtype=np.float64)
        self.curvefitfxn = curvefitfxn
        self.statfxn = statfxn
        self.alpha = alpha
        self.NIter = NIter
        self._boot_array = self._make_bootstrap_array()
        self._setup()

    def _setup(self):
        '''
        Utility method to setup the _boot_strap object's attributes of the
            preliminary results and the boot strapped
        '''
        # prelim results
        self.prelim_result, pcov = self.statfxn(self.curvefitfxn,
                                                self.data,
                                                self.outputdata)

        # setup bootstrap stats array
        self._boot_stats = np.empty([self.NIter, self.prelim_result.shape[0]])

        # fill in the results
        for r, boot in enumerate(self._boot_array):
            fitparams, covariance = self.statfxn(self.curvefitfxn,
                                                 boot[:, 0], boot[:, 1])
            self._boot_stats[r] = fitparams

    def BCA(self):
        """ BCA method of aquiring confidence intervals. """
        # empty arrays of results and CIs for each
        # fit parameter
        results = np.empty(self.prelim_result.shape[0])
        CI = np.empty([self.prelim_result.shape[0], 2])

        # use BCA to estimate each parameter
        for n, param in enumerate(self.prelim_result):
            res, ci = self._eval_BCA(param, self._boot_stats[:, n])
            results[n] = res
            CI[n] = ci

        return results, CI

    def percentile(self):
        """ Percentile method of aquiring confidence intervals. """
        # empty arrays of results and CIs for each
        # fit parameter
        results = np.empty([self._boot_stats.shape[1]])
        CI = np.empty([self._boot_stats.shape[1], 2])

        # use percentiles to estimate each parameter
        for n, bstat in enumerate(self._boot_stats.T):
            res, ci = self._eval_percentile(bstat)
            results[n] = res
            CI[n] = ci

        return results, CI
