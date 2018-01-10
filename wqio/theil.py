import os
from textwrap import dedent

import numpy
from scipy import stats
from matplotlib import pyplot
from probscale.algo import _estimate_from_fit

from wqio import utils


class TheilSenFit(object):
    def __init__(self, infl, effl, log_infl=False, log_effl=False):
        """ Theil-Sen Fit object

        Parameters
        ----------
        infl, effl : array-like
            Influent and effluent result to regress.
        log_infl, log_effl : bool
            When True, performs the fit on log-transformed data.

        See also
        --------
        wqio.utils.compute_theilslope

        """
        self.influent_data = infl
        self.effluent_data = effl

        self.log_infl = log_infl
        self.log_effl = log_effl

        self._theil_stats = None

    @property
    def infl(self):
        if self.log_infl:
            return numpy.log(self.influent_data)
        else:
            return self.influent_data

    @property
    def effl(self):
        if self.log_effl:
            return numpy.log(self.effluent_data)
        else:
            return self.effluent_data

    @property
    def theil_stats(self):
        if self._theil_stats is None:
            try:
                self._theil_stats = utils.compute_theilslope(self.effl, x=self.infl)
            except:
                self._theil_stats = utils.TheilStats(None, None, None, None)

        return self._theil_stats

    @property
    def med_slope(self):
        return self.theil_stats[0]

    @property
    def intercept(self):
        return self.theil_stats[1]

    @property
    def low_slope(self):
        return self.theil_stats[2]

    @property
    def high_slope(self):
        return self.theil_stats[3]

    @property
    def x_fit(self):
        xmin = self.influent_data.min()
        xmax = self.effluent_data.max()
        if self.log_infl:
            return numpy.logspace(numpy.log10(xmin), numpy.log10(xmax))
        else:
            return numpy.linspace(xmin, xmax)

    @property
    def med_estimate(self):
        return _estimate_from_fit(self.influent_data, self.med_slope, self.intercept,
                                  xlog=self.log_infl, ylog=self.log_effl)

    @property
    def errors(self):
        if self.log_effl:
            return numpy.log(self.effluent_data) - numpy.log(self.med_estimate)
        else:
            return self.effluent_data - self.med_estimate

    @property
    def MAD(self):
        if self.log_effl:
            return numpy.median(numpy.exp(numpy.abs(self.errors)))
        else:
            return numpy.median(numpy.abs(self.errors))

    @property
    def BCF(self):
        if self.log_effl:
            return numpy.mean(numpy.exp(utils.remove_outliers(self.errors)))
        else:
            return numpy.mean(utils.remove_outliers(self.errors))


def all_theil(infl, effl):
    """
    Convenience function to create the relevant TheilSenFits objects
    for a dataset in various log/linear spaces. The case with linear
    influent data, but logarithmic effluent data is omitted.

    Parameters
    -----------
    infl, effl : array-like
        Influent and effluent result to regress.

    Returns
    -------
    all_tr : list of TheilSenFit instances
    best_tr : TheilSenFit
        The TheilSenFit with the smallest mean absolute deviatation
        in *all_tr*.

    """

    all_tsf = [
        TheilSenFit(infl, effl, log_infl=False, log_effl=False),
        TheilSenFit(infl, effl, log_infl=True, log_effl=False),
        TheilSenFit(infl, effl, log_infl=True, log_effl=True),
    ]
    best_tsf = min(all_tsf, key=lambda tr: tr.MAD)
    return all_tsf, best_tsf
