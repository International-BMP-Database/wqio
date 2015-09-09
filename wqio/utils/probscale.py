import numpy as np
import matplotlib
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase
from matplotlib.ticker import (
    FixedLocator,
    NullLocator,
    Formatter,
    NullFormatter,
    FuncFormatter
)
import matplotlib.pyplot as plt
from scipy import stats

from .misc import sigFigs


class _minimal_norm(object):
    def ppf(self, q):
        return np.sqrt(2) * inv_erf(2*q - 1)

    def cdf(self, x):
        return 0.5 * (1 + erf(x/np.sqrt(2)))


def _get_probs(nobs):
    '''Returns the x-axis labels for a probability plot based
    on the number of observations (`nobs`)
    '''
    order = int(np.floor(np.log10(nobs)))
    base_probs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

    axis_probs = base_probs.copy()
    for n in range(order):
        if n <= 2:
            lower_fringe = np.array([1, 2, 5])
            upper_fringe = np.array([5, 8, 9])
        else:
            lower_fringe = np.array([1])
            upper_fringe = np.array([9])

        new_lower = lower_fringe/10**(n)
        new_upper = upper_fringe/10**(n) + axis_probs.max()
        axis_probs = np.hstack([new_lower, axis_probs, new_upper])

    return axis_probs


class ProbFormatter(Formatter):
    def __call__(self, x, pos=None):
        if x < 10:
            out = sigFigs(x, 1)
        elif x <= 99:
            out =  sigFigs(x, 2)
        else:
            order = np.ceil(np.round(np.abs(np.log10(100 - x)), 6))
            out = sigFigs(x, order + 2)

        return '{}'.format(out)


class ProbTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist):
        Transform.__init__(self)
        self.dist = dist

    def transform_non_affine(self, a):
        return self.dist.ppf(a / 100.)

    def inverted(self):
        return InvertedProbTransform(self.dist)


class InvertedProbTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist):
        self.dist = dist
        Transform.__init__(self)

    def transform_non_affine(self, a):
        return self.dist.cdf(a) * 100.

    def inverted(self):
        return ProbTransform()


class ProbScale(ScaleBase):
    """
    A probability scale.  Care is taken so non-positive
    values are not plotted.

    For computational efficiency (to push as much as possible to Numpy
    C code in the common cases), this scale provides different
    transforms depending on the base of the logarithm:

    """
    name = 'prob'


    def __init__(self, axis, **kwargs):
        self.dist = kwargs.pop('dist', stats.norm)
        self._transform = ProbTransform(self.dist)


    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """
        axis.set_major_locator(FixedLocator(_get_probs(1e10)))
        axis.set_major_formatter(FuncFormatter(ProbFormatter()))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)
