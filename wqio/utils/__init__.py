from matplotlib.scale import register_scale

from .misc import *
from .probscale import ProbScale
from . import figutils

from wqio.testing import NoseWrapper
test = NoseWrapper().test

register_scale(ProbScale)
