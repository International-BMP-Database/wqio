from matplotlib.scale import register_scale

from .misc import *
from .exceptions import DataError
from . import bootstrap
from . import ros
from .probscale import ProbScale
from . import figutils

from ..testing import NoseWrapper
test = NoseWrapper().test

register_scale(ProbScale)
