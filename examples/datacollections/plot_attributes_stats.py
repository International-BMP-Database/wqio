"""
DataCollection: Statistics
==========================

This examples shows how to create a ``DataCollection`` and some of the
general statistics available.

"""

import numpy
import seaborn
from matplotlib import pyplot

import wqio
from wqio.tests import helpers
from wqio.utils import head_tail


#######################################
# Load example data and show a few rows

df = helpers.make_dc_data_complex()
head_tail(df)


#####################################################################
# Create the data collection object, show some of the data attributes

dc = wqio.DataCollection(df, rescol='res', qualcol='qual',
                         stationcol='loc', paramcol='param',
                         ndval='<', othergroups=['state'],
                         pairgroups=['state', 'bmp'],
                         useros=True, filterfxn=None,
                         bsiter=10000)


#############################################################
# In general, stats are computed on the ``.tidy`` attribute
# of the object so that it has access to the ROS'd data to
# better handle non-detects.
#
# Also, where applicable, confidence statistics are estimated
# using the bias-corrected and accelerated bootstrapping
# procedcure implemented in ``wqio.ros``.

head_tail(dc.mean)
