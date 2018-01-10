"""
DataCollection: Data Attributes
===============================

This examples shows how to create a ``DataCollection`` and what
data it stores.

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
print(head_tail(df))


#####################################################################
# Create the data collection object, show some of the data attributes

dc = wqio.DataCollection(df, rescol='res', qualcol='qual',
                         stationcol='loc', paramcol='param',
                         ndval='<', othergroups=['state'],
                         pairgroups=['state', 'bmp'],
                         useros=True, filterfxn=None,
                         bsiter=10000)


#############################################################
# You can always get to the raw data you had when you created
# the object through the ``.raw_data`` attrbute.

print(head_tail(dc.raw_data))


#################################################################
# Most calculations of a ``DataCollection`` rely on the ``.data``
# attribute, which as a simple index and a boolean *censorship*
# column computed based on the values passed to ``qualcol`` and
# ``ndval`` when you created the object

print(head_tail(dc.data))


###############################################################
# From there, we can look at the ``.tidy`` attribute, which was
# computed by taking ``.data``, grouping by the ``stationcol``,
# ``paramcol``, and ``othergroups``, passing each group through
# the ROS function, and returning a simplified dataframe.

print(head_tail(dc.tidy))


################################################################
# There is also a ``.paired`` attribute computed by pivoting the
# ``.data`` attribute on the ``stationcol``, using ``paramcol``
# and ``pairgroups`` columns as the index values. In otherwords,
# ``stationcol``, ``paramcol``, and ``pairgroups`` must be able
# to form a unique index on the dataframe for this attribute
# to work.

print(head_tail(dc.paired))
