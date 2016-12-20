"""
Basic of a DataCollection
=========================

This examples show how to create a ``DataCollection``, access some
of its statistics, and pipe the data into seaborn plotting functions.

"""

import numpy
import seaborn
from matplotlib import pyplot

import wqio
from wqio.tests import helpers
from wqio.utils import head_tail

seaborn.set(style="ticks", context="talk")


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


##################
# ROS'd, tidy data

head_tail(dc.tidy)


########################
# Non-ROS'd, paired data

head_tail(dc.paired)


############################
# Show how much data we have
head_tail(dc.count)


#######################
# Show some basic stats

head_tail(dc.median)


####################################
# Show some hypothesis tests results

head_tail(dc.wilcoxon)


################################
# Use seaborn to make some plots
# This is basic boxplot

fg = seaborn.factorplot(
    x='state', y='ros_res', data=dc.tidy,
    col='param', col_wrap=2, hue='loc', kind='box',
    aspect=1.75, size=3, margin_titles=True,
).set(yscale='log').set_ylabels('Concentration')


#############################
# This is a split violin plot.
# The dashed lines show the quartiles

fg = (
    dc.tidy.assign(log_ros_res=numpy.log10(dc.tidy['ros_res']))
      .pipe(
            (seaborn.factorplot, 'data'),
            x='state', y='log_ros_res',
            col='param', col_wrap=2,
            hue='loc', hue_order=['Inflow', 'Outflow'],
            kind='violin', split=True, inner='quartile',
            aspect=1.75, size=3, margin_titles=True,
            linewidth=1.0
      ).set_ylabels('Log10 of Concentration')
)
