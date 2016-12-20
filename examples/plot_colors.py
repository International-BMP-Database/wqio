# -*- coding: utf-8 -*-
"""
Colormap example
================

Colors are important
"""
# Author: Michael Waskom
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 300)
xx, yy = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)

plt.figure()
plt.imshow(z)

plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('hot'))

plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('Spectral'),
           interpolation='none')

# Not needed for the Gallery.
# Only for direct execution
plt.show()
