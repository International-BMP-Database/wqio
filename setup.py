# Setup script for the wqio package
#
# Usage: python setup.py install
#
from setuptools import setup, find_packages


DESCRIPTION = "wqio: Water Quality Inflow/Outflow"
LONG_DESCRIPTION = DESCRIPTION
NAME = "wqio"
VERSION = "0.5.1"
AUTHOR = "Paul Hobson (Geosyntec Consultants)"
AUTHOR_EMAIL = "phobson@geosyntec.com"
URL = "https://github.com/Geosyntec/wqio"
DOWNLOAD_URL = "https://github.com/Geosyntec/wqio/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 3.6 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "statsmodels",
    "seaborn",
    "probscale",
]
PACKAGE_DATA = {
    "wqio.tests._data": ["*.csv", "*.tex", "matplotlibrc"],
    "wqio.tests._baseline_images.features_tests": ["*png"],
    "wqio.tests._baseline_images.hydro_tests": ["*png"],
    "wqio.tests._baseline_images.samples_tests": ["*png"],
    "wqio.tests._baseline_images.utils_tests.figutils_tests": ["*png"],
}

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    platforms=PLATFORMS,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
)
