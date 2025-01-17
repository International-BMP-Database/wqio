# Setup script for the wqio package
#
# Usage: python setup.py install
#
from setuptools import find_packages, setup

DESCRIPTION = "wqio: Water Quality Inflow/Outflow"
LONG_DESCRIPTION = DESCRIPTION
NAME = "wqio"
VERSION = "0.7.2"
AUTHOR = "Paul Hobson (Herrera Environmental Consultants)"
AUTHOR_EMAIL = "phobson@herrerainc.com"
URL = "https://github.com/International-BMP-Database/wqio"
DOWNLOAD_URL = "https://github.com/International-BMP-Database/wqio/archive/main.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 3.10 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
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
