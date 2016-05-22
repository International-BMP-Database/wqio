# wqio - Water Quality, Inflow/Outflow
[![Build Status](https://travis-ci.org/Geosyntec/wqio.svg?branch=master)](https://travis-ci.org/Geosyntec/wqio)
[![Coverage Status](https://coveralls.io/repos/Geosyntec/wqio/badge.svg?branch=master)](https://coveralls.io/r/Geosyntec/wqio?branch=master)
[![Binstar Badge](https://binstar.org/phobson/wqio/badges/version.svg)](https://binstar.org/phobson/wqio)
[![Binstar Badge](https://binstar.org/phobson/wqio/badges/license.svg)](https://binstar.org/phobson/wqio)


## Installation

### Stable
Through conda:

    $ conda install wqio --channel=phobson

Through pip (usually out of date)

    $ pip install wqio

### Development

From source:

    $ git clone git@github.com:Geosyntec/wqio.git
    $ cd wqio
    $ pip install -e .

## Running tests

### Install pytest and plugins

    $ conda install pytest pytest-cov
    $ conda install pytest-mpl --channel=conda-forge

### Run the tests with the test runner:

    $ python check_wqio.py --mpl --cov

### Run the tests from within python

    $ python
    >>> import wqio
    >>> wqio.test('--mpl', '--cov')
