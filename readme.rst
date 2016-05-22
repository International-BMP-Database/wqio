.. toctree::
   :maxdepth: 2

wqio - Water Quality, Inflow/Outflow
------------------------------------
.. image:: https://travis-ci.org/Geosyntec/wqio.svg?branch=master
    :target: https://travis-ci.org/Geosyntec/wqio

.. image:: https://coveralls.io/repos/github/Geosyntec/wqio/badge.svg?branch=master
    :target: https://coveralls.io/github/Geosyntec/wqio?branch=master

.. image:: https://anaconda.org/phobson/wqio/badges/version.svg
    :target: https://anaconda.org/phobson/wqio

.. image:: https://anaconda.org/phobson/wqio/badges/license.svg
    :target: https://anaconda.org/phobson/wqio

.. toctree::
   :maxdepth: 2

Installation
============

.. toctree::
   :maxdepth: 2

Stable
~~~~~~

Through conda::

    $ conda install wqio --channel=phobson

Through pip (usually out of date)::

    $ pip install wqio

Development
~~~~~~~~~~~

From source::

    $ git clone git@github.com:Geosyntec/wqio.git
    $ cd wqio
    $ pip install -e .

Running tests
=============

.. toctree::
   :maxdepth: 2

Install pytest and plugins::

    $ conda install pytest pytest-cov
    $ conda install pytest-mpl --channel=conda-forge

Run the tests with the test runner::

    $ python check_wqio.py --mpl --cov

Run the tests from within python::

    $ python
    >>> import matploltib as mpl; mpl.use('agg')
    >>> import wqio
    >>> wqio.test('--mpl', '--cov')
