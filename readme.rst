.. toctree::
   :maxdepth: 2

wqio - Water Quality, Inflow/Outflow
------------------------------------

.. toctree::
   :maxdepth: 2

.. image:: https://travis-ci.org/Geosyntec/wqio.svg?branch=master
    :target: https://travis-ci.org/Geosyntec/wqio


.. image:: https://codecov.io/github/Geosyntec/wqio/coverage.svg?precision=1
    :target: https://codecov.io/gh/Geosyntec/wqio

.. image:: https://anaconda.org/phobson/wqio/badges/version.svg
    :target: https://anaconda.org/phobson/wqio

.. image:: https://anaconda.org/phobson/wqio/badges/license.svg
    :target: https://anaconda.org/phobson/wqio

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

Periodic builds of ``master`` through conda::

    $ conda install wqio --channel=phobson/label/dev

Running tests
=============

.. toctree::
   :maxdepth: 2

Install pytest and plugins::

    $ conda install pytest pytest-cov
    $ conda install pytest-mpl --channel=conda-forge

Run the tests with the test runner::

    $ python check_wqio.py --mpl --cov --pep8

Run the tests from within python::

    $ python
    >>> import matploltib as mpl; mpl.use('agg')
    >>> import wqio
    >>> wqio.test('--mpl', '--cov', '--pep8')
