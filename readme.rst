.. toctree::
   :maxdepth: 2

wqio - Water Quality, Inflow/Outflow
------------------------------------
.. image:: https://travis-ci.org/Geosyntec/wqio.svg?branch=master
    :target: https://travis-ci.org/Geosyntec/wqio


.. image:: https://codecov.io/github/Geosyntec/wqio/coverage.svg?precision=1
    :target: https://codecov.io/gh/Geosyntec/wqio

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

    $ python check_wqio.py --mpl --cov

Or run the tests from within python::

    $ python
    >>> import matploltib as mpl; mpl.use('agg')
    >>> import wqio
    >>> wqio.test('--mpl', '--cov')


Configuring Atom/Sublime Text 3 to run the tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Sublime, got to Tools -> Build System -> New Build System.
Then add the following configuration and save as "wqio.sublime-build"::

    {
        "working_dir": "<path to the git repository>",
        "cmd": "<full path of the python executable> check_wqio.py --verbose <other pytest options>",
    }


In Atom, install the build_ package, create a new file called ".atom-build.yml" in the
top level of the project directory, and add the following contents::

    cmd: "<full path of the python executable>"
    name: "wqio"
    args:
      - check_wqio.py
      - --verbose
      - <other pytest options ...>
    cwd: <path to the git repository>
    sh: false
    keymap: ctrl-b
    atomCommandName: namespace:buildwqio

After this, hitting ctrl+b in either text editor will run the test suite.

.. _build: https://atom.io/packages/build
