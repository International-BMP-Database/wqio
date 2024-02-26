.. toctree::
   :maxdepth: 2

wqio - Water Quality, Inflow/Outflow
------------------------------------
.. image:: https://travis-ci.org/International-BMP-Database/wqio.svg?branch=main
    :target: https://travis-ci.org/International-BMP-Database/wqio

.. image:: https://codecov.io/github/International-BMP-Database/wqio/coverage.svg?precision=1
    :target: https://codecov.io/gh/International-BMP-Database/wqio

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

    $ git clone git@github.com:International-BMP-Database/wqio.git
    $ cd wqio
    $ pip install -e .

Periodic builds of ``main`` through conda::

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


On Releasing new versions
~~~~~~~~~~~~~~~~~~~~~~~~~

There are two Github actions that are run when tags are pushed.
The first builds and uploads to TestPyPI.
The second builds, uploads to PyPI, then signs the release and creates a Github release with a buunch of different assets.

To execute the test build, create a new tag that starts with the string ``test`` (i.e., ``test0.6.3``) and push that.

If that works, create Yet Another tag that starts with ``v`` (i.e., ``v0.6.3``) and push that.
The second tag will trigger the full release.
