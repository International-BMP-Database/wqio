from warnings import simplefilter
import distutils
import sys
import subprocess
import re
import os

from functools import wraps
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest

from numpy import errstate
import numpy.testing as nptest

from numpy.testing.noseclasses import NumpyTestProgram


def assert_timestamp_equal(x, y):
    nptest.assert_equal(x.strftime('%x %X'), y.strftime('%x %X'))

def setup_prefix(folder):
    for imgdir in ['baseline_images', 'results_images']:
        subdir = os.path.join('.', imgdir)
        subsubdir = os.path.join(subdir, folder)

        if not os.path.exists(subdir):
            os.mkdir(subdir)

        if not os.path.exists(subsubdir):
            os.mkdir(subsubdir)


def fail(message):
    raise AssertionError(message)


def wip(f):
    @wraps(f)
    def run_test(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            raise SkipTest("WIP test failed: " + str(e))
        fail("test passed but marked as work in progress")

    return attr('wip')(run_test)


def compare_versions(utility='latex'):
    "return True if a is greater than or equal to b"
    requirements = {
    	'latex': '3.1415'
    }

    available = {
    	'latex': checkdep_tex()
    }

    required = requirements[utility]
    present = available[utility]
    if present:
        present = distutils.version.LooseVersion(present)
        required = distutils.version.LooseVersion(required)
        if present >= required:
        	return True
        else:
        	return False
    else:
    	return False


def _show_package_info(package, name):
    packagedir = os.path.dirname(package.__file__)
    print("%s version %s is installed in %s" % (name, package.__version__, packagedir))

def _show_system_info():
    import nose

    pyversion = sys.version.replace('\n','')
    print("Python version %s" % pyversion)
    print("nose version %d.%d.%d" % nose.__versioninfo__)

    import numpy
    _show_package_info(numpy, 'numpy')

    import scipy
    _show_package_info(scipy, 'scipy')

    import matplotlib
    _show_package_info(matplotlib, 'matplotlib')

    import statsmodels
    _show_package_info(statsmodels, 'statsmodels')

    import pandas
    _show_package_info(pandas, 'pandas')

    import openpyxl
    _show_package_info(openpyxl, 'openpyxl')

class NoseWrapper(nptest.Tester):
    '''
    This is simply a monkey patch for numpy.testing.Tester.

    It allows extra_argv to be changed from its default None to ['--exe'] so
    that the tests can be run the same across platforms.  It also takes kwargs
    that are passed to numpy.errstate to suppress floating point warnings.
    '''


    def test(self, label='fast', verbose=1, extra_argv=['--exe', '--with-id'],
             doctests=False, coverage=False, packageinfo=True, **kwargs):
        '''
        Run tests for module using nose

        %(test_header)s
        doctests : boolean
            If True, run doctests in module, default False
        coverage : boolean
            If True, report coverage of NumPy code, default False
            (Requires the coverage module:
             http://nedbatchelder.com/code/modules/coverage.html)
        kwargs
            Passed to numpy.errstate.  See its documentation for details.
        '''

        # cap verbosity at 3 because nose becomes *very* verbose beyond that
        verbose = min(verbose, 3)
        nptest.utils.verbose = verbose

        if packageinfo:
            _show_system_info()

        if doctests:
            print("\nRunning unit tests and doctests for %s" % self.package_name)
        else:
            print("\nRunning unit tests for %s" % self.package_name)

        # reset doctest state on every run
        import doctest
        doctest.master = None

        argv, plugins = self.prepare_test_args(label, verbose, extra_argv,
                                               doctests, coverage)

        with errstate(**kwargs):
##            with catch_warnings():
            simplefilter('ignore', category=DeprecationWarning)
            t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
        return t.result


if sys.version_info[0] >= 3:
    def ascii(s): return bytes(s, 'ascii')

    def byte2str(b): return b.decode('ascii')

else:
    ascii = str

    def byte2str(b): return b


def checkdep_tex():
    try:
        s = subprocess.Popen(['tex','-version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        line = byte2str(s.stdout.readlines()[0])
        pattern = '3\.1\d+'
        match = re.search(pattern, line)
        v = match.group(0)
        return v
    except (IndexError, ValueError, AttributeError, OSError):
        return None

