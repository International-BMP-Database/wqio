from warnings import simplefilter
import distutils
import sys
import subprocess
import re
import os
from pkg_resources import resource_string

from functools import wraps
import nose.tools as nt
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest

from numpy import errstate
import numpy.testing as nptest
from numpy.testing.noseclasses import NumpyTestProgram
import pandas

from six import StringIO


def getTestROSData():
    '''
    Generates test data for an ROS estimate.
    Input:
        None

    Output:
        Structured array with the values (results or DLs) and qualifers
        (blank or "ND" for non-detects)
    '''
    raw_csv = StringIO(
        "res,qual\n2.00,=\n4.20,=\n4.62,=\n5.00,ND\n5.00,ND\n5.50,ND\n"
        "5.57,=\n5.66,=\n5.75,ND\n5.86,=\n6.65,=\n6.78,=\n6.79,=\n7.50,=\n"
        "7.50,=\n7.50,=\n8.63,=\n8.71,=\n8.99,=\n9.50,ND\n9.50,ND\n9.85,=\n"
        "10.82,=\n11.00,ND\n11.25,=\n11.25,=\n12.20,=\n14.92,=\n16.77,=\n"
        "17.81,=\n19.16,=\n19.19,=\n19.64,=\n20.18,=\n22.97,=\n"
    )

    return pandas.read_csv(raw_csv)


def assert_timestamp_equal(x, y):
    nptest.assert_equal(x.strftime('%x %X'), y.strftime('%x %X'))


def fail(message): # pragma: no cover
    raise AssertionError(message)


def wip(f):  # pragma: no cover
    @wraps(f)
    def run_test(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            raise SkipTest("WIP test failed: " + str(e))
        fail("test passed but marked as work in progress")

    return attr('wip')(run_test)


def compare_versions(utility='latex'):  # pragma: no cover
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


def _show_package_info(package, name):  # pragma: no cover
    packagedir = os.path.dirname(package.__file__)
    print("%s version %s is installed in %s" % (name, package.__version__, packagedir))


def _show_system_info():  # pragma: no cover
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


class NoseWrapper(nptest.Tester):  # pragma: no cover
    '''
    This is simply a monkey patch for numpy.testing.Tester.

    It allows extra_argv to be changed from its default None to ['--exe'] so
    that the tests can be run the same across platforms.  It also takes kwargs
    that are passed to numpy.errstate to suppress floating point warnings.
    '''


    def test(self, label='fast', verbose=1, with_id=True, exe=True,
             doctests=False, coverage=False, packageinfo=True, extra_argv=[],
             **kwargs):
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
        if with_id:
            extra_argv.extend(['--with-id'])

        if exe:
            extra_argv.extend(['--exe'])

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

        # with catch_warnings():
        with errstate(**kwargs):
            simplefilter('ignore', category=DeprecationWarning)
            t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
        return t.result


if sys.version_info[0] >= 3:
    def ascii(s): return bytes(s, 'ascii')

    def byte2str(b): return b.decode('ascii')

else:  # pragma: no cover
    ascii = str

    def byte2str(b): return b


def checkdep_tex():  # pragma: no cover
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


def assert_bigstring_equal(input_string, known_string, input_out, known_out):  # pragma: no cover
    try:
        nt.assert_equal(input_string, known_string)
    except:
        with open(input_out, 'w') as f:
            f.write(input_string)

        with open(known_out, 'w') as f:
            f.write(known_string)

        raise
