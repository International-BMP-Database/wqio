from warnings import simplefilter
import distutils
import sys
import subprocess
import re
import os
from functools import wraps
from pkg_resources import resource_filename

import numpy
import numpy.testing as nptest
import pandas

from six import StringIO


def seed(func):
    """ Decorator to seed the RNG before any function. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)
    return wrapper


def test_data_path(filename):
    path = resource_filename("wqio.tests._data", filename)
    return path


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
    import pytest

    pyversion = sys.version.replace('\n','')
    print("Python version %s" % pyversion)
    print("pytest version %d.%d.%d" % pytest.__versioninfo__)

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


def checkdep_tex():  # pragma: no cover
    if sys.version_info[0] >= 3:
        def byte2str(b): return b.decode('ascii')

    else:  # pragma: no cover

        def byte2str(b): return b

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
