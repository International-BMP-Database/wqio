import difflib
import os
import re
import subprocess
import sys
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from importlib import resources
from io import StringIO

import numpy
import pandas
import pytest
from packaging.version import Version


def get_img_tolerance():
    return int(os.environ.get("MPL_IMGCOMP_TOLERANCE", 15))


def seed(func):
    """Decorator to seed the RNG before any function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)

    return wrapper


def raises(error):
    """Wrapper around pytest.raises to support None."""
    if error:
        return pytest.raises(error)
    else:

        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e

        return not_raises()


def requires(module, modulename):
    def outer_wrapper(function):
        @wraps(function)
        def inner_wrapper(*args, **kwargs):
            if module is None:
                raise RuntimeError(f"{modulename} required for `{function.__name__}`")
            else:
                return function(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


@seed
def make_dc_data(ndval="ND", rescol="res", qualcol="qual"):
    dl_map = {
        "A": 0.1,
        "B": 0.2,
        "C": 0.3,
        "D": 0.4,
        "E": 0.1,
        "F": 0.2,
        "G": 0.3,
        "H": 0.4,
    }

    index = pandas.MultiIndex.from_product(
        [
            list("ABCDEFGH"),
            list("1234567"),
            ["GA", "AL", "OR", "CA"],
            ["Inflow", "Outflow", "Reference"],
        ],
        names=["param", "bmp", "state", "loc"],
    )

    array = numpy.random.lognormal(mean=0.75, sigma=1.25, size=len(index))
    data = pandas.DataFrame(data=array, index=index, columns=[rescol])
    data["DL"] = data.apply(lambda r: dl_map.get(r.name[0]), axis=1)

    data[rescol] = data.apply(
        lambda r: dl_map.get(r.name[0]) if r[rescol] < r["DL"] else r[rescol], axis=1
    )

    data[qualcol] = data.apply(lambda r: ndval if r[rescol] <= r["DL"] else "=", axis=1)

    return data


@seed
def make_dc_data_complex(dropsome=True):
    dl_map = {
        "A": 0.25,
        "B": 0.50,
        "C": 0.10,
        "D": 1.00,
        "E": 0.25,
        "F": 0.50,
        "G": 0.10,
        "H": 1.00,
    }

    index = pandas.MultiIndex.from_product(
        [
            list("ABCDEFGH"),
            list("1234567"),
            ["GA", "AL", "OR", "CA"],
            ["Inflow", "Outflow", "Reference"],
        ],
        names=["param", "bmp", "state", "loc"],
    )

    xtab = (
        pandas.DataFrame(index=index, columns=["res"]).unstack(level="param").unstack(level="state")
    )

    xtab_rows = xtab.shape[0]
    for c in xtab.columns:
        mu = numpy.random.uniform(low=-1.7, high=2)
        sigma = numpy.random.uniform(low=0.1, high=2)
        xtab[c] = numpy.random.lognormal(mean=mu, sigma=sigma, size=xtab_rows)

    data = xtab.stack(level="state").stack(level="param")

    data["DL"] = data.apply(lambda r: dl_map.get(r.name[-1]), axis=1)

    data["res"] = data.apply(
        lambda r: dl_map.get(r.name[-1]) if r["res"] < r["DL"] else r["res"], axis=1
    )

    data["qual"] = data.apply(lambda r: "<" if r["res"] <= r["DL"] else "=", axis=1)

    if dropsome:
        if int(dropsome) == 1:
            dropsome = 0.25
        index = numpy.random.uniform(size=data.shape[0]) >= dropsome
        data = data.loc[index]

    return data


def comp_statfxn(x, y):
    stat = namedtuple("teststat", ("statistic", "pvalue"))
    result = x.max() - y.min()
    return stat(result, result * 0.25)


def group_statfxn(*x):
    stat = namedtuple("teststat", ("statistic", "pvalue"))
    result = max([max(_) for _ in x])
    return stat(result, 0.25)


def group_statfxn_with_control(*x, control):
    stat = namedtuple("teststat", ("statistic", "pvalue"))
    x = [*x, control]
    result = max([max(_) for _ in x])
    return stat(result, 0.25)


def test_data_path(filename):
    path = resources.files("wqio.tests._data").joinpath(filename)
    return path


def getTestROSData():
    """
    Generates test data for an ROS estimate.
    Input:
        None

    Output:
        Structured array with the values (results or DLs) and qualifers
        (blank or "ND" for non-detects)
    """
    raw_csv = StringIO(
        "res,qual\n2.00,=\n4.20,=\n4.62,=\n5.00,ND\n5.00,ND\n5.50,ND\n"
        "5.57,=\n5.66,=\n5.75,ND\n5.86,=\n6.65,=\n6.78,=\n6.79,=\n7.50,=\n"
        "7.50,=\n7.50,=\n8.63,=\n8.71,=\n8.99,=\n9.50,ND\n9.50,ND\n9.85,=\n"
        "10.82,=\n11.00,ND\n11.25,=\n11.25,=\n12.20,=\n14.92,=\n16.77,=\n"
        "17.81,=\n19.16,=\n19.19,=\n19.64,=\n20.18,=\n22.97,=\n"
    )

    return pandas.read_csv(raw_csv)


def compare_versions(utility="latex"):  # pragma: no cover
    "return True if a is greater than or equal to b"
    requirements = {"latex": "3.1415"}

    available = {"latex": checkdep_tex()}

    required = requirements[utility]
    present = available[utility]
    if present:
        present = Version(present)
        required = Version(required)
        return present >= required

    else:
        return False


def _show_package_info(package, name):  # pragma: no cover
    packagedir = os.path.dirname(package.__file__)
    print(f"{name} version {package.__version__} is installed in {packagedir}")


def _show_system_info():  # pragma: no cover
    import pytest

    pyversion = sys.version.replace("\n", "")
    print(f"Python version {pyversion}")
    print(f"pytest version {pytest.__versioninfo__}")

    import numpy

    _show_package_info(numpy, "numpy")

    import scipy

    _show_package_info(scipy, "scipy")

    import matplotlib

    _show_package_info(matplotlib, "matplotlib")

    import statsmodels

    _show_package_info(statsmodels, "statsmodels")

    import pandas

    _show_package_info(pandas, "pandas")


def checkdep_tex():  # pragma: no cover
    def byte2str(b):
        return b.decode("ascii")

    try:
        s = subprocess.Popen(["tex", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        line = byte2str(s.stdout.readlines()[0])
        pattern = r"3\.1\d+"
        match = re.search(pattern, line)
        v = match.group(0)
        return v
    except (IndexError, ValueError, AttributeError, OSError):
        return None


def assert_bigstring_equal(
    input_string, known_string, input_out=None, known_out=None
):  # pragma: no cover
    if input_string != known_string:
        if input_out and known_out:
            with open(input_out, "w") as fi:
                fi.write(input_string)

            with open(known_out, "w") as fo:
                fo.write(known_string)

        message = "".join(
            difflib.ndiff(input_string.splitlines(True), known_string.splitlines(True))
        )
        raise AssertionError("Multi-line strings are unequal:\n" + message)
