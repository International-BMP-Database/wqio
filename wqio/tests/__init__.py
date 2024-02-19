import warnings

from pkg_resources import resource_filename

from wqio.tests.helpers import requires

try:
    import pytest
except ImportError:
    pytest = None


@requires(pytest, "pytest")
def test(*args):
    options = [resource_filename("wqio", "")]
    options.extend(list(args))
    return pytest.main(options)


@requires(pytest, "pytest")
def teststrict(*args):
    options = ["--mpl", "--doctest-modules", *list(args)]
    return test(*list(set(options)))


@requires(pytest, "pytest")
def test_nowarnings(*args):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return teststrict(*args)
