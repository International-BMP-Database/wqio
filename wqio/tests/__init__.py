from pkg_resources import resource_filename

import wqio


def test(alltests, *args):
    try:
        import pytest
    except ImportError:
        raise ImportError("pytest is requires to run tests")

    if alltests:
        options = [resource_filename('wqio', '')]
    else:
        options = []

    options.extend(list(args))
    return pytest.main(options)
