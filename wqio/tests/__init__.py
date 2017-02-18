from pkg_resources import resource_filename

import wqio


def test(*args, **kwargs):
    try:
        import pytest
    except ImportError:
        raise ImportError("pytest is requires to run tests")

    alltests = kwargs.pop('alltests', True)
    if alltests:
        options = [resource_filename('wqio', '')]
    else:
        options = []

    options.extend(list(args))
    return pytest.main(options)
