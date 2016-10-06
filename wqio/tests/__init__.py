from pkg_resources import resource_filename

import pytest

import wqio


def test(*args):
    options = [resource_filename('wqio', 'tests')]
    options.extend(list(args))
    return pytest.main(options)
