from pkg_resources import resource_filename

import wqio
from .helpers import requires

try:
    import pytest
except ImportError:
    pytest = None


@requires(pytest, 'pytest')
def test(*args):
    options = [resource_filename('wqio', '')]
    options.extend(list(args))
    return pytest.main(options)


@requires(pytest, 'pytest')
def teststrict(*args):
    options = [
        resource_filename('wqio', ''),
        '--pep8', '--mpl',
        *list(args)
    ]
    options = list(set(options))
    return pytest.main(options)
