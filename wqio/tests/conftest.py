import pytest


@pytest.fixture(scope='function', autouse=True)
def bug_workaround():
    print('setting up')
    pass
