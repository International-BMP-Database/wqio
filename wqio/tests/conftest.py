import pytest
from matplotlib import pyplot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    pyplot.close("all")
