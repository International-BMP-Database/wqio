import os
import tempfile
import shutil

import pytest
from pandas.util.testing import network

from wqio import datasets


@network(url='https://github.com/Geosyntec/water-quality-dataset')
@pytest.mark.parametrize('fname', ['bmpdata', 'nsqd'])
@pytest.mark.parametrize('use_tmpdir', [True, False])
def test_download(fname, use_tmpdir):
    if use_tmpdir:
        tmpdir = tempfile.mkdtemp()
    else:
        tmpdir = None

    fname = datasets.download(fname, data_dir=tmpdir)
    assert os.path.exists(fname)
    yield

    path, _file = os.path.split(fname)
    os.remove(fname)
    if tmpdir is not None:
        shutil.rmtree(tmpdir)
