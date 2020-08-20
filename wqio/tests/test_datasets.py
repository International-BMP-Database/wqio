import tempfile

import pytest

from wqio import datasets


@pytest.mark.parametrize("fname", ["bmpdata", "nsqd"])
def test_download(fname):
    with tempfile.TemporaryDirectory() as tmpdir:
        dstpath = datasets.download(fname, data_dir=tmpdir, year=2016)
        assert dstpath.exists()
        assert dstpath.name == fname + ".csv"
