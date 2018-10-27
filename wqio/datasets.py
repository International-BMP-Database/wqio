import os
from zipfile import ZipFile
from urllib import request
from pathlib import Path

from wqio import validate


def download(dataset, year=None, redownload=True, data_dir=None):
    fname = validate.dataset(dataset)

    if year is None:
        tag = 'master'
    else:
        tag = 'v{:d}'.format(year)

    url_template = 'https://github.com/Geosyntec/water-quality-datasets/blob/{tag:s}/data/{fname:s}?raw=true'
    src_url = url_template.format(tag=tag, fname=fname)

    if data_dir is None:
        base_dir = Path(os.environ.get('WQ_DATA', '~/.wq-data'))
        data_dir = base_dir.expanduser().absolute() / tag
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(exist_ok=True, parents=True)
    dst_path = data_dir / fname
    if not dst_path.exists() or redownload:
        request.urlretrieve(src_url, dst_path)

        with ZipFile(dst_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    return dst_path.parent / "{}.csv".format(dst_path.stem)
