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
        base_dir = os.environ.get('WQ_DATA', os.path.join('~', '.wq-data'))
        data_dir = os.path.join(os.path.expanduser(base_dir), tag)
        os.makedirs(data_dir, exist_ok=True)

    dst_path = os.path.join(data_dir, fname)
    if not os.path.exists(dst_path) or redownload:
        request.urlretrieve(src_url, dst_path)

        with ZipFile(dst_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    return Path(os.path.splitext(dst_path)[0] + '.csv')
