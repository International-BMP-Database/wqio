import os
from pathlib import Path
from urllib import request
from zipfile import ZipFile

from wqio import validate


def download(dataset, year=None, redownload=True, data_dir=None):
    fname = validate.dataset(dataset)

    tag = "main" if year is None else f"v{year:d}"

    url_template = (
        "https://github.com/Geosyntec/water-quality-datasets/blob/{tag:s}/data/{fname:s}?raw=true"
    )
    src_url = url_template.format(tag=tag, fname=fname)

    if data_dir is None:
        base_dir = Path(os.environ.get("WQ_DATA", "~/.wq-data"))
        data_dir = base_dir.expanduser().absolute() / tag
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(exist_ok=True, parents=True)
    dst_path = data_dir / fname
    if not dst_path.exists() or redownload:
        request.urlretrieve(src_url, dst_path)

        with ZipFile(dst_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    return dst_path.parent / f"{dst_path.stem}.csv"
