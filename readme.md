# `wqio`: water qualiity inflow and outflow

[![Publish Python 🐍 distribution 📦 to PyPI](https://github.com/International-BMP-Database/wqio/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/International-BMP-Database/wqio/actions/workflows/python-publish-pypi.yml)
[![ruff 'n' stuff](https://github.com/International-BMP-Database/wqio/actions/workflows/black.yml/badge.svg)](https://github.com/International-BMP-Database/wqio/actions/workflows/black.yml)
[![Run coverage via codecov](https://github.com/International-BMP-Database/wqio/actions/workflows/check-test-coverage.yml/badge.svg)](https://github.com/International-BMP-Database/wqio/actions/workflows/check-test-coverage.yml)
[![Run basic unit tests](https://github.com/International-BMP-Database/wqio/actions/workflows/python-runtests-basic.yml/badge.svg)](https://github.com/International-BMP-Database/wqio/actions/workflows/python-runtests-basic.yml)
[![Run image comparison tests](https://github.com/International-BMP-Database/wqio/actions/workflows/python-runtests-img-comp.yml/badge.svg)](https://github.com/International-BMP-Database/wqio/actions/workflows/python-runtests-img-comp.yml)

Tools to visualize and compare water quality data collected at different monitoring locations.
Includes: bias-corrected and accelerated (BCA) bootstrapping, censored (non-detect) data imputation via regression-on-order statistics (ROS), and paired data analysis.

## Installation

`wqio` is pure python, so installation on any platform should be straight-forward

### Via pip

```shell
python -m pip install wqio
```

Or:

### From source

```shell
git clone git@github.com:International-BMP-Database/wqio.git
cd wqio
pip install -e .
```

## Running tests

### From an installed package

```python
import wqio
wqio.test()
```

### From the source tree

```shell
python check_wqio.py
```

## Releases

There are two Github actions that are run when tags are pushed.
The first builds and uploads to TestPyPI if the tag is in the form `v<major>.<minor>.<micro>/*/test`.

The second builds, uploads to PyPI, then signs the release and creates a Github release with a bunch of different assets if the tag is in the form `v<major>.<minor>.<micro>/release`.

To execute the builda, create a new tag ends with e.g.,  `/test` (i.e., `v0.6.3/test`) and push that.

If that works, create Yet Another tag that ends with `/release` (i.e., `v0.6.3/release`) and push that.

All in all, that workflow looks like this:

```shell
# get latest source
git switch main
git fetch upstream
git merge --ff-only upstream/main
git tag -a "v0.6.4/test"  # add comment in text editor
git push upstream --tags
# watch, wait for CI to sucessfully build on TestPyPI
git tag -a "v0.6.4/release"  # add comment in text editor
# watch, wait for CI to sucessfully build on actual PyPI
```
