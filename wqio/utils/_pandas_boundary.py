"""Conversions between pandas and polars.

wqio works on polars dataframes. A few libraries it relies on (seaborn,
statsmodels, probscale) still require pandas, so this module is the *single*
place where pandas objects are created or consumed:

* ``from_pandas`` is the adapter for users who still have pandas data;
* ``to_pandas`` is used at the plotting/statsmodels call sites.

pandas is imported lazily so that ``import wqio`` does not require it for
anything except the (not yet ported) time-series modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import polars

if TYPE_CHECKING:  # pragma: no cover
    import pandas

__all__ = ["from_pandas", "to_pandas"]


def from_pandas(df: pandas.DataFrame, include_index: bool = True) -> polars.DataFrame:
    """Convert a pandas dataframe to a polars dataframe that wqio can consume.

    Parameters
    ----------
    df : pandas.DataFrame
        The data. Any (Multi)Index is moved into ordinary columns when
        ``include_index`` is True, since polars has no row index.
    include_index : bool, optional
        Whether the index levels become columns (default) or are dropped.

    Returns
    -------
    polars.DataFrame
        NaN values are converted to nulls.

    Example
    -------
    >>> import pandas
    >>> from wqio.utils import from_pandas
    >>> index = pandas.MultiIndex.from_product([["A", "B"], [1, 2]], names=["param", "bmp"])
    >>> pdf = pandas.DataFrame({"res": [1.0, 2.0, float("nan"), 4.0]}, index=index)
    >>> from_pandas(pdf).columns
    ['param', 'bmp', 'res']
    >>> from_pandas(pdf)["res"].null_count()
    1

    """
    df = df.reset_index() if include_index else df.reset_index(drop=True)
    return polars.from_pandas(df, nan_to_null=True)


def to_pandas(
    df: Any,
    categories: Mapping[str, Sequence[Any]] | None = None,
    **kwargs: Any,
) -> pandas.DataFrame:
    """Convert a polars dataframe to pandas for libraries that require it.

    The pandas frame is built from numpy arrays, which avoids requiring
    ``pyarrow``. Objects that are not polars dataframes (e.g., a pandas
    dataframe) are returned unchanged.

    Parameters
    ----------
    df : polars.DataFrame
    categories : mapping of column name to sequence, optional
        Columns that should become *ordered* pandas categoricals with the
        given categories (and order).
    **kwargs
        Passed to ``pandas.DataFrame``.
    """
    if not isinstance(df, polars.DataFrame):
        return df

    import pandas

    data = {name: df[name].to_numpy() for name in df.columns}
    frame = pandas.DataFrame(data, **kwargs)
    for column, cats in (categories or {}).items():
        frame[column] = pandas.Categorical(frame[column], categories=list(cats), ordered=True)
    return frame
