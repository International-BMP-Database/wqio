"""Helpers for checking the polars implementation against pandas-recorded goldens.

The goldens (``wqio/tests/_data/golden/*.parquet``) are flat frames: index levels
are ordinary columns and a column MultiIndex ``(a, b)`` is flattened to
``"a__b"``. The polars implementation represents the same shapes with struct
columns (one struct per station, with the statistics as fields), so ``flatten``
unnests structs into ``"<column>__<field>"`` columns before comparing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars
from polars.testing import assert_frame_equal

from wqio.tests import helpers

SEP = "__"


def flatten(df: polars.DataFrame, sep: str = SEP) -> polars.DataFrame:
    """Recursively unnest struct columns into ``<column><sep><field>`` columns."""
    out = df
    while any(isinstance(dtype, polars.Struct) for dtype in out.schema.values()):
        exprs = []
        for name, dtype in out.schema.items():
            if isinstance(dtype, polars.Struct):
                exprs.extend(
                    polars.col(name).struct.field(f.name).alias(f"{name}{sep}{f.name}")
                    for f in dtype.fields
                )
            else:
                exprs.append(polars.col(name))
        out = out.select(exprs)
    return out


def swap_levels(df: polars.DataFrame, sep: str = SEP) -> polars.DataFrame:
    """Rename ``"a__b"`` columns to ``"b__a"`` (split on the *last* separator).

    Pandas' paired frame is ``(value, station)``; the polars one is
    ``station -> {value}``. Columns without a separator are left alone.
    """

    def swap(col: str) -> str:
        head, found, tail = col.rpartition(sep)
        return f"{tail}{sep}{head}" if found and head else col

    return df.rename({c: swap(c) for c in df.columns})


def assert_matches_golden(
    result: polars.DataFrame,
    name: str,
    *,
    sort_by: Sequence[str] | None = None,
    rename: dict[str, str] | Callable[[str], str] | None = None,
    swap: bool = False,
    drop: Sequence[str] = (),
    rtol: float = 1e-9,
    atol: float = 1e-9,
    columns: Sequence[str] | None = None,
) -> None:
    """Assert ``result`` equals the recorded golden ``name`` (up to row/column order).

    Parameters
    ----------
    result
        Output of the polars implementation. Struct columns are flattened.
    name
        Golden file stem under ``wqio/tests/_data/golden``.
    sort_by
        Columns used to align rows. Defaults to every non-float column.
    rename
        Mapping (or function) applied to the *golden* column names, e.g. to
        account for a deliberate rename in the polars API.
    swap
        Swap the two levels of the golden column names (see ``swap_levels``).
    drop
        Columns to drop from both frames before comparing.
    columns
        If given, compare only these (flattened) columns.
    """
    expected = helpers.read_golden(name)
    if swap:
        expected = swap_levels(expected)
    if rename is not None:
        expected = expected.rename(rename)

    actual = flatten(result)
    if drop:
        actual = actual.drop(*drop, strict=False)
        expected = expected.drop(*drop, strict=False)
    if columns is not None:
        actual = actual.select(columns)
        expected = expected.select(columns)

    assert sorted(actual.columns) == sorted(expected.columns), (
        f"columns differ for golden {name!r}:\n"
        f"  only in result:   {sorted(set(actual.columns) - set(expected.columns))}\n"
        f"  only in expected: {sorted(set(expected.columns) - set(actual.columns))}"
    )

    keys = (
        list(sort_by)
        if sort_by is not None
        else [c for c, dt in actual.schema.items() if not dt.is_float()]
    )
    actual = actual.select(sorted(actual.columns)).sort(keys, nulls_last=True)
    expected = expected.select(sorted(expected.columns)).sort(keys, nulls_last=True)

    assert_frame_equal(
        actual,
        expected,
        check_dtypes=False,
        check_exact=False,
        rel_tol=rtol,
        abs_tol=atol,
    )
