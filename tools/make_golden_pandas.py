"""Record golden outputs from the *pandas* implementation of wqio.

Run this against the ``pandas-final`` tag (the last commit before the polars
conversion). The resulting parquet files under ``wqio/tests/_data/golden`` are
the parity oracle used while porting to polars. The script is not part of that
tag, so run it from a worktree (and give it the pandas-era ``helpers`` that build
the seeded data with numpy):

    git worktree add ../wqio-pandas-final pandas-final
    cp tools/make_golden_pandas.py ../wqio-pandas-final/tools/
    (cd ../wqio-pandas-final && uv run --frozen python tools/make_golden_pandas.py)
    cp ../wqio-pandas-final/wqio/tests/_data/golden/*.parquet wqio/tests/_data/golden/

Conventions used for every recorded output:

* index levels are moved into ordinary columns (``reset_index``);
* a column MultiIndex such as ``(station, result)`` is flattened to
  ``"<station>__<result>"`` (e.g. ``Inflow__mean``), which is exactly what
  ``wqio.tests.parity.flatten`` produces from the polars struct-per-station
  layout;
* bootstrapped statistics are computed after ``numpy.random.seed(0)`` but are
  only compared with a loose tolerance, because polars-ros uses a different RNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pandas
import polars

from wqio import DataCollection
from wqio.tests import helpers

OUT = Path(__file__).resolve().parents[1] / "wqio" / "tests" / "_data" / "golden"


def flatten(obj: pandas.DataFrame | pandas.Series) -> polars.DataFrame:
    df = obj.to_frame() if isinstance(obj, pandas.Series) else obj.copy()
    if isinstance(df.columns, pandas.MultiIndex):
        df.columns = ["__".join(str(c) for c in col if str(c) != "") for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    return polars.from_pandas(df)


def save(name: str, obj: pandas.DataFrame | pandas.Series) -> None:
    flatten(obj).write_parquet(OUT / f"{name}.parquet")
    print("wrote", name)


def remove_g_and_h(group: pandas.DataFrame) -> bool:
    return group.name[1] not in ["G", "H"]


def make_dc(data: pandas.DataFrame, ndval: str, **kwargs) -> DataCollection:
    return DataCollection(
        data,
        rescol="res",
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval=ndval,
        othergroups=None,
        pairgroups=["state", "bmp"],
        useros=True,
        filterfxn=remove_g_and_h,
        bsiter=10000,
        showpbar=False,
        **kwargs,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # inputs (so the polars test helpers need not re-create them with numpy's RNG)
    save("input_dc_data", helpers.make_dc_data())
    complex_data = helpers.make_dc_data_complex()
    save("input_dc_data_complex", complex_data)
    save("input_dc_data_complex_full", helpers.make_dc_data_complex(dropsome=False))

    dc = make_dc(complex_data, "<")

    save("dc_tidy", dc.tidy.set_index(dc.groupcols, append=False).sort_index())
    save("dc_paired", dc.paired)
    save("dc_count", dc.count)
    save("dc_inventory", dc.inventory)
    save("dc_n_unique_bmp", dc.n_unique("bmp"))
    numpy.random.seed(0)
    save("dc_stat_summary", dc.stat_summary())

    # deterministic statistics
    save("dc_std_dev", dc.std_dev)
    save("dc_logstd_dev", dc.logstd_dev)
    save("dc_geostd_dev", dc.geostd_dev)
    save("dc_percentile_25", dc.percentile(25))
    save("dc_shapiro", dc.shapiro())
    save("dc_shapiro_log", dc.shapiro_log())
    save("dc_lilliefors", dc.lilliefors())
    save("dc_lilliefors_log", dc.lilliefors_log())

    # bootstrapped statistics (loose tolerance on lower/upper)
    for name in ["median", "mean", "logmean", "geomean"]:
        numpy.random.seed(0)
        save(f"dc_{name}", getattr(dc, name))

    # comparisons
    for name in ["mann_whitney", "ranksums", "t_test", "levene", "wilcoxon", "kendall", "spearman"]:
        save(f"dc_{name}", getattr(dc, name)())
    save("dc_kruskal_wallis", dc.kruskal_wallis())
    save("dc_f_test", dc.f_test())

    hsd, scores = dc.tukey_hsd()
    save("dc_tukey_hsd", hsd)
    save("dc_tukey_scores", scores)
    save("dc_dunn", dc.dunn())

    # no non-detects
    dc_nonds = make_dc(complex_data, "junk")
    save("dc_noNDs_inventory", dc_nonds.inventory)
    save("dc_noNDs_tidy", dc_nonds.tidy.set_index(dc_nonds.groupcols).sort_index())


if __name__ == "__main__":
    main()
