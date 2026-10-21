"""Time the main ``DataCollection`` entry points on a large synthetic data set.

Run with the implementation under test installed:

    uv run --frozen python tools/benchmark.py --backend polars
    # in a worktree of the ``pandas-final`` tag:
    uv run --frozen python tools/benchmark.py --backend pandas

Both backends use the same seeded synthetic data (long format: param, state,
bmp, loc, res, qual) so the timings are comparable.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy
import polars

from wqio import DataCollection


def make_data(n_bmps: int, seed: int = 0) -> polars.DataFrame:
    rng = numpy.random.default_rng(seed)
    params = list("ABCDEFGH")
    states = ["GA", "AL", "OR", "CA"]
    locs = ["Inflow", "Outflow", "Reference"]
    bmps = [f"bmp{i:05d}" for i in range(n_bmps)]
    grid = polars.DataFrame({"param": params}).join(
        polars.DataFrame({"state": states}).join(
            polars.DataFrame({"bmp": bmps}).join(polars.DataFrame({"loc": locs}), how="cross"),
            how="cross",
        ),
        how="cross",
    )
    n = grid.height
    dl = grid["param"].replace_strict({p: 0.1 * (i + 1) for i, p in enumerate(params)})
    res = polars.Series(rng.lognormal(0.75, 1.25, size=n))
    nd = res < dl
    return grid.with_columns(
        res=polars.when(nd).then(dl).otherwise(res),
        qual=polars.when(nd).then(polars.lit("<")).otherwise(polars.lit("=")),
    ).filter(polars.Series(rng.uniform(size=n) >= 0.25))


def timed(label: str, fxn: Callable[[], object]) -> None:
    start = time.perf_counter()
    fxn()
    print(f"{label:<28s}{time.perf_counter() - start:8.2f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pandas", "polars"], required=True)
    parser.add_argument("--bmps", type=int, default=2000, help="rows ~= 8*4*3*bmps*0.75")
    parser.add_argument(
        "--othergroups",
        action="store_true",
        help="also group by state and bmp: many small groups, which stresses per-group overhead",
    )
    args = parser.parse_args()

    data = make_data(args.bmps)
    print(f"{data.height:,} rows, backend={args.backend}")

    if args.backend == "pandas":
        data = data.to_pandas()

    dc = DataCollection(
        data,
        rescol="res",
        qualcol="qual",
        stationcol="loc",
        paramcol="param",
        ndval="<",
        othergroups=["state", "bmp"] if args.othergroups else None,
        pairgroups=["state", "bmp"],
        useros=True,
        bsiter=500,
        showpbar=False,
    )
    timed("tidy (ROS)", lambda: dc.tidy)
    timed("paired", lambda: dc.paired)
    timed("count", lambda: dc.count)
    timed("inventory", lambda: dc.inventory)
    timed("mean (bootstrap, 500 it)", lambda: dc.mean)
    timed("std_dev", lambda: dc.std_dev)
    timed("wilcoxon (paired)", lambda: dc.wilcoxon())
    timed("mann_whitney", lambda: dc.mann_whitney())
    timed("stat_summary", lambda: dc.stat_summary())
    if args.othergroups:
        timed("t_test", lambda: dc.t_test())
        timed("generic_stat (var)", lambda: dc.generic_stat(numpy.var, use_bootstrap=False))
        timed("dunn", lambda: dc.dunn())


if __name__ == "__main__":
    main()
