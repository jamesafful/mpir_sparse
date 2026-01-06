#!/usr/bin/env python3
"""Run a small grid of MPIR-Sparse baselines on SuiteSparse matrices.

This script is meant to produce a single CSV that can be used for tables/plots.
It runs four configurations:
  1) fp64 GMRES (SciPy) as a non-IR baseline
  2) fixed GMRES-IR (mpir_sparse) with adaptivity OFF
  3) adaptive GMRES-IR (mpir_sparse) with adaptivity ON, refresh OFF
  4) adaptive GMRES-IR (mpir_sparse) with adaptivity ON, refresh ON

Notes:
- Requires ssgetpy for fetching SuiteSparse matrices.
- Matrices are cached by ssgetpy in its default location.
"""

import argparse
import re
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import ssgetpy  # type: ignore
except Exception as e:
    raise SystemExit(
        "This script requires ssgetpy. Install with: pip install -e '.[benchmarks]' "
        "or pip install ssgetpy==1.0rc2"
    ) from e

from mpir_sparse import solve
from mpir_sparse.preconditioners import make_preconditioner_with_info


def load_suitesparse(
    limit: int = 10,
    group: Optional[str] = None,
    kind: str = "real",
    min_n: int = 5000,
    max_n: int = 200000,
) -> List[Tuple[str, str, sp.spmatrix]]:
        # ssgetpy API has changed across versions; some keyword filters may not exist.
    # We call ssgetpy.search with a retry loop that drops unsupported kwargs.
    kwargs = dict(
        limit=limit,
        group=group,
        kind=kind,
        min_nrows=min_n,
        max_nrows=max_n,
        is_square=True,
        sort_by="sprank",
    )
    while True:
        try:
            cols = ssgetpy.search(**kwargs)
            break
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(?P<key>[^']+)'", str(e))
            if not m:
                raise
            bad = m.group("key")
            if bad not in kwargs:
                raise
            kwargs.pop(bad)


    # If size metadata is available, filter by (min_n, max_n) here as a fallback.
    def _size_ok(meta_obj) -> bool:
        nrows = getattr(meta_obj, "nrows", None)
        ncols = getattr(meta_obj, "ncols", None)
        if nrows is None or ncols is None:
            return True
        try:
            nrows = int(nrows); ncols = int(ncols)
        except Exception:
            return True
        n = max(nrows, ncols)
        return (n >= min_n) and (n <= max_n) and (nrows == ncols)

    cols = [m for m in cols if _size_ok(m)]

    mats: List[Tuple[str, str, sp.spmatrix]] = []
    for meta in cols:
        path = meta.download(extract=True)
        A = sp.load_npz(os.path.join(path, f"{meta.name}.npz"))
        mats.append((meta.group, meta.name, A))
    return mats


def gmres_fp64(A: sp.spmatrix, b: np.ndarray, M: Optional[spla.LinearOperator], rtol: float = 1e-12) -> Dict[str, Any]:
    t0 = time.time()
    iters = 0

    def cb(_):
        nonlocal iters
        iters += 1

    try:
        x, info = spla.gmres(A, b, M=M, rtol=rtol, atol=0.0, maxiter=2000, callback=cb, callback_type="pr_norm")
    except TypeError:
        # Older SciPy
        x, info = spla.gmres(A, b, M=M, tol=rtol, maxiter=2000, callback=cb)

    t1 = time.time()
    r = b - A @ x
    A_norm = spla.norm(A) if hasattr(spla, "norm") else float(np.linalg.norm(A @ np.random.randn(A.shape[1])))
    be = float(np.linalg.norm(r) / (A_norm * np.linalg.norm(x) + np.linalg.norm(b))) if (A_norm > 0) else float(np.linalg.norm(r))
    return {
        "converged": bool(info == 0),
        "time": t1 - t0,
        "outer_iters": 0,
        "inner_iters": int(iters),
        "final_backward_error": be,
        "escalations": "[]",
    }


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="Number of matrices to fetch")
    ap.add_argument("--group", type=str, default=None, help="SuiteSparse group filter (optional)")
    ap.add_argument("--precond", type=str, default="ilu", choices=["none", "ilu", "amg", "jacobi"], help="Preconditioner for IR runs")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")
    ap.add_argument("--trace", action="store_true", help="Write per-run traces (JSONL) into outdir/traces/")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    trace_dir = os.path.join(args.outdir, "traces")
    if args.trace:
        os.makedirs(trace_dir, exist_ok=True)

    mats = load_suitesparse(limit=args.limit, group=args.group)

    rows: List[Dict[str, Any]] = []

    for (grp, name, A) in mats:
        n = int(A.shape[0])
        x_true = np.ones(n)
        b = (A @ x_true).astype(np.float64)

        precond_kind = None if args.precond == "none" else args.precond
        precond_kwargs: Dict[str, Any] = {}
        if precond_kind == "ilu":
            precond_kwargs = {"drop_tol": 1e-4, "fill_factor": 10.0}

        # Build a baseline preconditioner for fp64 GMRES (if requested and possible)
        M_fp64 = None
        if precond_kind is not None:
            try:
                M_fp64, _ = make_preconditioner_with_info(A.astype(np.float64), kind=precond_kind, **precond_kwargs)
            except Exception:
                M_fp64 = None

        configs = [
            ("gmres_fp64", dict()),
            ("fixed_ir", dict(adaptive=False, precond_refresh=False)),
            ("adaptive_ir", dict(adaptive=True, precond_refresh=False)),
            ("adaptive_ir_refresh", dict(adaptive=True, precond_refresh=True)),
        ]

        for label, cfg_overrides in configs:
            trace_path = None
            if args.trace and label != "gmres_fp64":
                trace_path = os.path.join(trace_dir, f"{grp}_{name}__{label}.jsonl")

            t0 = time.time()
            if label == "gmres_fp64":
                out = gmres_fp64(A.astype(np.float64), b, M_fp64, rtol=1e-12)
            else:
                x, info = solve(
                    A,
                    b,
                    tol=1e-12,
                    k_max=10,
                    inner_tol=1e-3,
                    work_dtype=np.float32,
                    preconditioner=precond_kind,
                    precond_kwargs=precond_kwargs,
                    inner_solver="gmres",
                    restart=50,
                    scaling="diag",
                    trace_path=trace_path,
                    trace_every=1,
                    **cfg_overrides,
                )
                # Best effort: summarize inner iters from trace notes if present
                inner_hist = info.get("notes", {}).get("inner_iters_history", [])
                inner_total = int(sum(inner_hist)) if isinstance(inner_hist, list) else 0

                out = {
                    "converged": bool(info.get("converged", False)),
                    "time": float(time.time() - t0),
                    "outer_iters": int(info.get("iters", 0)),
                    "inner_iters": inner_total,
                    "final_backward_error": float(info.get("final_backward_error", float("nan"))),
                    "escalations": json.dumps(info.get("escalations", [])),
                }

            row = {
                "matrix": f"{grp}/{name}",
                "n": n,
                "nnz": int(A.nnz),
                "method": label,
                **out,
            }
            rows.append(row)
            print(json.dumps(row, indent=2))

    csv_path = os.path.join(args.outdir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else ["matrix", "method"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {csv_path} ({len(rows)} rows)")

if __name__ == "__main__":
    run()