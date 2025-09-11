#!/usr/bin/env python
from __future__ import annotations
import argparse, os, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import ssgetpy
from mpir_sparse.ir import solve

def load_suitesparse(limit=3, group=None, kind='real', min_n=5000, max_n=200000):
    cols = ssgetpy.search(nrows='>= %d' % min_n, nrows_le=max_n, ncols='square',
                          isBinary=False, isReal=(kind=='real'), group=group,
                          limit=limit, sort='sprank')
    mats = []
    for i, meta in enumerate(cols):
        print(f"Downloading {i+1}/{len(cols)}: {meta.group}/{meta.name} (n={meta.nrows}, nnz={meta.nnz})")
        path = meta.download(extract=True)
        A = sp.load_npz(os.path.join(path, f"{meta.name}.npz"))
        mats.append((meta.group, meta.name, A))
    return mats

def run_experiment(limit=3, precond=None, plot=False):
    results = []
    mats = load_suitesparse(limit=limit)
    for (grp, name, A) in mats:
        n = A.shape[0]
        x_true = np.ones(n)
        b = A @ x_true

        precond_kwargs = {}
        if precond == 'ilu':
            precond_kwargs = dict(drop_tol=1e-4, fill_factor=10.0)
        elif precond == 'amg':
            precond_kwargs = {}

        print(f"Running IR on {name} with preconditioner={precond}")
        t0 = time.time()
        x, info = solve(A, b, tol=1e-12, k_max=10, inner_tol=1e-3, work_dtype=np.float32,
                        preconditioner=precond, precond_kwargs=precond_kwargs)
        t1 = time.time()

        res = dict(
            matrix=f"{grp}/{name}",
            n=int(n),
            nnz=int(A.nnz),
            ir_time=t1-t0,
            ir_converged=bool(info['converged']),
            ir_iters=int(info['iters']),
            ir_final_be=float(info['final_backward_error']),
            ir_escalations=info['escalations'],
        )
        results.append(res)
        print(json.dumps(res, indent=2))

    if plot:
        labels = [r['matrix'] for r in results]
        times = [r['ir_time'] for r in results]
        plt.figure()
        plt.bar(range(len(times)), times)
        plt.xticks(range(len(times)), labels, rotation=45, ha='right')
        plt.ylabel("IR wall time (s)")
        plt.title("Mixed-Precision IR runtime on SuiteSparse")
        plt.tight_layout()
        plt.savefig("bench_ir_times.png", dpi=150)
        print("Saved plot: bench_ir_times.png")

    with open("results_ir.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results_ir.json")
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="number of matrices to fetch")
    ap.add_argument("--precond", type=str, default=None, choices=[None, "ilu", "amg"])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run_experiment(limit=args.limit, precond=args.precond, plot=args.plot)
