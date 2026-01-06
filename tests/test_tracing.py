import json
import numpy as np
import scipy.sparse as sp

from mpir_sparse import solve


def test_jsonl_trace_and_eta(tmp_path):
    A = sp.eye(50, format="csr", dtype=np.float64)
    b = np.ones(50, dtype=np.float64)

    trace_path = tmp_path / "trace.jsonl"
    x, info = solve(
        A, b,
        preconditioner="jacobi",
        inner_solver="gmres",
        restart=20,
        scaling="diag",
        trace_path=str(trace_path),
        trace_every=1,
        adaptive=True,
        precond_refresh=False,
    )

    assert trace_path.exists()
    lines = trace_path.read_text().strip().splitlines()
    assert len(lines) >= 2  # init + at least one iteration/converged/done
    recs = [json.loads(l) for l in lines]
    # Ensure at least one outer_iter record has eta
    outer = [r for r in recs if r.get("event") == "outer_iter"]
    assert len(outer) >= 1
    assert "eta" in outer[0]
    assert info["converged"]
