import numpy as np
import scipy.sparse as sp
from mpir_sparse.ir import solve

def test_poisson_2d():
    nx, ny = 40, 40
    N = nx * ny
    main = 4.0 * np.ones(N)
    ew = -1.0 * np.ones(N - 1); ew[np.arange(1, N) % nx == 0] = 0.0
    ns = -1.0 * np.ones(N - nx)
    A = sp.diags([main, ew, ew, ns, ns], [0, -1, 1, -nx, nx], format="csr", dtype=np.float64)
    x_true = np.ones(N)
    b = A @ x_true
    x, info = solve(A, b, tol=1e-12, k_max=8, inner_tol=1e-3, work_dtype=np.float32, preconditioner="ilu")
    relerr = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    assert info["converged"]
    assert info["final_backward_error"] <= 1e-9
    assert relerr <= 1e-7
