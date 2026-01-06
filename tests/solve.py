from mpir_sparse import solve
import numpy as np
import scipy.sparse as sp

A = sp.eye(100, format="csr")
b = np.ones(100)

x, info = solve(
    A, b,
    preconditioner="jacobi",
    inner_solver="gmres",
    restart=50,
    scaling="diag",
)
print(info["converged"], info["final_backward_error"])
