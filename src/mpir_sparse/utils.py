from __future__ import annotations
from typing import Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ArrayLike = Union[np.ndarray, sp.spmatrix, spla.LinearOperator]

def to_linear_operator(A: ArrayLike, dtype) -> spla.LinearOperator:
    m, n = A.shape
    if isinstance(A, spla.LinearOperator):
        def mv(x):
            y = A.matvec(x.astype(dtype, copy=False))
            return np.asarray(y, dtype=dtype)
        def rmv(x):
            y = A.rmatvec(x.astype(dtype, copy=False))
            return np.asarray(y, dtype=dtype)
        return spla.LinearOperator((m, n), matvec=mv, rmatvec=rmv, dtype=dtype)
    elif sp.issparse(A):
        A_cast = A.asformat('csr').astype(dtype, copy=False)
        def mv(x):
            return (A_cast @ x.astype(dtype, copy=False)).astype(dtype, copy=False)
        def rmv(x):
            return (A_cast.T @ x.astype(dtype, copy=False)).astype(dtype, copy=False)
        return spla.LinearOperator((m, n), matvec=mv, rmatvec=rmv, dtype=dtype)
    else:
        A_arr = np.array(A, dtype=dtype, copy=False)
        def mv(x):
            return A_arr @ x.astype(dtype, copy=False)
        def rmv(x):
            return A_arr.T @ x.astype(dtype, copy=False)
        return spla.LinearOperator((m, n), matvec=mv, rmatvec=rmv, dtype=dtype)

def unit_roundoff(dtype) -> float:
    return np.finfo(dtype).eps / 2.0

def power_norm2(A: ArrayLike, iters: int = 20, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = A.shape[1]
    x = rng.standard_normal(n)
    x = x / np.linalg.norm(x)
    for _ in range(iters):
        Ax = A @ x
        nAx = np.linalg.norm(Ax)
        if nAx == 0.0:
            return 0.0
        x = (A.T @ (Ax / nAx))
        nx = np.linalg.norm(x)
        if nx == 0.0:
            break
        x /= nx
    Ax = A @ x
    return np.linalg.norm(Ax)

def backward_error_ok(A_norm, x, b, r, tol) -> bool:
    lhs = np.linalg.norm(r)
    rhs = tol * (A_norm * np.linalg.norm(x) + np.linalg.norm(b))
    return lhs <= rhs


def diagonal_scaling(A, b, min_abs_diag: float = 1e-30):
    """Symmetric diagonal scaling using the diagonal of A.

    Constructs D = diag( 1/sqrt(|diag(A)|) ) and returns:
        A_s = D @ A @ D
        b_s = D @ b
    Such that if A_s y = b_s then x = D @ y solves A x = b.

    Notes
    -----
    - Only supports explicit (dense or sparse) matrices, not LinearOperator.
    - Falls back to no scaling if the diagonal is unusable.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if isinstance(A, spla.LinearOperator):
        return A, b, (lambda y: y), {"applied": False, "reason": "LinearOperator"}
    if sp.issparse(A):
        d = np.asarray(A.diagonal())
    else:
        d = np.asarray(np.diag(np.asarray(A)))

    absd = np.abs(d)
    if absd.size == 0 or not np.all(np.isfinite(absd)):
        return A, b, (lambda y: y), {"applied": False, "reason": "bad_diag"}
    absd = np.maximum(absd, min_abs_diag)
    s = (1.0 / np.sqrt(absd)).astype(np.float64)

    if sp.issparse(A):
        D = sp.diags(s, 0, format="csr")
        A_s = (D @ A) @ D
    else:
        D = np.diag(s)
        A_s = D @ np.asarray(A) @ D

    b_s = s * b

    def unscale(y):
        return (s * y)

    return A_s, b_s, unscale, {"applied": True, "min_abs_diag": float(min_abs_diag)}
