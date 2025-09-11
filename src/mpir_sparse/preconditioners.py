\
from __future__ import annotations
from typing import Optional
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import pyamg
    _HAS_PYAMG = True
except Exception:
    _HAS_PYAMG = False

def _ilu_preconditioner(A: sp.spmatrix, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    """Build a right-preconditioner M ≈ A^{-1} via ILU (SciPy spilu)."""
    A = A.asformat('csc')
    ilu = spla.spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
    def mv(x):
        return ilu.solve(x)
    return spla.LinearOperator(A.shape, matvec=mv, dtype=A.dtype)

def _amg_preconditioner(A: sp.spmatrix, strength: str = "symmetric"):
    """Build a right-preconditioner from PyAMG's smoothed aggregation solver."""
    if not _HAS_PYAMG:
        raise RuntimeError("pyamg not available. Install pyamg to use AMG preconditioner.")
    ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", strength=strength)
    return ml.aspreconditioner(cycle='V')

def make_preconditioner(A, kind: Optional[str] = None, **kwargs):
    """
    Create a preconditioner LinearOperator.

    Parameters
    ----------
    A : sparse matrix or array-like
        System matrix.
    kind : str or None
        One of: None, 'ilu', 'amg'.
    **kwargs : dict
        Extra options for the chosen preconditioner.

    Returns
    -------
    M : LinearOperator or None
        Right-preconditioner to pass as `M` into GMRES.
    """
    if kind is None:
        return None
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    if kind == 'ilu':
        return _ilu_preconditioner(A, **kwargs)
    elif kind == 'amg':
        return _amg_preconditioner(A, **kwargs)
    else:
        raise ValueError(f"Unknown preconditioner kind: {kind}")
