from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Optional dependency
try:
    import pyamg  # type: ignore
    _HAS_PYAMG = True
except Exception:
    pyamg = None  # type: ignore
    _HAS_PYAMG = False


@dataclass
class PreconditionerInfo:
    kind: str
    params: Dict[str, Any]


def _ilu_preconditioner(A: sp.spmatrix, drop_tol: float = 1e-4, fill_factor: float = 10.0) -> spla.LinearOperator:
    """Build a right-preconditioner M ≈ A^{-1} via ILU (SciPy spilu)."""
    Acsc = A.asformat("csc")
    ilu = spla.spilu(Acsc, drop_tol=drop_tol, fill_factor=fill_factor)

    def mv(x: np.ndarray) -> np.ndarray:
        return ilu.solve(x)

    return spla.LinearOperator(A.shape, matvec=mv, dtype=A.dtype)


def _jacobi_preconditioner(A: sp.spmatrix, damping: float = 1.0, min_abs_diag: float = 1e-30) -> spla.LinearOperator:
    """Damped Jacobi right-preconditioner: M = damping * diag(A)^{-1}."""
    d = np.asarray(A.diagonal())
    # Avoid divide by zero / tiny diagonals
    safe = np.where(np.abs(d) >= min_abs_diag, d, np.sign(d) * min_abs_diag + (d == 0) * min_abs_diag)
    invd = (damping / safe).astype(A.dtype, copy=False)

    def mv(x: np.ndarray) -> np.ndarray:
        return invd * x

    return spla.LinearOperator(A.shape, matvec=mv, dtype=A.dtype)


def _amg_preconditioner(A: sp.spmatrix, strength: str = "symmetric", cycle: str = "V", max_levels: Optional[int] = None) -> spla.LinearOperator:
    """Build a right-preconditioner from PyAMG's smoothed aggregation solver."""
    if not _HAS_PYAMG:
        raise RuntimeError("pyamg not available. Install pyamg to use AMG preconditioner.")
    assert pyamg is not None
    kwargs: Dict[str, Any] = {"symmetry": "symmetric", "strength": strength}
    if max_levels is not None:
        kwargs["max_levels"] = max_levels
    ml = pyamg.smoothed_aggregation_solver(A, **kwargs)
    return ml.aspreconditioner(cycle=cycle)


def make_preconditioner_with_info(A, kind: Optional[str] = None, **kwargs) -> Tuple[Optional[spla.LinearOperator], Optional[PreconditionerInfo]]:
    """Create a (right) preconditioner LinearOperator plus metadata.

    Parameters
    ----------
    A:
        Matrix (preferably sparse). If A is a LinearOperator, only kind=None is supported.
    kind:
        None, 'ilu', 'amg', or 'jacobi'.
    kwargs:
        Extra parameters forwarded to the chosen preconditioner builder.

    Returns
    -------
    M, info:
        M is a LinearOperator (or None). info is a PreconditionerInfo (or None).
    """
    if kind is None:
        return None, None
    if isinstance(A, spla.LinearOperator):
        raise ValueError(f"Cannot build preconditioner kind='{kind}' from a LinearOperator; provide an explicit sparse matrix.")
    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    k = kind.lower()
    if k == "ilu":
        drop_tol = float(kwargs.pop("drop_tol", 1e-4))
        fill_factor = float(kwargs.pop("fill_factor", 10.0))
        M = _ilu_preconditioner(A, drop_tol=drop_tol, fill_factor=fill_factor)
        return M, PreconditionerInfo("ilu", {"drop_tol": drop_tol, "fill_factor": fill_factor})
    if k == "amg":
        strength = str(kwargs.pop("strength", "symmetric"))
        cycle = str(kwargs.pop("cycle", "V"))
        max_levels = kwargs.pop("max_levels", None)
        M = _amg_preconditioner(A, strength=strength, cycle=cycle, max_levels=max_levels)
        return M, PreconditionerInfo("amg", {"strength": strength, "cycle": cycle, "max_levels": max_levels})
    if k == "jacobi":
        damping = float(kwargs.pop("damping", 1.0))
        min_abs_diag = float(kwargs.pop("min_abs_diag", 1e-30))
        M = _jacobi_preconditioner(A, damping=damping, min_abs_diag=min_abs_diag)
        return M, PreconditionerInfo("jacobi", {"damping": damping, "min_abs_diag": min_abs_diag})

    raise ValueError(f"Unknown preconditioner kind: {kind}")


def make_preconditioner(A, kind: Optional[str] = None, **kwargs) -> Optional[spla.LinearOperator]:
    """Backwards-compatible factory that returns only the operator (or None)."""
    M, _ = make_preconditioner_with_info(A, kind=kind, **kwargs)
    return M
