from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .utils import to_linear_operator, power_norm2, unit_roundoff, backward_error_ok
from .schedulers import AdaptiveScheduler, SchedulerConfig
from .preconditioners import make_preconditioner

ArrayLike = Union[np.ndarray, sp.spmatrix, spla.LinearOperator]

@dataclass
class IRConfig:
    tol: float = 1e-12
    k_max: int = 10
    rtol_inner: float = 1e-3
    maxit_inner: int = 200
    work_dtype: np.dtype = np.float32
    estimate_kappa: bool = True
    kappa_checks: int = 2
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)  # ✅

@dataclass
class IRInfo:
    converged: bool
    iters: int
    residual_norms: List[float]
    used_precisions: List[str]
    escalations: List[str]
    final_backward_error: float
    A_norm_est: float
    notes: Dict[str, Any]

def _gmres(Aop: spla.LinearOperator,
           rhs64: np.ndarray,
           rtol_inner: float,
           maxit_inner: int,
           M: Optional[spla.LinearOperator] = None) -> np.ndarray:
    rhs = rhs64.astype(Aop.dtype, copy=False)
    # Prefer new SciPy API (≥1.11) and fall back to legacy.
    try:
        d, info = spla.gmres(Aop, rhs,
                             rtol=rtol_inner, atol=0.0,
                             maxiter=maxit_inner, M=M)
    except TypeError:
        # Older SciPy expects `tol=` and may ignore `atol`.
        d, info = spla.gmres(Aop, rhs,
                             tol=rtol_inner,
                             maxiter=maxit_inner, M=M)
    if info != 0:
        raise RuntimeError(f"Inner GMRES failed to converge: info={info}")
    return d


# def _gmres(Aop: spla.LinearOperator,
#            rhs64: np.ndarray,
#            rtol_inner: float,
#            maxit_inner: int,
#            M: Optional[spla.LinearOperator] = None) -> np.ndarray:
#     rhs = rhs64.astype(Aop.dtype, copy=False)
#     d, info = spla.gmres(Aop, rhs, tol=rtol_inner, restart=None, maxiter=maxit_inner, M=M)
#     return d.astype(np.float64, copy=False)

def _kappa_proxy(A64, Aop_work, trials: int = 2) -> float:
    Anorm = power_norm2(A64, iters=10)
    inv_norm_est = 0.0
    n = A64.shape[1]
    rng = np.random.default_rng(0)
    for _ in range(trials):
        g = rng.standard_normal(n)
        y = _gmres(Aop_work, g.astype(np.float64), rtol_inner=1e-2, maxit_inner=200)
        Ay = (A64 @ y)
        num = np.linalg.norm(y)
        den = np.linalg.norm(Ay)
        if den > 0:
            inv_norm_est = max(inv_norm_est, num / den)
    return Anorm * inv_norm_est

def iterative_refinement(A: ArrayLike,
                         b: np.ndarray,
                         x0: Optional[np.ndarray] = None,
                         config: IRConfig = IRConfig(),
                         preconditioner: Optional[str] = None,
                         precond_kwargs: Optional[dict] = None) -> Tuple[np.ndarray, IRInfo]:
    assert A.shape[0] == A.shape[1] == b.shape[0]
    n = A.shape[0]
    precond_kwargs = precond_kwargs or {}

    # High-precision array/operator for residuals
    if isinstance(A, spla.LinearOperator):
        A64 = A
    elif sp.issparse(A):
        A64 = A.asformat('csr').astype(np.float64, copy=False)
    else:
        A64 = np.array(A, dtype=np.float64, copy=False)

    # Working-precision operator
    Aop = to_linear_operator(A, config.work_dtype)

    # Preconditioner (built in double; GMRES casts as needed)
    M = make_preconditioner(A64 if not isinstance(A64, spla.LinearOperator) else None, kind=preconditioner, **precond_kwargs)

    # Initial guess
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64, copy=True)

    A_norm_est = power_norm2(A64, iters=15)
    r = b.astype(np.float64) - (A64 @ x)
    res_hist = [np.linalg.norm(r)]
    used_precisions = [str(config.work_dtype)]
    escalations: List[str] = []
    notes: Dict[str, Any] = {}

    if config.estimate_kappa:
        kappa_hat = _kappa_proxy(A64, Aop, trials=config.kappa_checks)
        notes['kappa_proxy_initial'] = kappa_hat
        notes['u_work_times_kappa'] = unit_roundoff(config.work_dtype) * kappa_hat

    sched = AdaptiveScheduler(config.scheduler)
    rtol = config.rtol_inner
    work_dtype = config.work_dtype

    def _escalate_precision():
        nonlocal work_dtype, Aop
        if work_dtype != np.float64:
            work_dtype = np.float64
            Aop = to_linear_operator(A64, work_dtype)
            used_precisions.append('float64')
            return work_dtype, "precision_fp32_to_fp64"
        else:
            return work_dtype, None

    for k in range(config.k_max):
        if backward_error_ok(A_norm_est, x, b, r, config.tol):
            be = np.linalg.norm(r) / (A_norm_est * np.linalg.norm(x) + np.linalg.norm(b))
            return x, IRInfo(True, k, res_hist, used_precisions, escalations, be, A_norm_est, notes)

        d = _gmres(Aop, r, rtol_inner=rtol, maxit_inner=config.maxit_inner, M=M)
        x = x + d
        r = b.astype(np.float64) - (A64 @ x)
        res_hist.append(np.linalg.norm(r))

        # Scheduler decision
        new_rtol, new_dtype, note = sched.update_and_decide(res_hist, rtol, work_dtype, _escalate_precision)
        if note:
            escalations.append(f"{note} at iter {k+1}")
        rtol, work_dtype = new_rtol, new_dtype

        # Optional u*kappa check
        if config.estimate_kappa and (k == 0 or (k+1) % 3 == 0):
            kappa_hat = _kappa_proxy(A64, Aop, trials=1)
            notes.setdefault('kappa_proxy_iters', []).append(kappa_hat)
            if unit_roundoff(work_dtype) * kappa_hat > 0.1 and work_dtype != np.float64:
                work_dtype, note2 = _escalate_precision()
                if note2:
                    escalations.append(f"{note2} at iter {k+1}")

    converged = backward_error_ok(A_norm_est, x, b, r, config.tol)
    be = np.linalg.norm(r) / (A_norm_est * np.linalg.norm(x) + np.linalg.norm(b))
    return x, IRInfo(converged, len(res_hist)-1, res_hist, used_precisions, escalations, be, A_norm_est, notes)

def solve(A: ArrayLike,
          b: np.ndarray,
          tol: float = 1e-12,
          k_max: int = 10,
          inner_tol: float = 1e-3,
          work_dtype: np.dtype = np.float32,
          preconditioner: Optional[str] = None,
          precond_kwargs: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = IRConfig(tol=tol, k_max=k_max, rtol_inner=inner_tol, work_dtype=work_dtype)
    x, info = iterative_refinement(A, b, None, cfg, preconditioner, precond_kwargs)
    return x, asdict(info)
