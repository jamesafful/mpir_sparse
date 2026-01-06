from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .utils import (
    to_linear_operator,
    power_norm2,
    unit_roundoff,
    backward_error_ok,
    diagonal_scaling,
)
from .schedulers import AdaptiveScheduler, SchedulerConfig
from .preconditioners import make_preconditioner_with_info
from .tracing import JSONLTraceWriter

ArrayLike = Union[np.ndarray, sp.spmatrix, spla.LinearOperator]


@dataclass
class IRConfig:
    # Outer refinement
    tol: float = 1e-12
    k_max: int = 10

    # Inner Krylov solve
    rtol_inner: float = 1e-3
    atol_inner: float = 0.0
    maxit_inner: int = 200
    restart: Optional[int] = None
    inner_solver: str = "gmres"  # "gmres" or "lgmres"

    # Precision
    work_dtype: np.dtype = np.float32
    residual_dtype: np.dtype = np.float64

    # Robustness features
    scaling: Optional[str] = None  # None or "diag"
    residual_replacement: int = 0  # recompute residual every N outer steps (0 disables; residual is always computed in high precision anyway)
    max_inner_retries: int = 1     # extra attempts on inner solver failure
    # Adaptivity switches
    adaptive: bool = True  # if False, disables scheduler decisions

    # Tracing (JSONL)
    trace_path: Optional[str] = None  # if set, writes a JSONL trace
    trace_every: int = 1  # write every N outer iterations

    # Preconditioner refresh (ILU only, simple parameter tightening)
    precond_refresh: bool = False
    precond_refresh_max: int = 3
    ilu_drop_tol_min: float = 1e-8
    ilu_fill_factor_max: float = 50.0
    ilu_drop_tol_shrink: float = 0.1  # drop_tol *= shrink on refresh
    ilu_fill_factor_grow: float = 1.5  # fill_factor *= grow on refresh

    # Diagnostics / adaptivity
    estimate_kappa: bool = True
    kappa_checks: int = 2
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class IRInfo:
    converged: bool
    iters: int
    residual_history: List[float]
    used_precisions: List[str]
    escalations: List[str]
    final_backward_error: float
    A_norm_est: float
    notes: Dict[str, Any]
    trace_path: Optional[str] = None
    preconditioner_history: List[Dict[str, Any]] = field(default_factory=list)


def _krylov_solve(
    solver: str,
    Aop: spla.LinearOperator,
    rhs64: np.ndarray,
    rtol: float,
    atol: float,
    maxit: int,
    restart: Optional[int],
    M: Optional[spla.LinearOperator],
) -> Tuple[np.ndarray, int, int]:
    """Solve A d ≈ rhs using a chosen Krylov method.

    Returns (d, info_flag, inner_iters) where info_flag is 0 on success.
    """
    rhs = rhs64.astype(Aop.dtype, copy=False)

    inner_iters = 0
    def _cb(*args, **kwargs):
        nonlocal inner_iters
        inner_iters += 1

    if solver.lower() == "gmres":
        # SciPy API changed around 1.11 (rtol/atol vs tol).
        # We attach a callback to count inner iterations in a version-tolerant way.
        try:
            d, info = spla.gmres(
                Aop, rhs, M=M, restart=restart, maxiter=maxit, rtol=rtol, atol=atol,
                callback=_cb, callback_type='pr_norm'
            )
        except TypeError:
            # Older SciPy: tol parameter; may not accept atol/restart/callback_type
            kwargs: Dict[str, Any] = {'M': M, 'maxiter': maxit, 'tol': rtol, 'callback': _cb}
            if restart is not None:
                kwargs['restart'] = restart
            d, info = spla.gmres(Aop, rhs, **kwargs)
        return np.asarray(d, dtype=Aop.dtype), int(info), int(inner_iters)

    if solver.lower() == "lgmres":
        # LGMRES is a flexible variant (handles varying/inexact preconditioning better).
        try:
            d, info = spla.lgmres(Aop, rhs, M=M, maxiter=maxit, tol=rtol, callback=_cb)
        except TypeError:
            # Some versions accept rtol/atol; keep compatibility
            d, info = spla.lgmres(Aop, rhs, M=M, maxiter=maxit, rtol=rtol, atol=atol, callback=_cb)
        return np.asarray(d, dtype=Aop.dtype), int(info), int(inner_iters)

    raise ValueError(f"Unknown inner_solver: {solver}")


def _kappa_proxy(A64, Aop_work, trials: int = 2) -> float:
    # Cheap proxy: ||A|| * max ||y||/||A y|| over a few random solves.
    Anorm = power_norm2(A64, iters=10)
    inv_norm_est = 0.0
    n = A64.shape[1]
    rng = np.random.default_rng(0)
    for _ in range(trials):
        g = rng.standard_normal(n).astype(np.float64)
        y, info, _ = _krylov_solve("gmres", Aop_work, g, rtol=1e-2, atol=0.0, maxit=200, restart=None, M=None)
        if info != 0:
            continue
        Ay = (A64 @ y.astype(np.float64))
        num = float(np.linalg.norm(y))
        den = float(np.linalg.norm(Ay))
        if den > 0:
            inv_norm_est = max(inv_norm_est, num / den)
    return float(Anorm * inv_norm_est)


def iterative_refinement(
    A: ArrayLike,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    config: IRConfig = IRConfig(),
    preconditioner: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, IRInfo]:
    """Adaptive mixed-precision iterative refinement for Ax=b.

    - Residuals are computed in config.residual_dtype (default float64).
    - Inner solves use a Krylov method in config.work_dtype (default float32) and can adapt.
    """
    assert A.shape[0] == A.shape[1] == b.shape[0]
    n = A.shape[0]
    precond_kwargs = precond_kwargs or {}

    notes: Dict[str, Any] = {}

    # Optional symmetric diagonal scaling (matrix form only)
    unscale = lambda y: y
    A_work = A
    b_work = b.astype(np.float64, copy=False)
    if config.scaling is not None:
        if config.scaling.lower() != "diag":
            raise ValueError(f"Unknown scaling: {config.scaling}")
        A_work, b_work, unscale, s_info = diagonal_scaling(A, b.astype(np.float64), min_abs_diag=1e-30)
        notes["scaling"] = s_info

    # High-precision representation for residuals and A-norm estimate
    if isinstance(A_work, spla.LinearOperator):
        A64 = A_work
    elif sp.issparse(A_work):
        A64 = A_work.asformat("csr").astype(config.residual_dtype, copy=False)
    else:
        A64 = np.array(A_work, dtype=config.residual_dtype, copy=False)

    # Preconditioner built from explicit matrix if available
    M = None
    prec_info = None
    if preconditioner is not None:
        if isinstance(A64, spla.LinearOperator):
            raise ValueError("Cannot build matrix-based preconditioner when A is a LinearOperator. Provide an explicit sparse matrix.")
        M, prec_info = make_preconditioner_with_info(A64, kind=preconditioner, **precond_kwargs)
        if prec_info is not None:
            notes["preconditioner"] = {"kind": prec_info.kind, **prec_info.params}

    prec_hist: List[Dict[str, Any]] = []
    if prec_info is not None:
        prec_hist.append({'kind': prec_info.kind, **prec_info.params})
    refresh_count = 0

    # Initial guess in high precision (solution is returned in the original variable after unscale)
    x = np.zeros(n, dtype=config.residual_dtype) if x0 is None else np.array(x0, dtype=config.residual_dtype, copy=True)

    # Working-precision operator for Krylov
    work_dtype = config.work_dtype
    Aop = to_linear_operator(A64, work_dtype)

    A_norm_est = float(power_norm2(A64, iters=15))
    r = b_work.astype(config.residual_dtype, copy=False) - (A64 @ x)
    res_hist = [float(np.linalg.norm(r))]

    # Optional JSONL tracing
    trace = None
    if config.trace_path is not None:
        trace = JSONLTraceWriter(config.trace_path)
        trace.write({
            'event': 'init',
            'n': int(n),
            'preconditioner': preconditioner,
            'work_dtype': str(work_dtype),
            'rtol_inner': float(config.rtol_inner),
            'restart': config.restart,
            'inner_solver': config.inner_solver,
            'res_norm': float(res_hist[-1]),
        })
    used_precisions = [str(work_dtype)]
    eta_history: List[float] = []  # inner correction quality ||A d - r|| / ||r||
    inner_iters_history: List[int] = []
    escalations: List[str] = []

    # Initial kappa proxy
    if config.estimate_kappa:
        kappa_hat = _kappa_proxy(A64, Aop, trials=config.kappa_checks)
        notes["kappa_proxy_initial"] = float(kappa_hat)
        notes["u_work_times_kappa_initial"] = float(unit_roundoff(work_dtype) * kappa_hat)

    sched = AdaptiveScheduler(config.scheduler)
    rtol = float(config.rtol_inner)

    def _rebuild_operator(new_dtype: np.dtype) -> None:
        nonlocal Aop, work_dtype
        work_dtype = new_dtype
        Aop = to_linear_operator(A64, work_dtype)
        used_precisions.append(str(work_dtype))

    def _escalate_precision() -> Tuple[np.dtype, Optional[str]]:
        if work_dtype != np.float64:
            _rebuild_operator(np.float64)
            return work_dtype, "precision_fp32_to_fp64"
        return work_dtype, None

    for k in range(config.k_max):
        action_notes: List[str] = []
        r_norm_before = float(np.linalg.norm(r))
        if backward_error_ok(A_norm_est, x, b_work, r, config.tol):
            be = float(np.linalg.norm(r) / (A_norm_est * np.linalg.norm(x) + np.linalg.norm(b_work)))
            x_out = unscale(x)
            notes['eta_history'] = eta_history
            notes['inner_iters_history'] = inner_iters_history
            if trace is not None:
                trace.write({'event': 'converged', 'iter': int(k), 'res_norm': float(res_hist[-1]), 'backward_error': float(be)})
                trace.close()
            return x_out, IRInfo(True, k, res_hist, used_precisions, escalations, be, A_norm_est, notes, trace_path=config.trace_path, preconditioner_history=prec_hist)

        # Inner solve with a small retry ladder
        attempt = 0
        local_rtol = rtol
        local_maxit = config.maxit_inner
        local_solver = config.inner_solver
        d = None

        while True:
            d_try, info, inner_iters = _krylov_solve(
                local_solver, Aop, r, rtol=local_rtol, atol=config.atol_inner,
                maxit=local_maxit, restart=config.restart, M=M
            )
            if info == 0:
                d = d_try.astype(config.residual_dtype, copy=False)
                break

            attempt += 1
            if attempt > config.max_inner_retries:
                raise RuntimeError(
                    f"Inner {local_solver} failed: info={info} after {attempt} attempt(s). "
                    f"Try increasing maxit_inner/restart or using a preconditioner."
                )

            # Recovery ladder: tighten tol -> increase maxit -> escalate precision
            if local_rtol > 1e-8:
                new_rtol = max(1e-8, local_rtol / 10.0)
                escalations.append(f"inner_retry_tighten_tol {local_rtol:.1e}->{new_rtol:.1e} at iter {k+1}")
                local_rtol = new_rtol
                continue
            if local_maxit < 5 * config.maxit_inner:
                new_maxit = int(min(5 * config.maxit_inner, max(local_maxit * 2, local_maxit + 50)))
                escalations.append(f"inner_retry_increase_maxit {local_maxit}->{new_maxit} at iter {k+1}")
                local_maxit = new_maxit
                continue

            _, note = _escalate_precision()
            if note:
                escalations.append(f"{note} at iter {k+1}")
            local_solver = "lgmres"  # more forgiving when everything else is struggling
            escalations.append(f"inner_retry_switch_solver to lgmres at iter {k+1}")

        assert d is not None

        # Inner correction quality (eta): ||A d - r|| / ||r||, measured in high precision
        r_before = r
        rnorm = float(np.linalg.norm(r_before))
        if rnorm == 0.0:
            eta = 0.0
        else:
            eta = float(np.linalg.norm((A64 @ d) - r_before) / rnorm)
        eta_history.append(eta)
        inner_iters_history.append(int(inner_iters))

        # Apply correction and recompute residual in high precision
        x = x + d
        r = b_work.astype(config.residual_dtype, copy=False) - (A64 @ x)
        res_hist.append(float(np.linalg.norm(r)))
        rho = float(res_hist[-1] / res_hist[-2]) if res_hist[-2] > 0 else float('inf')

        # Scheduler decision (tighten inner tol or escalate precision)
        if config.adaptive:
            new_rtol, new_dtype, note = sched.update_and_decide(res_hist, rtol, work_dtype, _escalate_precision)
            if note:
                escalations.append(f"{note} at iter {k+1}")
                action_notes.append(note)
            rtol = float(new_rtol)
            if new_dtype != work_dtype:
                _rebuild_operator(new_dtype)
                action_notes.append(f"dtype->{new_dtype}")
        else:
            note = None

        # Optional ILU preconditioner refresh (simple tightening) when progress is poor but inner solve is effective
        if (config.precond_refresh and preconditioner == 'ilu' and prec_info is not None and refresh_count < config.precond_refresh_max):
            # Heuristic: if outer residual isn't contracting but eta suggests the correction solve is accurate, refresh ILU.
            if (rho > config.scheduler.stagnation_ratio) and (eta <= 0.2):
                drop_tol = float(precond_kwargs.get('drop_tol', prec_info.params.get('drop_tol', 1e-4)))
                fill_factor = float(precond_kwargs.get('fill_factor', prec_info.params.get('fill_factor', 10.0)))
                new_drop = max(config.ilu_drop_tol_min, drop_tol * config.ilu_drop_tol_shrink)
                new_fill = min(config.ilu_fill_factor_max, fill_factor * config.ilu_fill_factor_grow)
                if (new_drop < drop_tol) or (new_fill > fill_factor):
                    precond_kwargs['drop_tol'] = new_drop
                    precond_kwargs['fill_factor'] = new_fill
                    M, prec_info = make_preconditioner_with_info(A64, kind='ilu', **precond_kwargs)
                    refresh_count += 1
                    prec_hist.append({'kind': prec_info.kind, **prec_info.params})
                    msg = f"ilu_refresh drop_tol {drop_tol:.1e}->{new_drop:.1e}, fill {fill_factor:.1f}->{new_fill:.1f}"
                    escalations.append(f"{msg} at iter {k+1}")
                    action_notes.append(msg)

        # Optional periodic kappa monitoring
        if config.estimate_kappa and (k == 0 or (k + 1) % 3 == 0):
            kappa_hat = _kappa_proxy(A64, Aop, trials=1)
            notes.setdefault("kappa_proxy_iters", []).append(float(kappa_hat))
            if unit_roundoff(work_dtype) * kappa_hat > 0.1 and work_dtype != np.float64:
                _, note2 = _escalate_precision()
                if note2:
                    escalations.append(f"{note2} at iter {k+1}")


        # Trace this outer iteration
        if trace is not None and config.trace_every > 0 and ((k + 1) % config.trace_every == 0):
            trace.write({
                'event': 'outer_iter',
                'iter': int(k + 1),
                'res_norm_before': float(r_norm_before),
                'res_norm_after': float(res_hist[-1]),
                'rho': float(rho),
                'eta': float(eta),
                'inner_iters': int(inner_iters_history[-1]),
                'rtol_inner': float(rtol),
                'work_dtype': str(work_dtype),
                'inner_solver': str(local_solver),
                'actions': list(action_notes),
            })
    converged = backward_error_ok(A_norm_est, x, b_work, r, config.tol)
    be = float(np.linalg.norm(r) / (A_norm_est * np.linalg.norm(x) + np.linalg.norm(b_work)))
    x_out = unscale(x)
    notes['eta_history'] = eta_history
    notes['inner_iters_history'] = inner_iters_history
    if trace is not None:
        trace.write({'event': 'done', 'iter': int(len(res_hist)-1), 'res_norm': float(res_hist[-1]), 'backward_error': float(be), 'converged': bool(converged)})
        trace.close()
    return x_out, IRInfo(converged, len(res_hist) - 1, res_hist, used_precisions, escalations, be, A_norm_est, notes, trace_path=config.trace_path, preconditioner_history=prec_hist)


def solve(
    A: ArrayLike,
    b: np.ndarray,
    tol: float = 1e-12,
    k_max: int = 10,
    inner_tol: float = 1e-3,
    work_dtype: np.dtype = np.float32,
    preconditioner: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
    inner_solver: str = "gmres",
    restart: Optional[int] = None,
    scaling: Optional[str] = None,
    adaptive: bool = True,
    trace_path: Optional[str] = None,
    trace_every: int = 1,
    precond_refresh: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience wrapper around iterative_refinement.

    Returns (x, info_dict).
    """
    cfg = IRConfig(
        tol=tol,
        k_max=k_max,
        rtol_inner=inner_tol,
        work_dtype=work_dtype,
        inner_solver=inner_solver,
        restart=restart,
        scaling=scaling,
        adaptive=adaptive,
        trace_path=trace_path,
        trace_every=trace_every,
        precond_refresh=precond_refresh,
    )
    x, info = iterative_refinement(A, b, None, cfg, preconditioner, precond_kwargs)
    return x, asdict(info)
