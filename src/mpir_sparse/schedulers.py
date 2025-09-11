from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SchedulerConfig:
    stagnation_window: int = 4
    stagnation_ratio: float = 0.5
    min_rtol_inner: float = 1e-4

class AdaptiveScheduler:
    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg
        self.ratio_history: List[float] = []

    def update_and_decide(self, res_hist: List[float], rtol_inner: float, work_dtype, escalate_precision_cb):
        note = None
        new_tol = rtol_inner
        new_dtype = work_dtype
        w = self.cfg.stagnation_window
        if len(res_hist) >= w + 1 and min(res_hist[-w-1:-1]) > 0:
            window = res_hist[-(w+1):]
            ratios = [window[i+1]/window[i] for i in range(len(window)-1)]
            median_ratio = float(np.median(ratios))
            self.ratio_history.append(median_ratio)
            if median_ratio > self.cfg.stagnation_ratio:
                if rtol_inner > self.cfg.min_rtol_inner:
                    new_tol = max(self.cfg.min_rtol_inner, rtol_inner / 3.0)
                    note = f"tighten_tol {rtol_inner:.2e}->{new_tol:.2e}"
                else:
                    new_dtype, note2 = escalate_precision_cb()
                    note = note2 or "precision_escalation"
        return new_tol, new_dtype, note
