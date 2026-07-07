from __future__ import annotations

from typing import Dict

import numpy as np

from .problem import SimplexQP


def objective(problem: SimplexQP, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = problem.R @ x
    return float(y @ y + problem.q @ x)


def feasibility_residual(problem: SimplexQP, x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    sum_res = 0.0
    for block in problem.blocks:
        sum_res = max(sum_res, abs(float(np.sum(x[block]) - 1.0)))
    neg = max(0.0, float(-np.min(x))) if x.size else 0.0
    return {"block_sum_inf": sum_res, "nonneg_violation": neg, "max": max(sum_res, neg)}


def kkt_residual(problem: SimplexQP, x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    Qx = problem.R.T @ (problem.R @ x)
    grad = 2.0 * Qx + problem.q
    comp = 0.0
    stationarity = 0.0
    for block in problem.blocks:
        gb = grad[block]
        xb = x[block]
        tau = float(np.min(gb))
        stationarity = max(stationarity, float(np.max(np.maximum(tau - gb, 0.0))))
        comp = max(comp, float(np.max(np.abs(xb * (gb - tau)))))
    feas = feasibility_residual(problem, x)["max"]
    return {"feasibility": feas, "stationarity": stationarity, "complementarity": comp, "max": max(feas, stationarity, comp)}


def relative_gap(ub: float, lb: float) -> float:
    return float((ub - lb) / max(1.0, abs(ub)))
