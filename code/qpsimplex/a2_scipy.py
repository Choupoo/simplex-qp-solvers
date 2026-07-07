from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .metrics import kkt_residual, objective, relative_gap
from .problem import SimplexQP


@dataclass
class A2Result:
    x: np.ndarray
    value: float
    iterations: int
    elapsed: float
    success: bool
    status: str
    message: str
    kkt: Dict[str, float]


def _linear_constraint_matrix(problem: SimplexQP) -> np.ndarray:
    A = np.zeros((problem.m_blocks, problem.n))
    for k, block in enumerate(problem.blocks):
        A[k, block] = 1.0
    return A


def solve_with_scipy_slsqp(
    problem: SimplexQP,
    max_iter: int = 1000,
    ftol: float = 1e-10,
    x0: Optional[np.ndarray] = None,
) -> A2Result:
    if x0 is None:
        x0 = problem.uniform_feasible_point()
    Q = problem.Q
    q = problem.q
    A = _linear_constraint_matrix(problem)

    def fun(x: np.ndarray) -> float:
        return float(x @ Q @ x + q @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        return 2.0 * (Q @ x) + q

    cons = {"type": "eq", "fun": lambda x: A @ x - 1.0, "jac": lambda x: A}
    bounds = [(0.0, None)] * problem.n
    t0 = perf_counter()
    res = minimize(
        fun,
        x0,
        jac=jac,
        bounds=bounds,
        constraints=[cons],
        method="SLSQP",
        options={"maxiter": int(max_iter), "ftol": float(ftol), "disp": False},
    )
    elapsed = perf_counter() - t0
    x = np.asarray(res.x, dtype=float)
    return A2Result(
        x=x,
        value=objective(problem, x),
        iterations=int(getattr(res, "nit", -1)),
        elapsed=float(elapsed),
        success=bool(res.success),
        status=str(getattr(res, "status", "")),
        message=str(res.message),
        kkt=kkt_residual(problem, x),
    )
