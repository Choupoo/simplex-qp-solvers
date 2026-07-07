from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List

import numpy as np

from .metrics import objective
from .problem import SimplexQP


@dataclass
class FWPolishResult:
    x: np.ndarray
    value: float
    iterations: int
    elapsed: float
    fw_gap: float
    history: List[Dict[str, float]] = field(default_factory=list)


def frank_wolfe_polish(
    problem: SimplexQP,
    x0: np.ndarray,
    max_iter: int = 2000,
    tol_gap: float = 1e-8,
    log_every: int = 25,
) -> FWPolishResult:
    t0 = perf_counter()
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    # Repair tiny numerical errors in the starting point.
    x = np.maximum(x, 0.0)
    for block in problem.blocks:
        s = float(np.sum(x[block]))
        if s <= 0:
            x[block] = 1.0 / len(block)
        else:
            x[block] /= s
    history: List[Dict[str, float]] = []
    gap = np.inf
    Rx = problem.R @ x
    for it in range(1, max_iter + 1):
        grad = 2.0 * (problem.R.T @ Rx) + problem.q
        svec = np.zeros(problem.n)
        for block in problem.blocks:
            j = int(block[int(np.argmin(grad[block]))])
            svec[j] = 1.0
        d = svec - x
        gap = -float(grad @ d)
        val = float(Rx @ Rx + problem.q @ x)
        if it % log_every == 0 or it == 1:
            history.append({"iter": float(it), "value": val, "fw_gap": gap})
        if gap <= tol_gap:
            break
        Rd = problem.R @ d
        denom = 2.0 * float(Rd @ Rd)
        if denom <= 1e-30:
            alpha = 1.0
        else:
            alpha = min(1.0, max(0.0, gap / denom))
        if alpha <= 1e-16:
            break
        x += alpha * d
        Rx += alpha * Rd
    elapsed = perf_counter() - t0
    return FWPolishResult(x=x, value=objective(problem, x), iterations=it, elapsed=elapsed, fw_gap=float(gap), history=history)
