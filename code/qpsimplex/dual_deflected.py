from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np

from .metrics import objective, relative_gap
from .problem import SimplexQP



@dataclass
class DualOptions:
    max_iter: int = 10000
    tol_gap: float = 1e-5
    nu0: float = 1.0
    step_power: float = 0.75
    delta0: float = 1.0
    delta_power: float = 0.5
    alpha_min: float = 0.10
    verbose: bool = False
    log_every: int = 10

    def validate(self) -> None:
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.nu0 <= 0.0:
            raise ValueError("nu0 must be positive")
        if not (0.5 < self.step_power <= 1.0):
            raise ValueError("step_power must lie in (1/2, 1]")
        if self.delta0 <= 0.0:
            raise ValueError("delta0 must be positive")
        if self.delta_power <= 0.0:
            raise ValueError("delta_power must be positive")
        if not (0.0 < self.alpha_min <= 1.0):
            raise ValueError("alpha_min must lie in (0, 1]")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")


@dataclass
class DualResult:
    x: np.ndarray
    dual_lambda: np.ndarray
    primal_value: float
    dual_value: float
    gap: float
    relative_gap: float
    iterations: int
    elapsed: float
    status: str
    radius: float
    history: List[Dict[str, float]] = field(default_factory=list)


class DeflectedSubgradientSolver:
    def __init__(self, options: Optional[DualOptions] = None):
        self.options = options or DualOptions()
        self.options.validate()

    @staticmethod
    def _lagrangian_relaxation(problem: SimplexQP, lam: np.ndarray):
        reduced_cost = problem.q + problem.R.T @ lam
        x = np.zeros(problem.n)
        min_sum = 0.0
        for block in problem.blocks:
            local = int(np.argmin(reduced_cost[block]))
            j = int(block[local])
            x[j] = 1.0
            min_sum += float(reduced_cost[j])

        psi = -0.25 * float(lam @ lam) + min_sum
        # g in partial Phi(lambda), Phi=-psi.
        g = 0.5 * lam - problem.R @ x
        return psi, g, x

    @staticmethod
    def _project_ball(z: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0.0:
            return np.zeros_like(z)
        norm_z = float(np.linalg.norm(z))
        if norm_z <= radius:
            return z
        return (radius / norm_z) * z

    @staticmethod
    def _multiplier_radius(problem: SimplexQP) -> float:
        # rho = 2 ||R||_F sqrt(m) contains an unconstrained dual optimum.
        return float(2.0 * np.linalg.norm(problem.R, ord="fro") * np.sqrt(problem.m_blocks))

    def solve(self, problem: SimplexQP) -> DualResult:
        opt = self.options
        t0 = perf_counter()

        radius = self._multiplier_radius(problem)
        lam = np.zeros(problem.rank_dim)
        d_prev: Optional[np.ndarray] = None

        best_lb = -np.inf
        best_phi = np.inf
        best_lam = lam.copy()

        best_x = problem.uniform_feasible_point()
        best_ub = objective(problem, best_x)
        avg_x = np.zeros(problem.n)
        avg_weight_sum = 0.0

        history: List[Dict[str, float]] = []
        status = "max_iter"
        it = 0

        for it in range(1, opt.max_iter + 1):
            psi, g, x_vertex = self._lagrangian_relaxation(problem, lam)
            phi = -psi
            g_norm = float(np.linalg.norm(g))

            if psi > best_lb:
                best_lb = float(psi)
                best_phi = float(phi)
                best_lam = lam.copy()
            else:
                # Numerically, best_phi=-best_lb is more consistent than a
                # separate minimum updated under roundoff.
                best_phi = -best_lb

            nu = opt.nu0 / (it ** opt.step_power)

            # Every LR vertex is primal feasible.  Convex averages remain
            # feasible and provide valid primal upper bounds.
            # The report and experiments use the uniform running average.
            weight = 1.0
            avg_x = (avg_weight_sum * avg_x + x_vertex) / (avg_weight_sum + weight)
            avg_weight_sum += weight

            for candidate in (x_vertex, avg_x):
                value = objective(problem, candidate)
                if value < best_ub:
                    best_ub = float(value)
                    best_x = candidate.copy()

            if it == 1:
                delta = opt.delta0
                phi_level = best_phi - delta
                zeta = 1.0
                alpha = 1.0
                direction = g.copy()
            else:
                assert d_prev is not None
                delta = opt.delta0 / (it ** opt.delta_power)
                phi_level = best_phi - delta
                nu_prev = opt.nu0 / ((it - 1) ** opt.step_power)
                prev_norm2 = float(d_prev @ d_prev)
                denominator = float(phi - phi_level + nu_prev * prev_norm2)
                # phi-phi_level >= delta > 0, so the denominator is strictly
                # positive in exact arithmetic.  The max guards roundoff.
                denominator = max(denominator, np.finfo(float).tiny)
                zeta = float(nu_prev * prev_norm2 / denominator)
                zeta = min(1.0, max(0.0, zeta))
                # alpha >= zeta is exactly the condition in rule (4.15).
                # alpha_min also keeps a non-vanishing share of fresh oracle
                # information and satisfies the lower-bound convention used in
                # the paper's underlying analysis.
                alpha = max(opt.alpha_min, zeta)
                direction = alpha * g + (1.0 - alpha) * d_prev

            direction_norm = float(np.linalg.norm(direction))
            gap = max(0.0, best_ub - best_lb)
            relgap = relative_gap(best_ub, best_lb)

            if it == 1 or it % opt.log_every == 0:
                history.append(
                    {
                        "iter": float(it),
                        "dual": float(psi),
                        "phi": float(phi),
                        "best_dual": float(best_lb),
                        "best_phi": float(best_phi),
                        "best_primal": float(best_ub),
                        "gap": float(gap),
                        "rel_gap": float(relgap),
                        "subgrad_norm": float(g_norm),
                        "direction_norm": float(direction_norm),
                        "nu": float(nu),
                        "alpha": float(alpha),
                        "zeta": float(zeta),
                        "phi_level": float(phi_level),
                        "delta": float(delta),
                        "vertex_primal": float(objective(problem, x_vertex)),
                        "weak_duality_margin": float(objective(problem, x_vertex) - psi),
                        "lambda_norm": float(np.linalg.norm(lam)),
                        "radius": float(radius),
                    }
                )

            if opt.verbose and (it == 1 or it % max(1, opt.max_iter // 20) == 0):
                print(
                    f"it={it:6d} lb={best_lb: .8e} ub={best_ub: .8e} "
                    f"relgap={relgap:.3e} |g|={g_norm:.3e} "
                    f"nu={nu:.3e} alpha={alpha:.3e}"
                )

            if relgap <= opt.tol_gap:
                status = "optimal_gap"
                break
            trial = lam - nu * direction
            lam = self._project_ball(trial, radius)
            d_prev = direction

        elapsed = perf_counter() - t0
        final_gap = max(0.0, best_ub - best_lb)
        final_rel = relative_gap(best_ub, best_lb)
        return DualResult(
            x=best_x,
            dual_lambda=best_lam,
            primal_value=float(best_ub),
            dual_value=float(best_lb),
            gap=float(final_gap),
            relative_gap=float(final_rel),
            iterations=int(it),
            elapsed=float(elapsed),
            status=status,
            radius=float(radius),
            history=history,
        )
