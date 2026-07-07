#!/usr/bin/env python3
"""Lightweight correctness checks for the project code."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpsimplex.a2_scipy import solve_with_scipy_slsqp
from qpsimplex.dual_deflected import DeflectedSubgradientSolver, DualOptions
from qpsimplex.metrics import feasibility_residual, objective
from qpsimplex.problem import SimplexQP, generate_instance
from qpsimplex.primal_recovery import frank_wolfe_polish


def main() -> None:
    problem = generate_instance(4, 6, rank_dim=10, density=0.7, seed=99, name="self_check")
    by_q = SimplexQP.from_Q(problem.Q, problem.q, problem.blocks, name="self_check_from_Q")
    x0 = problem.uniform_feasible_point()
    err = abs(objective(problem, x0) - objective(by_q, x0))
    assert err < 1e-9, f"Q factorization changed objective by {err}"

    options = DualOptions(
        max_iter=5000,
        nu0=0.5,
        step_power=0.75,
        delta0=1.0,
        delta_power=0.5,
        alpha_min=0.1,
        log_every=1,
    )
    solver = DeflectedSubgradientSolver(options)
    # Check the explicit ball projection independently of the main run.
    test_radius = solver._multiplier_radius(by_q)
    projected = solver._project_ball(2.0 * test_radius * (by_q.R[:, 0] * 0.0 + 1.0), test_radius)
    assert abs(float(np.linalg.norm(projected)) - test_radius) < 1e-9

    a1 = solver.solve(by_q)
    polish = frank_wolfe_polish(by_q, a1.x, max_iter=1000, tol_gap=1e-8)

    assert feasibility_residual(by_q, polish.x)["max"] < 1e-9
    assert polish.value + 1e-7 >= a1.dual_value, "dual bound exceeds feasible primal value"
    assert a1.status in {"optimal_gap", "max_iter"}
    assert all(h["lambda_norm"] <= h["radius"] + 1e-9 for h in a1.history)
    assert all(h["direction_norm"] <= h["radius"] + 1e-8 for h in a1.history)
    assert all(-1e-12 <= h["zeta"] <= h["alpha"] + 1e-12 for h in a1.history)
    assert all(h["alpha"] <= 1.0 + 1e-12 for h in a1.history)
    assert all(h["delta"] > 0.0 for h in a1.history)
    assert all(h["phi"] - h["phi_level"] > 0.0 for h in a1.history)
    assert all(h["weak_duality_margin"] >= -1e-8 for h in a1.history)

    a2_cold = solve_with_scipy_slsqp(by_q, x0=None, max_iter=500)
    a2_warm = solve_with_scipy_slsqp(by_q, x0=polish.x, max_iter=500)
    rel = (polish.value - a2_cold.value) / max(1.0, abs(a2_cold.value))
    assert rel < 3e-3, f"A1 is unexpectedly far from cold-start A2: {rel}"
    a2_modes_rel = abs(a2_cold.value - a2_warm.value) / max(1.0, abs(a2_cold.value))
    assert a2_modes_rel < 1e-7, f"A2 cold and warm starts disagree: {a2_modes_rel}"

    print("self-check passed: theorem-rule invariants and numerical bounds verified")
    print(f"factorization error on uniform point: {err:.3e}")
    print(
        f"A1 value={polish.value:.8f}, dual={a1.dual_value:.8f}, "
        f"A2 cold={a2_cold.value:.8f}, A2 warm={a2_warm.value:.8f}, "
        f"rel={rel:.3e}, radius={a1.radius:.3e}"
    )


if __name__ == "__main__":
    main()
