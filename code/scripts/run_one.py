#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from qpsimplex.a2_scipy import solve_with_scipy_slsqp
from qpsimplex.dual_deflected import DeflectedSubgradientSolver, DualOptions
from qpsimplex.metrics import kkt_residual
from qpsimplex.problem import generate_instance, load_instance
from qpsimplex.primal_recovery import frank_wolfe_polish


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve one product-simplex QP instance.")
    parser.add_argument("--instance", type=str, help="Path to .npz instance. If omitted, generate one.")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=15)
    parser.add_argument("--rank", type=int, default=40)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--nu0", type=float, default=0.5)
    parser.add_argument("--step-power", type=float, default=0.75)
    parser.add_argument("--delta0", type=float, default=1.0)
    parser.add_argument("--delta-power", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    args = parser.parse_args()

    if args.instance:
        problem = load_instance(args.instance)
    else:
        problem = generate_instance(args.blocks, args.block_size, args.rank, args.density, args.seed)

    options = DualOptions(
        max_iter=args.max_iter,
        nu0=args.nu0,
        step_power=args.step_power,
        delta0=args.delta0,
        delta_power=args.delta_power,
        alpha_min=args.alpha_min,
        verbose=True,
        log_every=25,
    )
    a1 = DeflectedSubgradientSolver(options).solve(problem)
    polish = frank_wolfe_polish(problem, a1.x, max_iter=1500, tol_gap=1e-7)
    a1.x = polish.x
    a1.primal_value = polish.value
    a1.gap = max(0.0, a1.primal_value - a1.dual_value)
    a1.relative_gap = a1.gap / max(1.0, abs(a1.primal_value))
    a1.elapsed += polish.elapsed

    a2_cold = solve_with_scipy_slsqp(problem, x0=None)
    a2_warm = solve_with_scipy_slsqp(problem, x0=a1.x)

    print("\nA1 target-value deflected subgradient")
    print(f"value       : {a1.primal_value:.12g}")
    print(f"dual bound  : {a1.dual_value:.12g}")
    print(f"rel gap     : {a1.relative_gap:.3e}")
    print(f"dual iters  : {a1.iterations}")
    print(f"polish iters: {polish.iterations}")
    print(f"radius      : {a1.radius:.3e}")
    print(f"time        : {a1.elapsed:.3f}s")
    print(f"KKT residual: {kkt_residual(problem, a1.x)['max']:.3e}")

    print("\nA2 SciPy SLSQP - default/cold start")
    print(f"value       : {a2_cold.value:.12g}")
    print(f"success     : {a2_cold.success} ({a2_cold.message})")
    print(f"iterations  : {a2_cold.iterations}")
    print(f"time        : {a2_cold.elapsed:.3f}s")
    print(f"KKT residual: {a2_cold.kkt['max']:.3e}")

    print("\nA2 SciPy SLSQP - warm start from A1")
    print(f"value       : {a2_warm.value:.12g}")
    print(f"success     : {a2_warm.success} ({a2_warm.message})")
    print(f"iterations  : {a2_warm.iterations}")
    print(f"A2-only time: {a2_warm.elapsed:.3f}s")
    print(f"pipeline time (A1 + A2): {a1.elapsed + a2_warm.elapsed:.3f}s")
    print(f"KKT residual: {a2_warm.kkt['max']:.3e}")


if __name__ == "__main__":
    main()
