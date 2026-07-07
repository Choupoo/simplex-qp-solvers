from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from .a2_scipy import solve_with_scipy_slsqp
from .dual_deflected import DeflectedSubgradientSolver, DualOptions
from .metrics import feasibility_residual, kkt_residual
from .problem import SimplexQP, generate_instance, save_instance
from .primal_recovery import frank_wolfe_polish


def experiment_grid() -> List[Dict[str, float]]:
    """Representative dense and sparse families used in the report."""
    return [
        {"name": "small_dense", "n_blocks": 5, "block_size": 10, "rank_dim": 20, "density": 1.0},
        {"name": "medium_dense", "n_blocks": 10, "block_size": 10, "rank_dim": 35, "density": 1.0},
        {"name": "medium_sparse", "n_blocks": 15, "block_size": 10, "rank_dim": 45, "density": 0.20},
        {"name": "large_sparse", "n_blocks": 20, "block_size": 8, "rank_dim": 50, "density": 0.12},
    ]


def default_dual_options(seed: int = 0) -> DualOptions:
    del seed  
    return DualOptions(
        max_iter=5000,
        tol_gap=1e-5,
        nu0=0.5,
        step_power=0.75,
        delta0=1.0,
        delta_power=0.5,
        alpha_min=0.10,
        log_every=10,
    )


def run_single(problem: SimplexQP, seed: int = 0) -> Dict[str, float | str]:
    a1_raw = DeflectedSubgradientSolver(default_dual_options(seed)).solve(problem)
    polish = frank_wolfe_polish(problem, a1_raw.x, max_iter=1500, tol_gap=1e-7, log_every=50)
    a1 = a1_raw
    a1.x = polish.x
    a1.primal_value = polish.value
    a1.gap = max(0.0, a1.primal_value - a1.dual_value)
    a1.relative_gap = a1.gap / max(1.0, abs(a1.primal_value))
    a1.elapsed += polish.elapsed

    a2_cold = solve_with_scipy_slsqp(problem, max_iter=800, ftol=1e-10, x0=None)
    a2_warm = solve_with_scipy_slsqp(problem, max_iter=800, ftol=1e-10, x0=a1.x)

    kkt_a1 = kkt_residual(problem, a1.x)
    feas_a1 = feasibility_residual(problem, a1.x)
    rel_to_a2_cold = (a1.primal_value - a2_cold.value) / max(1.0, abs(a2_cold.value))
    rel_to_a2_warm = (a1.primal_value - a2_warm.value) / max(1.0, abs(a2_warm.value))
    warm_speedup = a2_cold.elapsed / max(a2_warm.elapsed, 1e-15)
    pipeline_time = a1.elapsed + a2_warm.elapsed

    return {
        "instance": problem.name,
        "n": problem.n,
        "blocks": problem.m_blocks,
        "rank_dim": problem.rank_dim,
        "avg_block_size": np.mean(problem.block_sizes()),
        "a1_value": a1.primal_value,
        "a1_dual_bound": a1.dual_value,
        "a1_gap": a1.gap,
        "a1_rel_gap": a1.relative_gap,
        "a1_iter": a1.iterations,
        "a1_time": a1.elapsed,
        "a1_status": a1.status,
        "a1_radius": a1.radius,
        "a1_polish_iter": polish.iterations,
        "a1_polish_gap": polish.fw_gap,
        "a1_kkt": kkt_a1["max"],
        "a1_feas": feas_a1["max"],
        "a2_cold_value": a2_cold.value,
        "a2_cold_iter": a2_cold.iterations,
        "a2_cold_time": a2_cold.elapsed,
        "a2_cold_success": str(a2_cold.success),
        "a2_cold_kkt": a2_cold.kkt["max"],
        "a2_warm_value": a2_warm.value,
        "a2_warm_iter": a2_warm.iterations,
        "a2_warm_time": a2_warm.elapsed,
        "a2_warm_success": str(a2_warm.success),
        "a2_warm_kkt": a2_warm.kkt["max"],
        "a1_plus_a2_warm_time": pipeline_time,
        "a2_warm_speedup_vs_cold": warm_speedup,
        "rel_value_diff_a1_minus_a2_cold": rel_to_a2_cold,
        "rel_value_diff_a1_minus_a2_warm": rel_to_a2_warm,
    }


def run_grid(out_dir: str | Path, seeds=(0, 1)) -> Path:
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float | str]] = []

    for spec in experiment_grid():
        for seed in seeds:
            name = f"{spec['name']}_seed{seed}"
            problem = generate_instance(
                n_blocks=int(spec["n_blocks"]),
                block_size=int(spec["block_size"]),
                rank_dim=int(spec["rank_dim"]),
                density=float(spec["density"]),
                seed=int(seed),
                name=name,
            )
            save_instance(problem, data_dir / f"{name}.npz")
            row = run_single(problem, seed=seed)
            rows.append(row)
            print(
                f"done {name}: A1={row['a1_value']:.6g}, "
                f"A2-cold={row['a2_cold_value']:.6g}, "
                f"A2-warm={row['a2_warm_value']:.6g}, "
                f"rel_diff={row['rel_value_diff_a1_minus_a2_cold']:.2e}, "
                f"cert_gap={row['a1_rel_gap']:.2e}"
            )

    csv_path = out_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path
