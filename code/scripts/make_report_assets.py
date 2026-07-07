#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qpsimplex.a2_scipy import solve_with_scipy_slsqp
from qpsimplex.dual_deflected import DeflectedSubgradientSolver, DualOptions
from qpsimplex.metrics import kkt_residual
from qpsimplex.problem import SimplexQP, generate_instance
from qpsimplex.primal_recovery import frank_wolfe_polish

RESULTS = ROOT / "results"
REPORT = ROOT / "report_assets"
FIGURES = REPORT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
EOL = r" \\"


def fmt_e(x: float) -> str:
    return f"{float(x):.2e}"


def escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def latex_row(cells) -> str:
    return " & ".join(str(c) for c in cells) + EOL


def make_main_tables() -> None:
    df = pd.read_csv(RESULTS / "summary.csv")

    # Detailed table: accuracy/certification plus both A2 initializations.
    rows = [
        r"\begin{tabular}{lrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Instance & $n$ & A1 obj. & A2-c obj. & A2-w obj. & cert. gap & A1 it. & A1 s & A2-c it. & A2-c s & A2-w it. & A2-w s & pipeline s & warm speedup & max KKT \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        rows.append(
            latex_row(
                [
                    escape(row["instance"]),
                    int(row["n"]),
                    f"{row['a1_value']:.4f}",
                    f"{row['a2_cold_value']:.4f}",
                    f"{row['a2_warm_value']:.4f}",
                    fmt_e(row["a1_rel_gap"]),
                    int(row["a1_iter"]),
                    f"{row['a1_time']:.2f}",
                    int(row["a2_cold_iter"]),
                    f"{row['a2_cold_time']:.2f}",
                    int(row["a2_warm_iter"]),
                    f"{row['a2_warm_time']:.2f}",
                    f"{row['a1_plus_a2_warm_time']:.2f}",
                    f"{row['a2_warm_speedup_vs_cold']:.2f}",
                    fmt_e(max(row["a1_kkt"], row["a2_cold_kkt"], row["a2_warm_kkt"])),
                ]
            )
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    (REPORT / "results_table.tex").write_text("\n".join(rows))

    df["family"] = df["instance"].str.replace(r"_seed\d+$", "", regex=True)
    agg = (
        df.groupby("family", sort=False)
        .agg(
            n=("n", "mean"),
            a1=("a1_time", "mean"),
            cold=("a2_cold_time", "mean"),
            warm=("a2_warm_time", "mean"),
            pipeline=("a1_plus_a2_warm_time", "mean"),
            speedup=("a2_warm_speedup_vs_cold", "mean"),
            cold_rel=("rel_value_diff_a1_minus_a2_cold", lambda x: np.max(np.abs(x))),
            warm_rel=("rel_value_diff_a1_minus_a2_warm", lambda x: np.max(np.abs(x))),
        )
        .reset_index()
    )
    agg["cold_over_a1"] = agg["cold"] / agg["a1"]
    agg["pipeline_over_cold"] = agg["pipeline"] / agg["cold"]
    rows = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Family & $n$ & A1 s & A2-c s & A2-w s & pipeline s & cold/A1 & pipeline/cold & warm speedup & max rel. diff. \\",
        r"\midrule",
    ]
    for _, row in agg.iterrows():
        rows.append(
            latex_row(
                [
                    escape(row["family"]),
                    int(round(row["n"])),
                    f"{row['a1']:.2f}",
                    f"{row['cold']:.2f}",
                    f"{row['warm']:.2f}",
                    f"{row['pipeline']:.2f}",
                    f"{row['cold_over_a1']:.1f}",
                    f"{row['pipeline_over_cold']:.2f}",
                    f"{row['speedup']:.2f}",
                    fmt_e(max(row["cold_rel"], row["warm_rel"])),
                ]
            )
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    (REPORT / "aggregate_table.tex").write_text("\n".join(rows))

    plt.figure(figsize=(6, 3.5))
    plt.bar(np.arange(len(df)), df["rel_value_diff_a1_minus_a2_cold"].abs())
    plt.xticks(np.arange(len(df)), df["instance"], rotation=60, ha="right", fontsize=7)
    plt.ylabel("absolute relative difference: A1 vs A2 cold")
    plt.tight_layout()
    plt.savefig(FIGURES / "relative_difference.pdf")
    plt.close()

    plt.figure(figsize=(7, 3.8))
    x = np.arange(len(agg))
    width = 0.22
    plt.bar(x - 1.5 * width, agg["a1"], width, label="A1")
    plt.bar(x - 0.5 * width, agg["cold"], width, label="A2 cold")
    plt.bar(x + 0.5 * width, agg["warm"], width, label="A2 warm (A2 only)")
    plt.bar(x + 1.5 * width, agg["pipeline"], width, label="A1 + A2 warm")
    plt.xticks(x, agg["family"], rotation=30, ha="right")
    plt.ylabel("seconds")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "time_scaling.pdf")
    plt.close()

def run_convergence_asset() -> None:
    problem = generate_instance(10, 10, rank_dim=35, density=1.0, seed=0, name="convergence_medium_dense")
    options = DualOptions(
        max_iter=5000,
        nu0=0.5,
        step_power=0.75,
        delta0=1.0,
        delta_power=0.5,
        alpha_min=0.1,
        log_every=5,
    )
    result = DeflectedSubgradientSolver(options).solve(problem)
    history = pd.DataFrame(result.history)
    history.to_csv(RESULTS / "convergence_history.csv", index=False)

    plt.figure(figsize=(6, 3.5))
    plt.plot(history["iter"], history["best_primal"], label="best feasible UB")
    plt.plot(history["iter"], history["best_dual"], label="record dual LB")
    plt.xlabel("dual iteration")
    plt.ylabel("objective bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "convergence_bounds.pdf")
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.semilogy(history["iter"], history["rel_gap"].clip(lower=1e-12))
    plt.xlabel("dual iteration")
    plt.ylabel("relative primal-dual certificate")
    plt.tight_layout()
    plt.savefig(FIGURES / "convergence_gap.pdf")
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(history["iter"], history["alpha"], label=r"$\alpha_k$")
    plt.plot(history["iter"], history["nu"], label=r"$\nu_k$")
    plt.xlabel("dual iteration")
    plt.ylabel("parameter value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "deflection_steps.pdf")
    plt.close()


def run_parameter_sensitivity() -> None:
    problem = generate_instance(10, 10, rank_dim=35, density=1.0, seed=0, name="sensitivity_medium_dense")
    a2 = solve_with_scipy_slsqp(problem, max_iter=800, ftol=1e-10)

    configs = [
        ("nu0=0.2", 0.2, 0.75, 1.0, 0.5, 0.10),
        ("nu0=0.5", 0.5, 0.75, 1.0, 0.5, 0.10),
        ("nu0=1.0", 1.0, 0.75, 1.0, 0.5, 0.10),
        ("p=0.60", 0.5, 0.60, 1.0, 0.5, 0.10),
        ("p=0.90", 0.5, 0.90, 1.0, 0.5, 0.10),
        ("alpha_min=0.3", 0.5, 0.75, 1.0, 0.5, 0.30),
    ]

    rows = []
    for label, nu0, power, delta0, delta_power, alpha_min in configs:
        options = DualOptions(
            max_iter=3500,
            nu0=nu0,
            step_power=power,
            delta0=delta0,
            delta_power=delta_power,
            alpha_min=alpha_min,
            log_every=100,
        )
        result = DeflectedSubgradientSolver(options).solve(problem)
        polish = frank_wolfe_polish(problem, result.x, max_iter=800, tol_gap=5e-7, log_every=100)
        rel = (polish.value - a2.value) / max(1.0, abs(a2.value))
        rows.append(
            {
                "setting": label,
                "nu0": nu0,
                "step_power": power,
                "delta0": delta0,
                "delta_power": delta_power,
                "alpha_min": alpha_min,
                "a1_value": polish.value,
                "a1_dual_bound": result.dual_value,
                "a1_rel_gap": (polish.value - result.dual_value) / max(1.0, abs(polish.value)),
                "rel_to_a2": rel,
                "dual_iter": result.iterations,
                "polish_iter": polish.iterations,
                "time": result.elapsed + polish.elapsed,
                "kkt": kkt_residual(problem, polish.x)["max"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "parameter_sensitivity.csv", index=False)

    table = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Setting & $\nu_0$ & $p$ & $\alpha_{\min}$ & rel. diff. vs A2 & cert. gap & KKT \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        table.append(
            latex_row(
                [
                    escape(row["setting"]),
                    f"{row['nu0']:.1f}",
                    f"{row['step_power']:.2f}",
                    f"{row['alpha_min']:.2f}",
                    fmt_e(row["rel_to_a2"]),
                    fmt_e(row["a1_rel_gap"]),
                    fmt_e(row["kkt"]),
                ]
            )
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (REPORT / "parameter_table.tex").write_text("\n".join(table))

    nu_df = df[df["setting"].str.startswith("nu0")]
    plt.figure(figsize=(6, 3.5))
    plt.plot(nu_df["nu0"], nu_df["a1_rel_gap"], marker="o")
    plt.xlabel(r"initial stepsize $\nu_0$")
    plt.ylabel("final relative certificate")
    plt.tight_layout()
    plt.savefig(FIGURES / "parameter_nu0.pdf")
    plt.close()


def run_psd_factorization_check() -> None:
    original = generate_instance(4, 5, rank_dim=8, density=0.7, seed=123, name="psd_check")
    reconstructed = SimplexQP.from_Q(original.Q, original.q, original.blocks, name="psd_check_from_Q")
    rng = np.random.default_rng(7)
    max_err = 0.0
    for _ in range(20):
        x = np.zeros(original.n)
        for block in original.blocks:
            weights = rng.random(len(block))
            x[block] = weights / weights.sum()
        original_value = float(x @ original.Q @ x + original.q @ x)
        reconstructed_value = float(x @ reconstructed.Q @ x + reconstructed.q @ x)
        max_err = max(max_err, abs(original_value - reconstructed_value))

    (RESULTS / "psd_factorization_check.txt").write_text(
        f"rank(original R)={original.rank_dim}\n"
        f"rank(reconstructed R)={reconstructed.rank_dim}\n"
        f"max objective discrepancy on feasible samples={max_err:.3e}\n"
    )


def main() -> None:
    make_main_tables()
    run_convergence_asset()
    run_parameter_sensitivity()
    run_psd_factorization_check()
    print("Report assets regenerated.")


if __name__ == "__main__":
    main()
