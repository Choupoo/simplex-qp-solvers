# Product-Simplex Convex QP Solvers

This repository contains the implementation and experimental code for **Non-ML Project 5**.  The project studies convex quadratic programs over a Cartesian product of simplices and compares two solution approaches:

- **A1**: a solver-free Lagrangian dual decomposition method using a target-value deflected subgradient update and a Frank-Wolfe polishing phase.
- **A2**: a direct general-purpose constrained solve using SciPy's SLSQP implementation, run both from a cold start and from the A1 solution as a warm start.

The goal is not only to compute good primal solutions, but also to compare the methods in terms of objective value, feasibility, KKT residuals, timing, and primal-dual certification.

## Problem formulation

The optimization problem is

```math
\min_{x \in K} \; f(x) = x^T Q x + q^T x, \qquad Q \succeq 0,
```

where the feasible region is a product of simplex blocks:

```math
K = \left\{x \in \mathbb{R}^n : \sum_{i \in I_k} x_i = 1,\ k=1,\ldots,m,\ x_i \ge 0\right\}.
```

Since `Q` is positive semidefinite, the objective is convex.  The implementation primarily works with a factorization

```math
Q = R^T R,
```

so that

```math
f(x) = \|Rx\|_2^2 + q^T x.
```

Instances can either be generated directly from `R`, or constructed from a native positive-semidefinite matrix `Q` via `SimplexQP.from_Q(...)`.

## Algorithms

### A1: target-value deflected dual subgradient method

A1 introduces a linking variable `y = Rx` and dualizes the linking equation.  For a multiplier `lambda`, the Lagrangian relaxation separates into:

- a closed-form minimization over `y`;
- a linear minimization over each simplex block.

The resulting dual function is

```math
\psi(\lambda)
= -\frac{1}{4}\|\lambda\|^2
+ \sum_{k=1}^m \min_{i \in I_k}\{q_i + (R^T\lambda)_i\}.
```

The implemented method minimizes `Phi(lambda) = -psi(lambda)` using a projected target-value deflected subgradient update:

```math
\nu_k = \frac{\nu_0}{k^p},
```

```math
\alpha_k = \max\left\{\alpha_{\min},
\frac{\nu_{k-1}\|d_{k-1}\|^2}
{\Phi_k - \Phi_k^{lev} + \nu_{k-1}\|d_{k-1}\|^2}\right\},
```

```math
d_k = \alpha_k g_k + (1-\alpha_k)d_{k-1},
```

```math
\lambda_{k+1} = \mathrm{proj}_{B_\rho}(\lambda_k - \nu_k d_k),
\qquad
\rho = 2\|R\|_F\sqrt{m}.
```

A1 keeps a feasible primal incumbent from Lagrangian vertices and their running averages.  It also reports a valid primal-dual certificate based on the best primal upper bound and the record dual lower bound.  A conditional-gradient / Frank-Wolfe polishing phase is then used to improve the feasible primal point.

### A2: SciPy SLSQP solver

A2 solves the original constrained convex QP directly:

```math
\min_x \; x^TQx + q^Tx
\quad\text{s.t.}\quad
Ax = \mathbf{1},\ x \ge 0,
```

where `A` is the block-incidence matrix.  The code supplies the exact objective, gradient, equality constraints, and nonnegativity bounds to SciPy SLSQP.

Two configurations are compared:

- **A2-cold**: initialized at the uniform feasible point.
- **A2-warm**: initialized at the polished A1 solution.

The warm-start time is only the SLSQP refinement time.  The end-to-end warm-start pipeline time is therefore

```math
t_{pipeline} = t_{A1} + t_{A2,warm}.
```

## Repository structure

```text
code/
├── qpsimplex/
│   ├── __init__.py
│   ├── problem.py             # Problem model, PSD factorization, instance generation, save/load
│   ├── dual_deflected.py      # A1 target-value deflected dual method
│   ├── primal_recovery.py     # Frank-Wolfe polishing phase
│   ├── a2_scipy.py            # A2 SciPy SLSQP solver
│   ├── metrics.py             # Objective, feasibility residual, KKT residual, relative gap
│   └── experiments.py         # Reproducible A1/A2 experiment grid
├── scripts/
│   ├── run_one.py             # Solve one generated or saved instance
│   ├── run_experiments.py     # Reproduce the A1/A2 comparison grid
│   ├── make_report_assets.py  # Regenerate LaTeX tables and PDF figures
│   └── run_self_check.py      # Lightweight correctness and consistency checks
├── results/                   # Generated CSV files and saved instances
└── report_assets/             # Generated LaTeX tables and PDF figures
```

The report PDF can be stored at the repository root, for example as `Non-ML Project5.pdf` or `report.pdf`.

## Requirements

The code is written in Python and uses:

- Python 3.11 recommended;
- NumPy;
- SciPy;
- pandas;
- matplotlib.

A Conda environment can be created with:

```bash
conda create -n nonml5 python=3.11 -y
conda activate nonml5
conda install -c conda-forge numpy scipy pandas matplotlib -y
```

Alternatively, using `venv` and `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib
```

The scripts are intended to be executed from the `code/` directory.

## Quick start

Move into the code directory:

```bash
cd code
```

Run the self-check:

```bash
python scripts/run_self_check.py
```

Expected output includes a line similar to:

```text
self-check passed: theorem-rule invariants and numerical bounds verified
```

Solve one generated instance:

```bash
python scripts/run_one.py \
  --blocks 8 \
  --block-size 15 \
  --rank 40 \
  --density 0.5 \
  --seed 0
```

This runs:

1. A1 target-value deflected subgradient method;
2. Frank-Wolfe polishing;
3. A2-cold from the uniform feasible point;
4. A2-warm from the polished A1 point.

The script prints objective values, dual bound, relative gap, iteration counts, timing, and KKT residuals.

## Reproducing the numerical experiments

From the project root, run:

```bash
cd code
python scripts/run_self_check.py
python scripts/run_experiments.py
python scripts/make_report_assets.py
cd ..
```

The experiment script regenerates saved instances and summary CSV files under:

```text
code/results/
```

The report-asset script regenerates LaTeX tables and PDF figures under:

```text
code/report_assets/
```

Typical generated files include:

```text
code/results/summary.csv
code/results/convergence_history.csv
code/results/parameter_sensitivity.csv
code/results/psd_factorization_check.txt
code/report_assets/results_table.tex
code/report_assets/aggregate_table.tex
code/report_assets/parameter_table.tex
code/report_assets/figures/convergence_bounds.pdf
code/report_assets/figures/convergence_gap.pdf
code/report_assets/figures/deflection_steps.pdf
code/report_assets/figures/time_scaling.pdf
code/report_assets/figures/relative_difference.pdf
```

## Using the package in Python

### Generate and solve an instance

```python
from qpsimplex.problem import generate_instance
from qpsimplex.dual_deflected import DeflectedSubgradientSolver, DualOptions
from qpsimplex.primal_recovery import frank_wolfe_polish
from qpsimplex.a2_scipy import solve_with_scipy_slsqp
from qpsimplex.metrics import kkt_residual

problem = generate_instance(
    n_blocks=8,
    block_size=15,
    rank_dim=40,
    density=0.5,
    seed=0,
    name="demo_instance",
)

options = DualOptions(
    max_iter=5000,
    tol_gap=1e-5,
    nu0=0.5,
    step_power=0.75,
    delta0=1.0,
    delta_power=0.5,
    alpha_min=0.1,
)

# A1: dual method
solver = DeflectedSubgradientSolver(options)
a1 = solver.solve(problem)

# Optional primal polishing
polish = frank_wolfe_polish(problem, a1.x, max_iter=1500, tol_gap=1e-7)

# A2 cold and warm starts
a2_cold = solve_with_scipy_slsqp(problem, x0=None, max_iter=800, ftol=1e-10)
a2_warm = solve_with_scipy_slsqp(problem, x0=polish.x, max_iter=800, ftol=1e-10)

print("A1 polished value:", polish.value)
print("A1 dual bound:", a1.dual_value)
print("A1 KKT residual:", kkt_residual(problem, polish.x)["max"])
print("A2 cold value:", a2_cold.value)
print("A2 warm value:", a2_warm.value)
```

### Use a native PSD matrix `Q`

```python
import numpy as np

from qpsimplex.problem import SimplexQP
from qpsimplex.dual_deflected import DeflectedSubgradientSolver, DualOptions

# Example data
Q = np.eye(6)
q = np.random.default_rng(0).normal(size=6)
blocks = [np.array([0, 1, 2]), np.array([3, 4, 5])]

problem = SimplexQP.from_Q(Q, q, blocks, name="native_Q_instance")
options = DualOptions(max_iter=5000, nu0=0.5, step_power=0.75, alpha_min=0.1)
result = DeflectedSubgradientSolver(options).solve(problem)

print(result.primal_value, result.dual_value, result.relative_gap)
```

## Experiment grid

The default experiments use four representative instance families:

| Family | Blocks | Block size | Rank dimension | Density |
|---|---:|---:|---:|---:|
| `small_dense` | 5 | 10 | 20 | 1.00 |
| `medium_dense` | 10 | 10 | 35 | 1.00 |
| `medium_sparse` | 15 | 10 | 45 | 0.20 |
| `large_sparse` | 20 | 8 | 50 | 0.12 |

By default, each family is tested on seeds `0` and `1`.

## Metrics

The reported metrics include:

- primal objective value;
- A1 dual lower bound;
- absolute and relative primal-dual gap;
- number of A1 dual iterations;
- number of Frank-Wolfe polishing iterations;
- A2 SLSQP iterations;
- wall-clock time;
- feasibility residual;
- blockwise KKT residual;
- A2 warm-start speedup;
- end-to-end pipeline time `A1 + A2-warm`.

The relative primal-dual certificate is computed as

```math
\frac{UB - LB}{\max\{1, |UB|\}}.
```

## Main empirical conclusion

On the tested instance families, A2-cold is the fastest standalone method in wall-clock time.  Warm-starting SLSQP from the A1 solution reduces the SLSQP refinement time, but the full pipeline `A1 + A2-warm` is slower than directly running A2-cold.

The main advantage of A1 is therefore not speed on this experiment grid.  Its advantage is that it is solver-free and returns certified primal-dual information: a feasible primal point, a valid dual lower bound, and a relative primal-dual optimality certificate.  A2-warm is useful when an A1 solution has already been computed and an additional primal refinement is desired.

These conclusions are implementation-level observations for the current code, hardware, solver settings, stopping rules, and generated instances.  They should not be interpreted as universal complexity claims.

## Self-checks

The self-check script verifies several important properties:

- PSD factorization from `Q` preserves the objective value on a feasible point;
- multiplier projection respects the explicit radius;
- A1 returns a valid status;
- A1 lower bounds do not exceed feasible primal values beyond tolerance;
- deflection parameters satisfy the expected inequalities;
- A2-cold and A2-warm reach consistent objective values on a small test instance.

Run it with:

```bash
cd code
python scripts/run_self_check.py
```

## References

- S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge University Press, 2004.
- G. d'Antonio and A. Frangioni, "Convergence Analysis of Deflected Conditional Approximate Subgradient Methods," *SIAM Journal on Optimization*, 20(1), 357-386, 2009.
- M. Frank and P. Wolfe, "An Algorithm for Quadratic Programming," *Naval Research Logistics Quarterly*, 3(1-2), 95-110, 1956.
- D. Kraft, *A Software Package for Sequential Quadratic Programming*, Technical Report DFVLR-FB 88-28, 1988.
- P. Virtanen et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, 17, 261-272, 2020.

## License

No license file is included by default.  Add a license before making the repository public if you want to specify how others may use or redistribute the code.
