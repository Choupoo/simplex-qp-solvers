from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


def factor_psd_matrix(Q: np.ndarray, tol: float = 1e-10, relative: bool = True) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if not np.allclose(Q, Q.T, atol=10 * tol, rtol=10 * tol):
        raise ValueError("Q must be symmetric within tolerance")
    # Symmetrize explicitly to remove harmless roundoff asymmetries.
    Qs = 0.5 * (Q + Q.T)
    eigvals, eigvecs = np.linalg.eigh(Qs)
    scale = max(1.0, float(np.max(np.abs(eigvals))) if eigvals.size else 1.0)
    eps = tol * scale if relative else tol
    if np.min(eigvals) < -eps:
        raise ValueError(f"Q is not positive semidefinite: min eigenvalue {np.min(eigvals):.3e}")
    positive = eigvals > eps
    if not np.any(positive):
        return np.zeros((0, Q.shape[0]))
    # If Q = V diag(w) V.T, then R = diag(sqrt(w)) V.T.
    return (np.sqrt(eigvals[positive])[:, None] * eigvecs[:, positive].T)


@dataclass(frozen=True)
class SimplexQP:
    R: np.ndarray
    q: np.ndarray
    blocks: List[np.ndarray]
    name: str = "instance"

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float)
        q = np.asarray(self.q, dtype=float).reshape(-1)
        if R.ndim != 2:
            raise ValueError("R must be a 2D array")
        if R.shape[1] != q.size:
            raise ValueError("R has incompatible number of columns")
        if not self.blocks:
            raise ValueError("at least one simplex block is required")
        covered = np.concatenate([np.asarray(b, dtype=int).reshape(-1) for b in self.blocks])
        if covered.size != q.size:
            raise ValueError("blocks do not cover exactly n indices")
        if np.unique(covered).size != q.size or covered.min() != 0 or covered.max() != q.size - 1:
            raise ValueError("blocks must form a partition of {0, ..., n-1}")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "blocks", [np.asarray(b, dtype=int).reshape(-1) for b in self.blocks])

    @classmethod
    def from_Q(
        cls,
        Q: np.ndarray,
        q: np.ndarray,
        blocks: List[np.ndarray],
        name: str = "instance_from_Q",
        tol: float = 1e-10,
    ) -> "SimplexQP":
        R = factor_psd_matrix(Q, tol=tol)
        return cls(R=R, q=np.asarray(q, dtype=float), blocks=blocks, name=name)

    @property
    def n(self) -> int:
        return int(self.q.size)

    @property
    def rank_dim(self) -> int:
        return int(self.R.shape[0])

    @property
    def m_blocks(self) -> int:
        return len(self.blocks)

    @property
    def Q(self) -> np.ndarray:
        return self.R.T @ self.R

    def uniform_feasible_point(self) -> np.ndarray:
        x = np.zeros(self.n)
        for block in self.blocks:
            x[block] = 1.0 / len(block)
        return x

    def block_sizes(self) -> List[int]:
        return [int(len(b)) for b in self.blocks]


def make_equal_blocks(n_blocks: int, block_size: int) -> List[np.ndarray]:
    blocks: List[np.ndarray] = []
    start = 0
    for _ in range(n_blocks):
        blocks.append(np.arange(start, start + block_size, dtype=int))
        start += block_size
    return blocks


def generate_instance(
    n_blocks: int,
    block_size: int,
    rank_dim: int | None = None,
    density: float = 1.0,
    seed: int = 0,
    q_scale: float = 1.0,
    r_scale: float = 1.0,
    name: str | None = None,
) -> SimplexQP:
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0, 1]")
    rng = np.random.default_rng(seed)
    n = n_blocks * block_size
    if rank_dim is None:
        rank_dim = min(n, max(5, n // 3))
    R = rng.normal(size=(rank_dim, n))
    if density < 1.0:
        mask = rng.random(size=R.shape) <= density
        R *= mask
    # Scale so that ||R x||^2 is numerically comparable across dimensions.
    denom = np.sqrt(max(1, rank_dim * density))
    R = r_scale * R / denom
    q = q_scale * rng.normal(size=n)
    blocks = make_equal_blocks(n_blocks, block_size)
    if name is None:
        name = f"K{n_blocks}_s{block_size}_r{rank_dim}_d{density:g}_seed{seed}"
    return SimplexQP(R=R, q=q, blocks=blocks, name=name)


def save_instance(problem: SimplexQP, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store blocks as an object array so variable block sizes are supported.
    np.savez_compressed(
        path,
        R=problem.R,
        q=problem.q,
        blocks=np.array(problem.blocks, dtype=object),
        name=np.array(problem.name),
    )


def load_instance(path: str | Path) -> SimplexQP:
    data = np.load(path, allow_pickle=True)
    blocks = [np.asarray(b, dtype=int) for b in data["blocks"]]
    name = str(data["name"]) if "name" in data else Path(path).stem
    return SimplexQP(R=data["R"], q=data["q"], blocks=blocks, name=name)
