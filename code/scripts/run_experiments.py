#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpsimplex.experiments import run_grid

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    csv_path = run_grid(root / "results")
    print(f"Wrote {csv_path}")
