from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "quantlab")

DIRS = [
    "src/quantlab",
    "tests",
    "notebooks",
    "data/raw",
    "data/processed",
    "report",
]

FILES: dict[str, str] = {
    "pyproject.toml": '''\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "quantlab"
version = "0.1.0"
description = "Real returns, noisy estimates and portfolio construction"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "yfinance",
]

[project.optional-dependencies]
dev = ["pytest", "jupyterlab"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
''',
    ".gitignore": '''\
# environments
.venv/
venv/

# python
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/

# notebooks
.ipynb_checkpoints/

# data: never commit downloaded prices
data/

# editors / OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# LaTeX build junk
report/*.aux
report/*.log
report/*.out
report/*.synctex.gz
''',
    "README.md": '''\
# QuantLab

**Question:** how non-Normal are real returns, and does mean-variance
optimisation survive estimation error?

## How to reproduce

```
pip install -e ".[dev]"
pytest
jupyter lab
```

## Conventions

- Prices: adjusted close (`auto_adjust=True`)
- Returns: log for statistics, simple for portfolio arithmetic
- Annualising: mean x 252, volatility x sqrt(252)
- No look-ahead: anything used at date t is knowable at t-1
- Random functions take a `seed` and use `np.random.default_rng(seed)`

## Data

Download date: (fill in)
Universe: (fill in)
Excluded tickers: (fill in)

## Results

(fill in)

## Limitations

(fill in: survivorship bias, one market, one period, ...)
''',
    "src/quantlab/__init__.py": '''\
"""QuantLab: data pipeline, returns, estimators, optimisers and backtests."""

__version__ = "0.1.0"
''',
    "tests/test_smoke.py": '''\
"""Checks that the environment and the package are wired up correctly."""
import numpy as np
import pandas as pd

import quantlab


def test_package_imports():
    assert quantlab.__version__ == "0.1.0"


def test_parquet_roundtrip(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"A": np.arange(5.0)}, index=idx)
    path = tmp_path / "x.parquet"
    df.to_parquet(path)
    out = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, out, check_freq=False)
''',
}


def main() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for rel, content in FILES.items():
        path = ROOT / rel
        if path.exists():
            print(f"skip   {path} (already exists)")
            continue
        path.write_text(content, encoding="utf-8")
        print(f"create {path}")
    print(f"\nDone. Next: cd {ROOT} and follow the install steps.")


if __name__ == "__main__":
    main()
