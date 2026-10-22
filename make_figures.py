"""Create the four figures used in report/writeup.tex (saved in report/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from quantlab.backtest import (
    equal_weight, make_shrunk_mv, minvar_lw, minvar_lw_long_only, minvar_sample, walk_forward,
)
from quantlab.covariance import mp_density, mp_edges
from quantlab.data_loader import load_prices
from quantlab.diagnostics import fit_t
from quantlab.metrics import drawdown, wealth
from quantlab.returns import log_returns, simple_returns

SAVE = dict(dpi=200, bbox_inches="tight")


def fig_qq(r: pd.DataFrame, out: Path) -> None:
    x = r["^GSPC"]
    df_ = fit_t(r[["^GSPC"]]).loc["^GSPC", "df"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    stats.probplot(x, dist="norm", plot=ax[0])
    ax[0].set_title("S&P 500 vs Normal")
    stats.probplot(x, dist=stats.t, sparams=(df_,), plot=ax[1])
    ax[1].set_title(f"S&P 500 vs Student-t (df = {df_:.1f})")
    fig.tight_layout()
    fig.savefig(out / "fig1_qq.png", **SAVE)
    plt.close(fig)


def fig_clustering(r: pd.DataFrame, out: Path) -> None:
    x = r["^GSPC"]
    vol = x.rolling(21).std() * np.sqrt(252)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(vol, linewidth=0.8)
    ax[0].set_title("S&P 500 rolling 21-day volatility (annualised)")
    plot_acf(x.abs(), lags=40, ax=ax[1], title="ACF of |r|")
    fig.tight_layout()
    fig.savefig(out / "fig2_clustering.png", **SAVE)
    plt.close(fig)


def fig_mp(stocks: pd.DataFrame, out: Path) -> None:
    n, p = stocks.shape
    eig = np.linalg.eigvalsh(np.corrcoef(stocks.to_numpy(), rowvar=False))
    lo, hi = mp_edges(n, p)
    grid = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(eig, bins=40, density=True, alpha=0.6, label="sample eigenvalues")
    ax.plot(grid, mp_density(grid, n, p), "r", label="Marchenko-Pastur (noise)")
    ax.axvline(hi, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 3)
    ax.set_title(f"Correlation eigenvalues, n = {n}, p = {p} (largest {eig.max():.1f} off-scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_mp.png", **SAVE)
    plt.close(fig)


def fig_backtest(rets: pd.DataFrame, out: Path) -> None:
    strategies = {
        "1/N": equal_weight,
        "MinVar sample": minvar_sample,
        "MinVar LW": minvar_lw,
        "MinVar LW long-only": minvar_lw_long_only,
        "Shrunk MV": make_shrunk_mv(k=0.5),
    }
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for name, fn in strategies.items():
        res = walk_forward(rets, fn, window=504, rebalance=21, cost=0.001)
        ax[0].plot(wealth(res["ret"]), label=name, linewidth=1)
        ax[1].plot(drawdown(res["ret"]), label=name, linewidth=1)
    ax[0].set_yscale("log")
    ax[0].set_title("Growth of 1, net of 10 bps costs (log scale)")
    ax[1].set_title("Drawdown")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_backtest.png", **SAVE)
    plt.close(fig)


def build(prices: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = log_returns(prices)
    stocks = r.drop(columns="^GSPC")
    rets = simple_returns(prices).drop(columns="^GSPC")
    fig_qq(r, out)
    fig_clustering(r, out)
    fig_mp(stocks, out)
    fig_backtest(rets, out)
    print("saved:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    build(load_prices(), Path("report/figures"))
