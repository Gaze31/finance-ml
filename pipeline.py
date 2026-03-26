"""
Auto-sklearn Quantitative Finance Pipeline
==========================================
Walk-forward return prediction using automated machine learning.

Pipeline:
  OHLCV data → Feature engineering (lagged) → Walk-forward CV
  → AutoSklearnClassifier → Ensemble → Signal → Long/short portfolio → PnL

Author: Quant Research
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (replaces live data feed for demo)
# In production: swap this with yfinance, Bloomberg, or your data vendor
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_ohlcv(
    n_days: int = 1500,
    n_assets: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic OHLCV data with realistic stylized facts:
    - Fat-tailed returns (Student-t)
    - Volatility clustering (GARCH-like)
    - Momentum and mean-reversion regimes
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]

    records = []
    for ticker in tickers:
        price = 100.0
        vol = 0.015
        prices = []

        for _ in range(n_days):
            # GARCH-like vol updating
            shock = rng.standard_t(df=5) * vol
            vol = np.clip(0.85 * vol + 0.15 * abs(shock) + 0.001, 0.005, 0.06)
            price *= (1 + shock)
            price = max(price, 1.0)
            prices.append(price)

        prices = np.array(prices)
        highs  = prices * (1 + rng.uniform(0, 0.01, n_days))
        lows   = prices * (1 - rng.uniform(0, 0.01, n_days))
        opens  = prices * (1 + rng.normal(0, 0.003, n_days))
        vols   = rng.lognormal(mean=14, sigma=0.4, size=n_days).astype(int)

        df = pd.DataFrame({
            "date":   dates,
            "ticker": ticker,
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  prices,
            "volume": vols,
        })
        records.append(df)

    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# All features are lagged by 1 period — strict no-lookahead guarantee
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs cross-sectional factor features per asset.
    All features are shifted(1) to prevent lookahead bias.

    Features:
      mom_5, mom_20, mom_60  : Price momentum over multiple horizons
      rvol_10, rvol_20       : Realized volatility (rolling std of log-returns)
      rsi_14                 : RSI oscillator
      vol_z                  : Volume z-score vs 20d mean
      hl_range               : High-low range as fraction of close (intraday vol proxy)
      target                 : Binary — 1 if next-day return > 0
    """
    out = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        ret = g["close"].pct_change()
        lret = np.log(g["close"]).diff()

        f = pd.DataFrame(index=g.index)
        f["date"]    = g["date"]
        f["ticker"]  = ticker

        # Momentum features
        f["mom_5"]   = g["close"].pct_change(5).shift(1)
        f["mom_20"]  = g["close"].pct_change(20).shift(1)
        f["mom_60"]  = g["close"].pct_change(60).shift(1)

        # Volatility features
        f["rvol_10"] = lret.rolling(10).std().shift(1)
        f["rvol_20"] = lret.rolling(20).std().shift(1)

        # RSI
        f["rsi_14"]  = compute_rsi(g["close"], 14).shift(1)

        # Volume z-score
        vol_mean     = g["volume"].rolling(20).mean()
        vol_std      = g["volume"].rolling(20).std()
        f["vol_z"]   = ((g["volume"] - vol_mean) / (vol_std + 1e-10)).shift(1)

        # Intraday range proxy
        f["hl_range"] = ((g["high"] - g["low"]) / (g["close"] + 1e-10)).shift(1)

        # Target: 1 if tomorrow's return > 0
        f["target"]   = (ret > 0).astype(int)
        f["fwd_ret"]  = ret  # kept for PnL calculation

        out.append(f)

    features = pd.concat(out).dropna()
    return features.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# Expanding window with configurable burn-in and test period
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "mom_5", "mom_20", "mom_60",
    "rvol_10", "rvol_20",
    "rsi_14", "vol_z", "hl_range"
]


def get_expanding_date_splits(
    dates: pd.Series,
    burn_in_days: int = 252,
    test_days: int = 21
):
    """
    Yields (train_dates, test_dates) pairs for expanding window CV.
    Each fold trains on all data up to a cutoff, tests on the next window.
    """
    unique_dates = sorted(dates.unique())
    n = len(unique_dates)

    if n <= burn_in_days:
        raise ValueError(f"Need > {burn_in_days} unique dates, got {n}")

    start = burn_in_days
    while start < n:
        end = min(start + test_days, n)
        train_dates = unique_dates[:start]
        test_dates  = unique_dates[start:end]
        yield train_dates, test_dates
        start += test_days


def run_walk_forward(
    features: pd.DataFrame,
    burn_in_days: int = 252,
    test_days: int = 21,
    use_autosklearn: bool = True,
    automl_time_secs: int = 120,
) -> pd.DataFrame:
    """
    Runs the walk-forward backtesting loop.

    Parameters
    ----------
    features       : DataFrame from build_features()
    burn_in_days   : Minimum training periods before first prediction
    test_days      : Number of periods per test fold
    use_autosklearn: If True, uses AutoSklearn; falls back to sklearn GBM if False
    automl_time_secs: Budget per fold for AutoSklearn search

    Returns
    -------
    DataFrame with columns: date, ticker, signal, actual, fwd_ret
    """
    from sklearn.preprocessing import StandardScaler

    results = []
    splits  = list(get_expanding_date_splits(features["date"], burn_in_days, test_days))
    n_folds = len(splits)

    print(f"\n{'─'*60}")
    print(f"  Walk-Forward Backtest")
    print(f"  Total folds : {n_folds}")
    print(f"  Burn-in     : {burn_in_days} days")
    print(f"  Test window : {test_days} days per fold")
    print(f"  Engine      : {'AutoSklearn' if use_autosklearn else 'GradientBoosting (fallback)'}")
    print(f"{'─'*60}\n")

    for fold_idx, (train_dates, test_dates) in enumerate(splits):
        train_mask = features["date"].isin(train_dates)
        test_mask  = features["date"].isin(test_dates)

        X_tr = features.loc[train_mask, FEATURE_COLS]
        y_tr = features.loc[train_mask, "target"]
        X_te = features.loc[test_mask,  FEATURE_COLS]
        y_te = features.loc[test_mask,  "target"]

        # Fit scaler on train only — no data leakage
        scaler   = StandardScaler().fit(X_tr)
        X_tr_sc  = scaler.transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        if use_autosklearn:
            try:
                import autosklearn.classification
                import autosklearn.metrics

                clf = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=automl_time_secs,
                    per_run_time_limit=automl_time_secs // 5,
                    ensemble_size=50,
                    metric=autosklearn.metrics.roc_auc,
                    memory_limit=4096,
                    n_jobs=-1,
                )
                clf.fit(X_tr_sc, y_tr)
                proba = clf.predict_proba(X_te_sc)[:, 1]

            except ImportError:
                print("  [WARN] auto-sklearn not installed — using GBM fallback")
                proba = _fit_gbm(X_tr_sc, y_tr, X_te_sc)
        else:
            proba = _fit_gbm(X_tr_sc, y_tr, X_te_sc)

        fold_result = features.loc[test_mask, ["date", "ticker", "fwd_ret"]].copy()
        fold_result["signal"] = proba
        fold_result["actual"] = y_te.values
        results.append(fold_result)

        # Fold summary
        ic = spearmanr(proba, y_te.values).correlation
        print(f"  Fold {fold_idx+1:>3}/{n_folds}  |  "
              f"Train: {len(train_dates):>4}d  |  "
              f"Test: {len(X_te):>5} obs  |  "
              f"IC: {ic:+.4f}")

    return pd.concat(results).reset_index(drop=True)


def _fit_gbm(X_tr, y_tr, X_te) -> np.ndarray:
    """Fallback when auto-sklearn is not available."""
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# Long top decile, short bottom decile — daily rebalancing
# ─────────────────────────────────────────────────────────────────────────────

def construct_portfolio(
    predictions: pd.DataFrame,
    long_quantile: float = 0.9,
    short_quantile: float = 0.1,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Builds a dollar-neutral long/short portfolio from model signals.

    For each date:
      - Long  top    (1 - long_quantile)  fraction of assets by signal
      - Short bottom (short_quantile)     fraction of assets by signal
      - Equal weight within each leg
      - Deduct round-trip transaction costs on full turnover (conservative)

    Returns daily portfolio returns.
    """
    tc = transaction_cost_bps / 10_000

    daily_returns = []
    for date, group in predictions.groupby("date"):
        if len(group) < 10:
            continue

        q_hi = group["signal"].quantile(long_quantile)
        q_lo = group["signal"].quantile(short_quantile)

        longs  = group[group["signal"] >= q_hi]
        shorts = group[group["signal"] <= q_lo]

        if longs.empty or shorts.empty:
            continue

        long_ret  = longs["fwd_ret"].mean()
        short_ret = shorts["fwd_ret"].mean()

        # Long/short return minus TC (both legs)
        portfolio_ret = 0.5 * long_ret - 0.5 * short_ret - tc

        daily_returns.append({
            "date":          date,
            "portfolio_ret": portfolio_ret,
            "long_ret":      long_ret,
            "short_ret":     short_ret,
            "n_longs":       len(longs),
            "n_shorts":      len(shorts),
        })

    return pd.DataFrame(daily_returns).sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance(pnl: pd.DataFrame, rf_annual: float = 0.05) -> dict:
    """
    Full performance report for the strategy.

    Metrics: CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Hit Rate, IC
    """
    rets = pnl["portfolio_ret"]
    cum  = (1 + rets).cumprod()

    # Annualization factor (252 trading days)
    ann = 252
    rf_daily = (1 + rf_annual) ** (1/ann) - 1

    total_days   = len(rets)
    total_return = cum.iloc[-1] - 1
    cagr         = (1 + total_return) ** (ann / total_days) - 1

    excess       = rets - rf_daily
    sharpe       = excess.mean() / (excess.std() + 1e-10) * np.sqrt(ann)

    downside     = rets[rets < rf_daily]
    sortino      = excess.mean() / (downside.std() + 1e-10) * np.sqrt(ann)

    running_max  = cum.cummax()
    drawdown     = cum / running_max - 1
    max_dd       = drawdown.min()

    calmar       = cagr / (abs(max_dd) + 1e-10)
    hit_rate     = (rets > 0).mean()

    return {
        "Total return (%)":  round(total_return * 100, 2),
        "CAGR (%)":          round(cagr * 100, 2),
        "Sharpe ratio":      round(sharpe, 3),
        "Sortino ratio":     round(sortino, 3),
        "Max drawdown (%)":  round(max_dd * 100, 2),
        "Calmar ratio":      round(calmar, 3),
        "Hit rate (%)":      round(hit_rate * 100, 2),
        "Trading days":      total_days,
    }


def compute_ic_series(predictions: pd.DataFrame) -> pd.DataFrame:
    """Computes daily Information Coefficient (Spearman rank correlation)."""
    ic_rows = []
    for date, group in predictions.groupby("date"):
        if len(group) < 5:
            continue
        ic = spearmanr(group["signal"], group["actual"]).correlation
        ic_rows.append({"date": date, "ic": ic})
    ic_df = pd.DataFrame(ic_rows)
    ic_df["ic_ma"] = ic_df["ic"].rolling(21).mean()
    return ic_df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(pnl: pd.DataFrame, ic_df: pd.DataFrame, perf: dict):
    """Generates a 4-panel performance dashboard."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mtick
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots")
        return

    cum_ret  = (1 + pnl["portfolio_ret"]).cumprod() - 1
    drawdown = (1 + pnl["portfolio_ret"]).cumprod()
    drawdown = drawdown / drawdown.cummax() - 1

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0D0D0F")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT  = "#7F77DD"
    RED     = "#D85A30"
    GREEN   = "#1D9E75"
    MUTED   = "#888780"
    BG      = "#0D0D0F"
    PANEL   = "#16161A"
    TEXT    = "#E8E6E0"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="500", pad=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color("#2C2C2A")
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # ── Panel 1: Cumulative PnL ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(pnl["date"], cum_ret * 100, color=ACCENT, linewidth=1.2, label="Strategy")
    ax1.axhline(0, color=MUTED, linewidth=0.4, linestyle="--")
    ax1.fill_between(pnl["date"], cum_ret * 100, 0,
                     where=(cum_ret >= 0), alpha=0.12, color=GREEN)
    ax1.fill_between(pnl["date"], cum_ret * 100, 0,
                     where=(cum_ret < 0),  alpha=0.12, color=RED)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    style_ax(ax1, "Cumulative PnL")

    # ── Panel 2: Drawdown ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(pnl["date"], drawdown * 100, 0, color=RED, alpha=0.6)
    ax2.plot(pnl["date"], drawdown * 100, color=RED, linewidth=0.8)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    style_ax(ax2, "Drawdown")

    # ── Panel 3: IC Series ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(ic_df["date"], ic_df["ic"], color=MUTED, alpha=0.4, width=1.5, label="Daily IC")
    ax3.plot(ic_df["date"], ic_df["ic_ma"], color=ACCENT, linewidth=1.2, label="21d MA")
    ax3.axhline(0, color=MUTED, linewidth=0.4, linestyle="--")
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.5)
    style_ax(ax3, "Information Coefficient")

    # ── Panel 4: Performance Table ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    ax4.axis("off")
    style_ax(ax4, "Performance Summary")

    rows = list(perf.items())
    n    = len(rows)
    for i, (label, val) in enumerate(rows):
        y = 0.88 - i * (0.88 / n)
        color = GREEN if isinstance(val, float) and val > 0 and "drawdown" not in label.lower() \
                else (RED if isinstance(val, float) and val < 0 else TEXT)
        ax4.text(0.05, y, label,      transform=ax4.transAxes,
                 color=MUTED, fontsize=9)
        ax4.text(0.75, y, str(val),   transform=ax4.transAxes,
                 color=color, fontsize=9, fontweight="500", ha="right")

    fig.suptitle(
        "Auto-sklearn · Quantitative Finance Pipeline",
        color=TEXT, fontsize=13, fontweight="500", y=0.98
    )

    plt.savefig("strategy_report.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print("\n  Chart saved → strategy_report.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  AUTO-SKLEARN QUANT FINANCE PIPELINE")
    print("═"*60)

    # ── Step 1: Data ──
    print("\n[1/5]  Generating synthetic OHLCV data...")
    raw = generate_synthetic_ohlcv(n_days=1500, n_assets=50)
    print(f"       {raw.shape[0]:,} rows | {raw['ticker'].nunique()} assets | "
          f"{raw['date'].nunique()} trading days")

    # ── Step 2: Features ──
    print("\n[2/5]  Engineering features (lagged, no lookahead)...")
    features = build_features(raw)
    print(f"       {features.shape[0]:,} observations | "
          f"{len(FEATURE_COLS)} features | "
          f"target balance: {features['target'].mean():.2%}")

    # ── Step 3: Walk-forward ──
    print("\n[3/5]  Running walk-forward backtest...")
    predictions = run_walk_forward(
        features,
        burn_in_days=252,
        test_days=21,
        use_autosklearn=False,   # Set True when auto-sklearn is installed
        automl_time_secs=120,
    )
    print(f"\n       {len(predictions):,} out-of-sample predictions")

    # ── Step 4: Portfolio ──
    print("\n[4/5]  Constructing long/short portfolio...")
    pnl = construct_portfolio(
        predictions,
        long_quantile=0.9,
        short_quantile=0.1,
        transaction_cost_bps=5.0,
    )

    # ── Step 5: Analytics ──
    print("\n[5/5]  Computing performance metrics...")
    ic_df = compute_ic_series(predictions)
    perf  = compute_performance(pnl)

    print("\n" + "─"*60)
    print("  PERFORMANCE REPORT")
    print("─"*60)
    for k, v in perf.items():
        print(f"  {k:<25} {v}")
    print(f"  {'Mean IC':<25} {ic_df['ic'].mean():+.4f}")
    print(f"  {'IC t-stat':<25} "
          f"{ic_df['ic'].mean() / (ic_df['ic'].std() / len(ic_df)**0.5):+.2f}")
    print("─"*60)

    plot_results(pnl, ic_df, perf)
    return pnl, predictions, perf


if __name__ == "__main__":
    pnl, predictions, perf = main()