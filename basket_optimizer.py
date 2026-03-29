# =============================================================================
# BasketOptimizer: Johansen Cointegration + Optuna Bayesian Optimization
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optuna
import sys # Ajout pour quitter proprement
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BasketOptimizer:
    """
    Finds a mean-reverting synthetic asset (spread) from a basket of equities.

    Pipeline:
        1. Load & split data  →  In-Sample / Out-of-Sample
        2. Johansen Test      →  baseline cointegrating eigenvectors
        3. Optuna Search      →  weights that maximize OOS Sharpe Ratio
        4. Metrics & Plot     →  Sharpe, Max Drawdown, Half-Life
    """

    # -------------------------------------------------------------------------
    # LEARN: Cointegration vs. Correlation
    # -------------------------------------------------------------------------
    # Correlation measures whether two series *move together* at a point in
    # time (a contemporaneous, static relationship). Two series can be highly
    # correlated yet still drift apart permanently.
    #
    # Cointegration is a stronger, dynamic condition: a LINEAR COMBINATION of
    # the series is *stationary* (mean-reverting). This means no matter how far
    # apart the series wander, they are mathematically bound to converge.
    # The Johansen test finds the *number* of such relationships (cointegrating
    # rank) AND the coefficient vectors (eigenvectors) that produce stationarity.
    # This is what makes the spread "tradeable" via a pairs/basket strategy.
    # -------------------------------------------------------------------------

    def __init__(self, csv_path: str, train_ratio: float = 0.7, n_trials: int = 200):
        self.csv_path   = csv_path
        self.train_ratio = train_ratio
        self.n_trials   = n_trials

        # Will be populated during run()
        self.prices_is  = None   # In-Sample prices
        self.prices_oos = None   # Out-of-Sample prices
        self.johansen_weights = None
        self.optimized_weights = None
        self.best_sharpe = None
        self.study = None

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================

    def load_and_split(self) -> None:
        """Load CSV, parse date index, forward-fill, and split 70/30."""
        try:
            df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"[Erreur] Le fichier '{self.csv_path}' est introuvable.")
            sys.exit(1)

        df = df.sort_index().ffill().dropna()

        # VÉRIFICATION AJOUTÉE : S'assurer que les données ne sont pas vides
        if df.empty:
            print(f"[Erreur] Aucune donnée valide trouvée dans '{self.csv_path}'.")
            print("Vérifiez que le fichier CSV contient des données et n'est pas entièrement filtré par dropna().")
            sys.exit(1)

        split_idx = int(len(df) * self.train_ratio)
        
        # VÉRIFICATION AJOUTÉE : S'assurer qu'il y a assez de données pour le split
        if split_idx == 0 or split_idx == len(df):
            print(f"[Erreur] Pas assez de données pour séparer In-Sample et Out-of-Sample avec le ratio {self.train_ratio}.")
            sys.exit(1)

        self.prices_is  = df.iloc[:split_idx]
        self.prices_oos = df.iloc[split_idx:]

        print(f"[Data]  Total rows : {len(df)}")
        print(f"        In-Sample  : {self.prices_is.index[0].date()} → "
              f"{self.prices_is.index[-1].date()}  ({len(self.prices_is)} rows)")
        print(f"        OOS        : {self.prices_oos.index[0].date()} → "
              f"{self.prices_oos.index[-1].date()}  ({len(self.prices_oos)} rows)")
        print(f"        Assets     : {list(df.columns)}\n")

    # =========================================================================
    # 2a. JOHANSEN COINTEGRATION TEST
    # =========================================================================

    # LEARN: The Johansen Test
    # -------------------------------------------------------------------------
    # Given a (T × k) matrix of price series, Johansen solves a generalized
    # eigenvalue problem on the VAR residuals to find vectors β such that
    #   β' * P_t   is stationary (I(0)) even though each P_t is I(1).
    #
    # The test returns:
    #   • Trace / Max-Eigenvalue statistics  → how many cointegrating vectors exist
    #   • Eigenvectors (evec)                → the hedge ratios / basket weights
    #
    # We use det_order=0 (no deterministic trend in the cointegrating relation)
    # and k_ar_diff=1 (one lag). The FIRST eigenvector (column 0) corresponds
    # to the largest eigenvalue, i.e., the *strongest* mean-reverting combination.
    # -------------------------------------------------------------------------

    def run_johansen(self) -> np.ndarray:
        """Run Johansen test on in-sample prices; return first eigenvector."""
        result = coint_johansen(
            endog=self.prices_is.values,
            det_order=0,
            k_ar_diff=1
        )

        evec = result.evec[:, 0]          # strongest cointegrating vector
        evec /= np.abs(evec).sum()        # normalize to unit L1 norm

        self.johansen_weights = evec

        print("[Johansen]")
        print(f"  Trace statistic (r=0): {result.lr1[0]:.4f}  "
              f"(5% critical: {result.cvt[0, 1]:.4f})")
        print(f"  Eigenvector (weights): "
              f"{dict(zip(self.prices_is.columns, np.round(evec, 6)))}\n")

        return evec

    # =========================================================================
    # 2b. SPREAD & METRICS HELPERS
    # =========================================================================

    @staticmethod
    def _build_spread(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Compute the synthetic spread: S_t = Σ w_i * P_i_t (log prices)."""
        log_px = np.log(prices)
        return log_px.values @ weights

    @staticmethod
    def _sharpe(spread: np.ndarray, annual: int = 252) -> float:
        """Annualized Sharpe of the daily PnL of being LONG the spread."""
        pnl = np.diff(spread)
        if pnl.std() < 1e-9:
            return -999.0
        return (pnl.mean() / pnl.std()) * np.sqrt(annual)

    @staticmethod
    def _max_drawdown(spread: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown (in spread units)."""
        roll_max = np.maximum.accumulate(spread)
        dd = spread - roll_max
        return float(dd.min())

    # LEARN: Half-Life of Mean Reversion
    # -------------------------------------------------------------------------
    # The Ornstein-Uhlenbeck (OU) process models a mean-reverting spread as:
    #   dS = κ(μ − S) dt + σ dW
    # where κ is the speed of mean reversion.
    #
    # We estimate κ from a discrete OLS regression:
    #   ΔS_t = a + b * S_{t-1} + ε
    # where  b ≈ −κ (the reversion coefficient).
    #
    # Half-life = ln(2) / κ = −ln(2) / b
    # This tells you: "on average, how many days does it take the spread
    # to revert HALFWAY back to its mean?" A short half-life (5–30 days)
    # is ideal for a stat-arb strategy.
    # -------------------------------------------------------------------------

    @staticmethod
    def _half_life(spread: np.ndarray) -> float:
        """Estimate OU half-life via OLS regression on lagged spread."""
        s  = spread[:-1].reshape(-1, 1)
        ds = np.diff(spread)
        reg = OLS(ds, add_constant(s)).fit()
        b   = reg.params[1]               # mean-reversion coefficient
        if b >= 0:
            return np.inf                  # not mean-reverting
        return -np.log(2) / b

    # =========================================================================
    # 2c. OPTUNA BAYESIAN OPTIMISATION
    # =========================================================================

    # LEARN: Why Optimize for Sharpe Ratio?
    # -------------------------------------------------------------------------
    # Raw PnL is misleading — a strategy making $1M with $100M drawdown is
    # worse than one making $500K with no drawdown. The Sharpe Ratio normalizes
    # returns by their volatility:
    #
    #   Sharpe = (Mean Daily PnL) / (Std Daily PnL) × √252
    #
    # Bayesian Optimization (via Optuna's TPE sampler) is preferred over grid
    # search because it MODELS the objective function and proposes the next
    # trial intelligently, concentrating samples in promising regions of the
    # weight space. This makes it far more sample-efficient than random search,
    # especially in continuous multi-dimensional spaces.
    #
    # Key design choice: We optimize weights on IN-SAMPLE data but EVALUATE the
    # final Sharpe on OUT-OF-SAMPLE data. This prevents overfitting to noise
    # that would occur if we both optimized and evaluated on the same data.
    # -------------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: maximize IS Sharpe w.r.t. basket weights."""
        n = len(self.prices_is.columns)

        # Sample raw weights from a symmetric range around Johansen baseline
        raw = np.array([
            trial.suggest_float(f"w{i}", -1.0, 1.0) for i in range(n)
        ])

        # Blend Johansen prior with Optuna suggestion (keeps search grounded)
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        weights = alpha * self.johansen_weights + (1 - alpha) * raw

        # L1-normalize to avoid trivially large weights
        if np.abs(weights).sum() < 1e-9:
            return -999.0
        weights /= np.abs(weights).sum()

        spread = self._build_spread(self.prices_is, weights)
        return self._sharpe(spread)

    def optimize(self) -> np.ndarray:
        """Run Optuna study; return the best weights found."""
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="BasketOptimizer"
        )
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best = self.study.best_params
        n = len(self.prices_is.columns)
        raw = np.array([best[f"w{i}"] for i in range(n)])
        alpha = best["alpha"]
        weights = alpha * self.johansen_weights + (1 - alpha) * raw
        weights /= np.abs(weights).sum()

        self.optimized_weights = weights
        print(f"[Optuna]  Trials run       : {self.n_trials}")
        print(f"          Best IS Sharpe   : {self.study.best_value:.4f}")
        print(f"          Optimized weights: "
              f"{dict(zip(self.prices_is.columns, np.round(weights, 6)))}\n")
        return weights

    # =========================================================================
    # 3. REPORT METRICS
    # =========================================================================

    def report_metrics(self) -> dict:
        """Compute and print OOS Sharpe, Max Drawdown, and Half-Life."""
        spread_oos = self._build_spread(self.prices_oos, self.optimized_weights)

        sharpe   = self._sharpe(spread_oos)
        mdd      = self._max_drawdown(spread_oos)
        hl       = self._half_life(spread_oos)

        self.best_sharpe = sharpe

        print("=" * 50)
        print("  OUT-OF-SAMPLE PERFORMANCE")
        print("=" * 50)
        print(f"  Sharpe Ratio   : {sharpe:>10.4f}")
        print(f"  Max Drawdown   : {mdd:>10.4f}  (spread units)")
        print(f"  Half-Life      : {hl:>10.2f}  days")
        print("=" * 50 + "\n")

        return {"sharpe": sharpe, "max_drawdown": mdd, "half_life_days": hl}

    # =========================================================================
    # 4. VISUALISATION
    # =========================================================================

    def plot(self) -> None:
        """
        4-panel figure:
          (A) Individual price series (normalized)
          (B) IS spread vs. ±1σ / ±2σ bands
          (C) OOS spread vs. ±1σ bands
          (D) Optuna optimization history
        """
        fig = plt.figure(figsize=(16, 12), facecolor="#0f0f0f")
        gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)
        COLOR = {"aapl": "#00bfff", "msft": "#ff6b6b",
                 "googl": "#98fb98", "spread": "#ffd700",
                 "band": "#888888", "bg": "#1a1a1a"}

        ax1 = fig.add_subplot(gs[0, 0])   # Normalized prices
        ax2 = fig.add_subplot(gs[0, 1])   # IS spread
        ax3 = fig.add_subplot(gs[1, 0])   # OOS spread
        ax4 = fig.add_subplot(gs[1, 1])   # Optuna history

        def _style(ax, title):
            ax.set_facecolor(COLOR["bg"])
            ax.set_title(title, color="white", fontsize=11, pad=8)
            ax.tick_params(colors="gray")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            ax.yaxis.label.set_color("gray")
            ax.xaxis.label.set_color("gray")

        # --- Panel A: Normalized prices ---
        colors_cycle = [COLOR["aapl"], COLOR["msft"], COLOR["googl"], "#da70d6", "#f4a460"]
        norm = self.prices_is / self.prices_is.iloc[0]
        for col, c in zip(norm.columns, colors_cycle):
            ax1.plot(norm.index, norm[col], label=col, color=c, linewidth=1.2)
        ax1.legend(facecolor="#222222", edgecolor="#444", labelcolor="white", fontsize=8)
        _style(ax1, "Normalized Prices (In-Sample)")

        # --- Panel B: IS spread ---
        spread_is = self._build_spread(self.prices_is, self.optimized_weights)
        s_mean, s_std = spread_is.mean(), spread_is.std()
        idx_is = self.prices_is.index
        ax2.plot(idx_is, spread_is, color=COLOR["spread"], linewidth=1.0, label="Spread")
        for mult, alpha in [(1, 0.5), (2, 0.25)]:
            ax2.axhline(s_mean + mult * s_std, color=COLOR["band"],
                        linestyle="--", alpha=alpha, linewidth=0.8)
            ax2.axhline(s_mean - mult * s_std, color=COLOR["band"],
                        linestyle="--", alpha=alpha, linewidth=0.8)
        ax2.axhline(s_mean, color="white", linestyle=":", linewidth=0.8, label="μ")
        ax2.legend(facecolor="#222222", edgecolor="#444", labelcolor="white", fontsize=8)
        _style(ax2, "Spread — In-Sample (Optimized Weights)")

        # --- Panel C: OOS spread ---
        spread_oos = self._build_spread(self.prices_oos, self.optimized_weights)
        idx_oos = self.prices_oos.index
        ax3.plot(idx_oos, spread_oos, color="#ff8c00", linewidth=1.0, label="OOS Spread")
        for mult, alpha in [(1, 0.5)]:
            ax3.axhline(s_mean + mult * s_std, color=COLOR["band"],
                        linestyle="--", alpha=alpha, linewidth=0.8, label=f"±{mult}σ (IS)")
            ax3.axhline(s_mean - mult * s_std, color=COLOR["band"],
                        linestyle="--", alpha=alpha, linewidth=0.8)
        ax3.axhline(s_mean, color="white", linestyle=":", linewidth=0.8, label="IS μ")
        ax3.legend(facecolor="#222222", edgecolor="#444", labelcolor="white", fontsize=8)
        _style(ax3, f"Spread — Out-of-Sample  |  Sharpe: {self.best_sharpe:.3f}")

        # --- Panel D: Optuna history ---
        values = [t.value for t in self.study.trials if t.value is not None]
        best_so_far = np.maximum.accumulate(values)
        ax4.plot(values, color="#666666", linewidth=0.7, alpha=0.5, label="Trial Sharpe")
        ax4.plot(best_so_far, color="#00ff7f", linewidth=1.4, label="Best so far")
        ax4.set_xlabel("Trial")
        ax4.set_ylabel("Sharpe (IS)")
        ax4.legend(facecolor="#222222", edgecolor="#444", labelcolor="white", fontsize=8)
        _style(ax4, "Optuna Optimization History")

        fig.suptitle("BasketOptimizer — Johansen + Bayesian Optimization",
                     color="white", fontsize=14, y=0.98)
        plt.savefig("basket_optimizer_report.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()
        print("[Plot]  Saved → basket_optimizer_report.png")

    # =========================================================================
    # MASTER RUN METHOD
    # =========================================================================

    def run(self) -> dict:
        """Execute the full pipeline end-to-end."""
        print("\n" + "━" * 55)
        print("  BasketOptimizer  —  Starting Pipeline")
        print("━" * 55 + "\n")

        self.load_and_split()
        self.run_johansen()
        self.optimize()
        metrics = self.report_metrics()
        self.plot()

        return metrics


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    optimizer = BasketOptimizer(
        csv_path="basket_prices.csv",
        train_ratio=0.70,    # 70% in-sample, 30% OOS
        n_trials=300         # increase for better search coverage
    )
    
    # Exécution de l'optimiseur
    results = optimizer.run()


    # =============================================================================
    # RESULTS TABLE  —  LaTeX-ready summary
    # =============================================================================
    # Ce bloc ne s'exécutera que si 'optimizer.run()' se termine avec succès
    import textwrap

    assets  = list(optimizer.prices_is.columns)
    weights = optimizer.optimized_weights
    sharpe  = optimizer.best_sharpe

    col_w = max(len(a) for a in assets) + 2          # dynamic column width
    divider = "+" + "-"*(col_w+2) + "+" + "-"*14 + "+"

    print("\n" + "="*len(divider))
    print("  FINAL RESULTS  —  Copy-paste values for LaTeX")
    print("="*len(divider))

    # --- Cointegration Weights ---
    print(f"\n  {'Asset':<{col_w}}  {'Weight':>12}")
    print("  " + divider)
    for asset, w in zip(assets, weights):
        print(f"  | {asset:<{col_w}}| {w:>+12.6f} |")
    print("  " + divider)

    # --- Sharpe Ratio ---
    print(f"\n  {'Metric':<{col_w}}  {'Value':>12}")
    print("  " + divider)
    print(f"  | {'OOS Sharpe Ratio':<{col_w}}| {sharpe:>+12.4f} |")
    print("  " + divider)

    # --- Raw numbers for quick copy ---
    print("\n  ── Raw values (no formatting) ──")
    for asset, w in zip(assets, weights):
        print(f"  {asset}: {w:.10f}")
    print(f"  Sharpe: {sharpe:.10f}")
    print()