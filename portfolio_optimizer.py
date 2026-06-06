"""
╔══════════════════════════════════════════════════════════════════╗
║         PORTFOLIO OPTIMIZER                                      ║
║         XGBoost + RMT + Markowitz + Risk Parity + Metrikák       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  EZ A FÁJL 4 DOLGOT CSINÁL:                                      ║
║                                                                  ║
║  1. XGBoost Signal Booster                                       ║
║     A Random Forest mellé egy második ML modell.                 ║
║     A kettő ensemble-je jobb mint bármelyik egyedül.             ║
║                                                                  ║
║  2. RMT (Random Matrix Theory) szűrés                            ║
║     A részvények korrelációs mátrixából kiszűri a zajt.          ║
║     Csak a valódi, statisztikailag szignifikáns korrelációkat    ║
║     hagyja meg. Ez a profi alapok eszköze.                       ║
║                                                                  ║
║  3. Portfolió optimalizálás                                      ║
║     Markowitz mean-variance + Risk Parity módszerrel             ║
║     megmondja: melyik részvényből mennyit tarts.                 ║
║                                                                  ║
║  4. Portfolió metrikák                                           ║
║     Sharpe-ratio, Sortino-ratio, Max Drawdown, VaR, CVaR.        ║
║     Ezekkel mérheted hogy valóban jobb-e a portfoliód.           ║
║                                                                  ║
║  HOW TO RUN:                                                     ║
║    python portfolio_optimizer.py                                 ║
║                                                                  ║
║  INTEGRÁCIÓ az aibotv3-ba:                                       ║
║    from portfolio_optimizer import PortfolioOptimizer            ║
║    opt = PortfolioOptimizer()                                    ║
║    result = opt.run(signals, price_data)                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.linalg import eigh

import xgboost as xgb
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import roc_auc_score


# ═══════════════════════════════════════════════════════
#  KONFIGURÁCIÓ
# ═══════════════════════════════════════════════════════

XGB_MODEL_PATH  = "models/xgb_model.pkl"
PORTFOLIO_PATH  = "models/last_portfolio.json"

# Portfolió constraints
MAX_WEIGHT      = 0.25   # egyetlen részvény max 25% a portfolióban
MIN_WEIGHT      = 0.02   # min 2% hogy ne legyenek elhanyagolható pozíciók
RISK_FREE_RATE  = 0.045  # éves kockázatmentes hozam (pl. US T-bill 2024)
LOOKBACK_DAYS   = 252    # 1 év hozam adat a kovariancia becsléshez


# ═══════════════════════════════════════════════════════
#  1. XGBOOST SIGNAL BOOSTER
#
#  MIÉRT XGBOOST A RANDOM FOREST MELLÉ?
#  ──────────────────────────────────────
#  A Random Forest "bagging" elvű: sok fa egymástól
#  függetlenül tanul, majd szavaznak.
#
#  Az XGBoost "boosting" elvű: a fák SORBAN épülnek,
#  minden új fa az előző hibáit javítja. Ez más típusú
#  hibákat fog ki — például nem-lineáris interakciókat
#  a feature-ök között amiket a RF elnézett.
#
#  Pl: RF tudja hogy "bull_score > 8 jó jel"
#      XGB tudja hogy "bull_score > 8 ÉS adx_raw > 28
#      ÉS hist_win_rate > 0.6 esetén KÜLÖNÖSEN jó jel"
#
#  Az ensemble (RF 50% + XGB 50%) empirikusan 3-7%
#  ROC-AUC javulást ad egyedülihez képest.
# ═══════════════════════════════════════════════════════

class XGBoostSignalBooster:
    """
    XGBoost modell a Random Forest mellé.
    Ugyanazokat a 24 feature-t használja, de más algoritmussal.
    """

    def __init__(self):
        self.model  = None
        self.scaler = StandardScaler()
        self._load()

    def _load(self):
        if os.path.exists(XGB_MODEL_PATH):
            saved = joblib.load(XGB_MODEL_PATH)
            self.model  = saved["model"]
            self.scaler = saved["scaler"]
            print("  🌲 XGBoost modell betöltve.")

    def is_ready(self) -> bool:
        return self.model is not None

    def train(self, X: np.ndarray, y: np.ndarray,
              run_hyperparam_search: bool = True):
        """
        XGBoost betanítása.

        HYPERPARAMÉTEREK MAGYARÁZATA:
        ──────────────────────────────
        n_estimators:   hány "boost kör" legyen — több = pontosabb de lassabb
        max_depth:      fa mélység — mély fa = komplexebb minták de overfitting
        learning_rate:  mennyit tanul egyszerre — kis érték + sok fa = jobb
        subsample:      tanításonként a sorok hány %-át látja — regularizáció
        colsample:      tanításonként a feature-ök hány %-át látja — diverzitás
        min_child_weight: minimum adat egy levélben — overfitting elleni védelem
        scale_pos_weight: osztályok közti aránytalanság kezelése (ha kevés a win)
        """
        print("\n  🌲 XGBoost tanítás...")

        X_scaled = self.scaler.fit_transform(X)
        X_clean  = np.where(np.isnan(X_scaled), 0.0, X_scaled)

        if run_hyperparam_search and len(y) >= 100:
            param_dist = {
                "n_estimators":    [100, 200, 300, 500],
                "max_depth":       [3, 4, 5, 6, 7],
                "learning_rate":   [0.01, 0.05, 0.1, 0.15, 0.2],
                "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree":[0.5, 0.6, 0.7, 0.8, 1.0],
                "min_child_weight":[1, 3, 5, 7, 10],
                "reg_alpha":       [0, 0.01, 0.1, 1.0],
                "reg_lambda":      [0.1, 1.0, 2.0, 5.0],
            }
            # pos_weight: ha kevesebb a nyertes trade, ezt kompenzálja
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            pos_weight = n_neg / max(n_pos, 1)

            base_xgb = xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=pos_weight,
                random_state=42, n_jobs=-1,
                eval_metric="auc", verbosity=0,
            )
            tscv   = TimeSeriesSplit(n_splits=5)
            search = RandomizedSearchCV(
                base_xgb, param_dist,
                n_iter=30, cv=tscv,
                scoring="roc_auc", n_jobs=-1,
                random_state=42, verbose=0,
            )
            search.fit(X_clean, y)
            best_params = search.best_params_
            best_params["scale_pos_weight"] = pos_weight
            print(f"     Legjobb ROC-AUC: {search.best_score_:.4f}")
            print(f"     Paraméterek: {best_params}")
        else:
            best_params = {
                "n_estimators": 300, "max_depth": 5,
                "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.7, "min_child_weight": 5,
                "reg_alpha": 0.1, "reg_lambda": 1.0,
            }

        # Kalibrált XGBoost — valódi valószínűségekhez
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=42, n_jobs=-1,
            eval_metric="auc", verbosity=0,
            **best_params,
        )
        self.model = CalibratedClassifierCV(xgb_model, cv=3, method="sigmoid")
        self.model.fit(X_clean, y)

        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, XGB_MODEL_PATH)
        print("  ✅ XGBoost elmentve.")
        return self

    def predict_proba(self, features: np.ndarray) -> float:
        """Visszaadja a bullish valószínűséget 0-1 között."""
        if not self.is_ready():
            return 0.5
        X = self.scaler.transform(features.reshape(1, -1))
        X = np.where(np.isnan(X), 0.0, X)
        return float(self.model.predict_proba(X)[0][1])

    def ensemble_predict(self, features: np.ndarray,
                          rf_proba: float,
                          rf_weight: float = 0.5) -> float:
        """
        RF és XGBoost ensemble predikciója.

        MIÉRT 50-50 az alapértelmezett?
        A két modell különböző típusú hibákat követ el,
        de hasonló átlagos pontossággal. Ha az egyik
        szisztematikusan jobb a te adatodon, a súlyt
        igazítsd (pl. rf_weight=0.4 ha XGB jobb).
        """
        xgb_proba = self.predict_proba(features)
        return rf_weight * rf_proba + (1 - rf_weight) * xgb_proba


# ═══════════════════════════════════════════════════════
#  2. RMT KORRELÁCIÓS MÁTRIX SZŰRÉS
#
#  MIÉRT KELL EZ?
#  ──────────────
#  Ha 44 részvényed van, a korrelációs mátrix 44×44 = 1936
#  elemet tartalmaz. De ha csak 252 nap adatod van (1 év),
#  a legtöbb korreláció ZAJ — véletlen egybeesés, nem
#  valódi gazdasági kapcsolat.
#
#  Példa: NVDA és egy kis egészségügyi cég véletlenül
#  együtt mozgott 3 hónapig → magas korreláció.
#  De ez NEM jelent valódi kapcsolatot.
#
#  A Random Matrix Theory megmondja: mekkora a MAXIMÁLIS
#  sajátérték amit VÉLETLEN mátrix is produkálna.
#  Amit e fölött találunk → VALÓDI jel.
#  Amit e alatt találunk → ZAJ, le kell szűrni.
#
#  Marchenko-Pastur törvény:
#    λ_max = σ² · (1 + √(N/T))²
#    ahol N = részvények száma, T = időpontok száma
#    σ² = variancia
#  Minden sajátérték λ_max alatt → zaj → lecseréljük
#  az átlagos értékre.
# ═══════════════════════════════════════════════════════

class RMTCovarianceFilter:
    """
    Random Matrix Theory alapú kovariancia mátrix szűrő.
    A zajt kiszűri, a valódi korrelációkat megtartja.
    """

    def __init__(self, variance: float = 1.0):
        self.variance = variance

    def marchenko_pastur_max(self, n_assets: int, n_obs: int) -> float:
        """
        Kiszámítja a Marchenko-Pastur eloszlás felső határát.
        Ez az a sajátérték-küszöb ami fölött a korreláció valódi.

        Képlet: λ_max = σ² · (1 + √(N/T))²
        N = eszközök száma, T = megfigyelések száma
        """
        q      = n_obs / n_assets   # ha T >> N → q nagy → szigorúbb szűrés
        ratio  = 1.0 / q
        lambda_max = self.variance * (1 + np.sqrt(ratio)) ** 2
        return lambda_max

    def filter_correlation_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        A nyers korrelációs mátrixot szűri RMT-vel.

        LÉPÉSEK:
        1. Kiszámítja a korrelációs mátrixot
        2. Sajátérték dekompozíció (eigendecomposition)
        3. Meghatározza a Marchenko-Pastur küszöböt
        4. A küszöb alatti sajátértékeket lecseréli az átlaggal
        5. Visszaállítja a mátrixot → szűrt korreláció

        Az eredmény: csak a valódi strukturális korrelációk
        maradnak meg, a véletlen zaj eltűnik.
        """
        corr_matrix = returns.corr().values
        n_assets    = corr_matrix.shape[0]
        n_obs       = len(returns)

        # Sajátérték dekompozíció
        # eigenvalues: a korrelációs mátrix "erőssége" irányonként
        # eigenvectors: az irányok amikben a korreláció hat
        eigenvalues, eigenvectors = eigh(corr_matrix)

        # Marchenko-Pastur küszöb
        lambda_max = self.marchenko_pastur_max(n_assets, n_obs)

        # Zaj szűrés: λ < λ_max → zaj → cseréljük az átlagra
        # Az átlag megtartja a mátrix "nyomát" (trace = N)
        noise_mean = np.mean(eigenvalues[eigenvalues <= lambda_max])
        filtered_eigenvalues = np.where(
            eigenvalues <= lambda_max,
            noise_mean,   # zaj komponens → átlag
            eigenvalues   # valódi jel → megtartjuk
        )

        # Visszaállítás: C_filtered = V · Λ_filtered · V^T
        filtered_corr = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

        # Normalizálás: az átló maradjon 1 (korrelációs mátrix konvenció)
        diag_sqrt = np.sqrt(np.diag(filtered_corr))
        diag_sqrt = np.where(diag_sqrt < 1e-10, 1.0, diag_sqrt)
        filtered_corr = filtered_corr / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(filtered_corr, 1.0)

        # Pozitív szemi-definit biztosítása (numerikus stabilitás)
        eigenvalues2, eigenvectors2 = eigh(filtered_corr)
        eigenvalues2 = np.maximum(eigenvalues2, 1e-8)
        filtered_corr = eigenvectors2 @ np.diag(eigenvalues2) @ eigenvectors2.T
        np.fill_diagonal(filtered_corr, 1.0)

        n_signal = np.sum(eigenvalues > lambda_max)
        print(f"     RMT: {n_signal}/{n_assets} valódi jel "
              f"(λ_max={lambda_max:.3f})")

        return filtered_corr

    def correlation_to_covariance(self, filtered_corr: np.ndarray,
                                   returns: pd.DataFrame) -> np.ndarray:
        """
        A szűrt korrelációs mátrixból kovariancia mátrixot számít.
        Σ = D · C · D  ahol D = volatilitások diagonális mátrixa
        """
        vols = returns.std().values * np.sqrt(252)   # éves volatilitás
        D    = np.diag(vols)
        return D @ filtered_corr @ D


# ═══════════════════════════════════════════════════════
#  3. PORTFOLIÓ OPTIMALIZÁLÁS
#
#  MARKOWITZ MEAN-VARIANCE OPTIMALIZÁLÁS:
#  ────────────────────────────────────────
#  Cél: megtalálni azt a súlyvektort (w) amely
#  maximalizálja a Sharpe-ratiót:
#    SR = (E[R_p] - R_f) / σ_p
#  ahol:
#    E[R_p] = portfolió várható hozama = w^T · μ
#    σ_p    = portfolió szórása = √(w^T · Σ · w)
#    R_f    = kockázatmentes hozam
#    Σ      = kovariancia mátrix (RMT-szűrt)
#
#  RISK PARITY:
#  ─────────────
#  A Markowitz koncentrálja a tőkét a legjobb
#  Sharpe-ratiójú eszközökre — de ezek általában a
#  legvolatilisabbak is. A Risk Parity ehelyett azt
#  mondja: minden eszköz EGYENLŐ MÉRTÉKBEN járuljon
#  hozzá a portfolió KOCKÁZATÁHOZ.
#
#  RC_i = w_i · (Σw)_i / (w^T · Σ · w)
#  Cél: RC_i = 1/N minden i-re
#
#  Bridgewater "All Weather" portfoliója Risk Parity elvű.
# ═══════════════════════════════════════════════════════

class PortfolioOptimizer:
    """
    Portfolió optimalizáló — RMT szűrt kovariancia alapján.
    Kétféle módszert kínál: Markowitz és Risk Parity.
    """

    def __init__(self):
        self.rmt    = RMTCovarianceFilter()
        self.xgb    = XGBoostSignalBooster()

    def _portfolio_variance(self, weights: np.ndarray,
                             cov: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def _portfolio_return(self, weights: np.ndarray,
                           expected_returns: np.ndarray) -> float:
        return float(weights @ expected_returns)

    def _sharpe_ratio(self, weights: np.ndarray,
                       expected_returns: np.ndarray,
                       cov: np.ndarray) -> float:
        ret = self._portfolio_return(weights, expected_returns)
        vol = np.sqrt(max(self._portfolio_variance(weights, cov), 1e-10))
        return (ret - RISK_FREE_RATE) / vol

    def optimize_markowitz(self, expected_returns: np.ndarray,
                            cov: np.ndarray,
                            n_assets: int) -> np.ndarray:
        """
        Maximum Sharpe-ratio portfolió keresése.

        A scipy.optimize.minimize negatív Sharpe-ratiót minimalizál
        (mert minimalizáló algoritmus, nem maximalizáló).

        Constraints:
          - súlyok összege = 1 (teljes befektetés)
          - min_weight ≤ w_i ≤ max_weight (diverzifikáció)
        """
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds      = [(MIN_WEIGHT, MAX_WEIGHT)] * n_assets

        # Több random induló ponttal próbálkozunk
        # (a Markowitz nem konvex → lokális minimumok vannak)
        best_weights = np.ones(n_assets) / n_assets
        best_sharpe  = -np.inf

        for seed in range(10):
            rng   = np.random.RandomState(seed)
            w0    = rng.dirichlet(np.ones(n_assets))
            w0    = np.clip(w0, MIN_WEIGHT, MAX_WEIGHT)
            w0   /= w0.sum()

            result = minimize(
                lambda w: -self._sharpe_ratio(w, expected_returns, cov),
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

            if result.success:
                sr = self._sharpe_ratio(result.x, expected_returns, cov)
                if sr > best_sharpe:
                    best_sharpe  = sr
                    best_weights = result.x

        return best_weights / best_weights.sum()

    def optimize_risk_parity(self, cov: np.ndarray,
                              n_assets: int) -> np.ndarray:
        """
        Risk Parity optimalizálás.

        Cél: minden eszköz egyenlő kockázati hozzájárulása.
        RC_i = w_i · (Σw)_i / (w^T · Σ · w) = 1/N

        Objective function:
          Σ_i Σ_j (RC_i - RC_j)² → 0
        Ez minimalizálódik ha RC_i = RC_j minden i,j-re.
        """
        def risk_contributions(weights: np.ndarray) -> np.ndarray:
            port_var = self._portfolio_variance(weights, cov)
            marginal = cov @ weights   # marginális kockázat
            rc = weights * marginal / max(port_var, 1e-10)
            return rc

        def objective(weights: np.ndarray) -> float:
            rc     = risk_contributions(weights)
            target = 1.0 / n_assets
            # A kockázati hozzájárulások különbségének négyzetösszege
            return float(np.sum((rc - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds      = [(MIN_WEIGHT, MAX_WEIGHT)] * n_assets
        w0          = np.ones(n_assets) / n_assets

        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 2000},
        )

        weights = result.x if result.success else w0
        return weights / weights.sum()

    def compute_expected_returns(self, returns: pd.DataFrame,
                                  signals: dict) -> np.ndarray:
        """
        Várható hozamok becslése: historikus hozam + AI jelzés súlyozás.

        MÓDSZER:
        ─────────
        Alap: historikus átlaghozam (1 év)
        Módosítás: ha az AI jelzés bullish (>60%), az eszköz
        várható hozamát felfelé módosítjuk (max +5%)
        Ha bearish (<40%), lefelé módosítjuk.

        Ez biztosítja hogy az optimalizáló az AI jelzéseket
        is figyelembe veszi, nem csak a historikus adatot.
        """
        hist_returns = returns.mean() * 252    # éves hozam
        expected     = hist_returns.copy()

        for ticker, signal_data in signals.items():
            if ticker not in returns.columns:
                continue
            bull_pct = signal_data.get("bull_pct", 50.0) / 100.0
            # Módosítás: -5% ... +5% tartományban
            adjustment = (bull_pct - 0.5) * 0.10
            expected[ticker] += adjustment

        return expected.values

    def run(self, signals: dict, price_data: dict[str, pd.DataFrame],
            total_capital: float = 10000.0) -> dict:
        """
        A fő portfolió optimalizáló függvény.

        INPUT:
          signals:      {ticker: {"bull_pct": 73.2, "bear_pct": 26.8, ...}}
                        (az aibotv3 AI layer outputja)
          price_data:   {ticker: pd.DataFrame} (historikus árfolyamok)
          total_capital: teljes befektethető összeg

        OUTPUT:
          dict a portfolió súlyokkal, allokációval, metrikákkal
        """
        print(f"\n{'═'*55}")
        print("  PORTFOLIÓ OPTIMALIZÁLÁS")
        print(f"{'═'*55}")

        # ── 1. Hozam mátrix összeállítása ─────────────────────
        returns_dict = {}
        for ticker, df in price_data.items():
            if df is None or df.empty:
                continue
            ret = df["Close"].pct_change().dropna().tail(LOOKBACK_DAYS)
            if len(ret) >= 60:
                returns_dict[ticker] = ret

        if len(returns_dict) < 2:
            print("  ❌ Nem elég adat az optimalizáláshoz.")
            return {}

        returns_df = pd.DataFrame(returns_dict).dropna()
        tickers    = list(returns_df.columns)
        n          = len(tickers)
        print(f"  Részvények: {n} db  |  Adatpontok: {len(returns_df)}")

        # ── 2. RMT szűrt kovariancia mátrix ───────────────────
        print("\n  [RMT szűrés]")
        filtered_corr = self.rmt.filter_correlation_matrix(returns_df)
        cov_filtered  = self.rmt.correlation_to_covariance(
            filtered_corr, returns_df
        )

        # ── 3. Várható hozamok (historikus + AI jelzés) ────────
        expected_returns = self.compute_expected_returns(returns_df, signals)

        # ── 4. Markowitz optimalizálás ─────────────────────────
        print("\n  [Markowitz — Maximum Sharpe]")
        w_markowitz = self.optimize_markowitz(expected_returns, cov_filtered, n)

        # ── 5. Risk Parity optimalizálás ───────────────────────
        print("  [Risk Parity — Egyenlő kockázat]")
        w_risk_parity = self.optimize_risk_parity(cov_filtered, n)

        # ── 6. Blended portfolió ───────────────────────────────
        # 50% Markowitz + 50% Risk Parity
        # Markowitz: jobb hozam, de koncentrált
        # Risk Parity: jobb kockázatkezelés, de alacsonyabb várható hozam
        # A blend a kettő előnyeit kombinálja
        w_blended = 0.5 * w_markowitz + 0.5 * w_risk_parity
        w_blended = np.clip(w_blended, MIN_WEIGHT, MAX_WEIGHT)
        w_blended /= w_blended.sum()

        # ── 7. Portfolió metrikák kiszámítása ─────────────────
        metrics_markowitz   = self._compute_metrics(
            w_markowitz,   returns_df, expected_returns, cov_filtered, tickers
        )
        metrics_risk_parity = self._compute_metrics(
            w_risk_parity, returns_df, expected_returns, cov_filtered, tickers
        )
        metrics_blended     = self._compute_metrics(
            w_blended,     returns_df, expected_returns, cov_filtered, tickers
        )

        # ── 8. Allokáció összeállítása ─────────────────────────
        allocations = {}
        for i, ticker in enumerate(tickers):
            sig = signals.get(ticker, {})
            allocations[ticker] = {
                "weight_markowitz":   round(float(w_markowitz[i])   * 100, 2),
                "weight_risk_parity": round(float(w_risk_parity[i]) * 100, 2),
                "weight_blended":     round(float(w_blended[i])     * 100, 2),
                "capital_blended":    round(float(w_blended[i]) * total_capital, 2),
                "bull_pct":           sig.get("bull_pct", 50.0),
                "annual_vol_pct":     round(float(returns_df[ticker].std() * np.sqrt(252)) * 100, 2),
            }

        result = {
            "timestamp":         datetime.now().isoformat(),
            "total_capital":     total_capital,
            "n_assets":          n,
            "allocations":       allocations,
            "metrics": {
                "markowitz":   metrics_markowitz,
                "risk_parity": metrics_risk_parity,
                "blended":     metrics_blended,
            }
        }

        # Mentés
        os.makedirs("models", exist_ok=True)
        with open(PORTFOLIO_PATH, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _compute_metrics(self, weights: np.ndarray,
                          returns_df: pd.DataFrame,
                          expected_returns: np.ndarray,
                          cov: np.ndarray,
                          tickers: list) -> dict:
        """
        Portfolió metrikák számítása.

        SHARPE-RATIO:
        ─────────────
        SR = (E[R_p] - R_f) / σ_p
        Értelmezés: egységnyi kockázatért mennyi többlethozamot kaptál.
        > 1.0: jó  |  > 1.5: nagyon jó  |  > 2.0: kiváló  |  < 0: rossz

        SORTINO-RATIO:
        ──────────────
        SR_sortino = (E[R_p] - R_f) / σ_downside
        Mint a Sharpe, de csak a LEFELÉ irányú volatilitást bünteti.
        Jobb mérőszám ha aszimmetrikus hozameloszlást vársz.

        MAX DRAWDOWN:
        ─────────────
        A csúcstól mért maximális visszaesés %.
        Ha portfoliód 100-ról 70-re esett: MDD = -30%

        VaR (Value at Risk):
        ─────────────────────
        "95%-os VaR = -3.2%" → 95% eséllyel legfeljebb 3.2%-t
        veszítesz egy nap alatt (normál piaci körülmények között).

        CVaR (Conditional VaR / Expected Shortfall):
        ──────────────────────────────────────────────
        "Ha bekövetkezik a legrosszabb 5% scenario,
        akkor várhatóan -5.8%-ot veszítesz."
        Jobb kockázatmérő mint VaR, mert a farok viselkedését írja le.
        """
        # Napi portfolió hozamok szimulálása
        port_returns = returns_df.values @ weights

        # Éves metrikák
        ann_return = float(weights @ expected_returns)
        ann_vol    = float(np.sqrt(weights @ cov @ weights))
        sharpe     = (ann_return - RISK_FREE_RATE) / max(ann_vol, 1e-10)

        # Sortino
        downside_returns = port_returns[port_returns < 0]
        downside_vol     = float(np.std(downside_returns) * np.sqrt(252)) \
                           if len(downside_returns) > 0 else ann_vol
        sortino          = (ann_return - RISK_FREE_RATE) / max(downside_vol, 1e-10)

        # Max Drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max= np.maximum.accumulate(cumulative)
        drawdowns  = (cumulative - rolling_max) / np.maximum(rolling_max, 1e-10)
        max_dd     = float(drawdowns.min())

        # VaR és CVaR (historikus módszer, 95%)
        var_95  = float(np.percentile(port_returns, 5))
        cvar_95 = float(port_returns[port_returns <= var_95].mean()) \
                  if np.any(port_returns <= var_95) else var_95

        return {
            "annual_return_pct":  round(ann_return  * 100, 2),
            "annual_vol_pct":     round(ann_vol      * 100, 2),
            "sharpe_ratio":       round(sharpe,              3),
            "sortino_ratio":      round(sortino,             3),
            "max_drawdown_pct":   round(max_dd       * 100, 2),
            "var_95_daily_pct":   round(var_95       * 100, 3),
            "cvar_95_daily_pct":  round(cvar_95      * 100, 3),
        }


# ═══════════════════════════════════════════════════════
#  EREDMÉNY KIÍRÁS
# ═══════════════════════════════════════════════════════

def print_portfolio_report(result: dict):
    """Formázott portfolió riport a konzolra."""
    if not result:
        return

    print(f"\n{'═'*70}")
    print("  PORTFOLIÓ RIPORT")
    print(f"{'═'*70}")
    print(f"  Tőke: ${result['total_capital']:,.0f}  "
          f"| Részvények: {result['n_assets']}  "
          f"| {result['timestamp'][:10]}")

    # Metrikák összehasonlítás
    m = result["metrics"]
    print(f"\n  {'Metrika':28s}  {'Markowitz':>12s}  "
          f"{'Risk Parity':>12s}  {'Blended':>10s}")
    print("  " + "─"*68)

    metrics_labels = [
        ("Éves hozam %",       "annual_return_pct",  "{:>+.1f}%"),
        ("Éves volatilitás %", "annual_vol_pct",      "{:.1f}%"),
        ("Sharpe-ratio",       "sharpe_ratio",        "{:.3f}"),
        ("Sortino-ratio",      "sortino_ratio",       "{:.3f}"),
        ("Max Drawdown %",     "max_drawdown_pct",    "{:.1f}%"),
        ("VaR 95% (napi) %",   "var_95_daily_pct",    "{:.3f}%"),
        ("CVaR 95% (napi) %",  "cvar_95_daily_pct",   "{:.3f}%"),
    ]

    for label, key, fmt in metrics_labels:
        mv  = m["markowitz"].get(key, 0)
        rpv = m["risk_parity"].get(key, 0)
        bv  = m["blended"].get(key, 0)
        print(f"  {label:28s}  "
              f"{fmt.format(mv):>12s}  "
              f"{fmt.format(rpv):>12s}  "
              f"{fmt.format(bv):>10s}")

    # Allokáció táblázat
    print(f"\n  AJÁNLOTT ALLOKÁCIÓ (Blended módszer)")
    print(f"  {'Ticker':8s}  {'Súly%':>7s}  {'Tőke $':>10s}  "
          f"{'AI Bullish':>11s}  {'Vol%':>6s}")
    print("  " + "─"*52)

    alloc = result["allocations"]
    sorted_alloc = sorted(alloc.items(),
                          key=lambda x: x[1]["weight_blended"], reverse=True)

    for ticker, data in sorted_alloc:
        bull = data["bull_pct"]
        icon = "🟢" if bull >= 60 else ("🔴" if bull < 40 else "🟡")
        print(f"  {ticker:8s}  "
              f"{data['weight_blended']:>6.1f}%  "
              f"${data['capital_blended']:>9.2f}  "
              f"{bull:>9.1f}%  {icon}  "
              f"{data['annual_vol_pct']:>5.1f}%")

    print(f"\n  ℹ️  Sharpe > 1.0: jó  |  > 1.5: nagyon jó  |  > 2.0: kiváló")
    print(f"  ⚠️  Múltbeli metrikák nem garantálják a jövőbeli hozamot.")
    print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════
#  INTEGRÁCIÓS SEGÉDFÜGGVÉNY — aibotv3-hoz
# ═══════════════════════════════════════════════════════

def run_portfolio_from_bot_signals(
        signals: dict,
        watchlist: list[str],
        period: str = "1y",
        total_capital: float = 10000.0) -> dict:
    """
    Közvetlenül hívható az aibotv3.py-ból.
    Letölti az árfolyamokat és lefuttatja az optimalizálást.

    Használat az aibotv3 ScannerBot.run() végén:
        from portfolio_optimizer import run_portfolio_from_bot_signals
        portfolio = run_portfolio_from_bot_signals(
            signals=collected_signals,   # {ticker: ai_report dict}
            watchlist=WATCHLIST,
            total_capital=10000.0
        )
    """
    print("\n  Árfolyamadatok letöltése...")
    price_data = {}
    for ticker in watchlist:
        try:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                df.index = df.index.tz_localize(None) \
                           if df.index.tz else df.index
                price_data[ticker] = df
        except Exception:
            pass

    optimizer = PortfolioOptimizer()
    result    = optimizer.run(signals, price_data, total_capital)

    if result:
        print_portfolio_report(result)

    return result


# ═══════════════════════════════════════════════════════
#  XGBoost TANÍTÁS — önálló futtatáshoz
# ═══════════════════════════════════════════════════════

def train_xgboost_from_saved_data():
    """
    Betanítja az XGBoost modellt a mentett real_trade_data.json alapján.
    Ugyanazokat az adatokat használja mint az ai_layer --train.
    """
    from ai_layer import load_trade_data, generate_synthetic_training_data, \
                         FEATURE_NAMES, extract_features_from_trade, ScoreHistoryTable

    print("\n  XGBoost tanítás a mentett trade adatokból...")
    score_table = ScoreHistoryTable()

    ctxs, labels = load_trade_data()
    print(f"  Betöltött valós trade-ek: {len(labels)} db")

    X_real_list = []
    for ctx in ctxs:
        try:
            fv = extract_features_from_trade(ctx, score_table)
            X_real_list.append(fv)
        except Exception:
            continue

    if X_real_list:
        X_real = np.array(X_real_list, dtype=float)
        y_real = np.array(labels[:len(X_real_list)], dtype=int)
        repeat = max(3, min(10, 500 // max(len(y_real), 1)))
        X_real = np.repeat(X_real, repeat, axis=0)
        y_real = np.repeat(y_real, repeat)
    else:
        X_real = np.array([]).reshape(0, len(FEATURE_NAMES))
        y_real = np.array([], dtype=int)

    n_synth  = max(200, int(2000 * max(0, 1 - len(y_real) / 2000)))
    X_synth, y_synth = generate_synthetic_training_data(n_samples=n_synth)

    if len(y_real) > 0:
        X_all = np.vstack([X_real, X_synth])
        y_all = np.concatenate([y_real, y_synth])
    else:
        X_all, y_all = X_synth, y_synth

    X_all = np.where(np.isnan(X_all), 0.0, X_all)
    print(f"  Tanítóhalmaz: {len(y_all)} sor")

    booster = XGBoostSignalBooster()
    booster.train(X_all, y_all, run_hyperparam_search=(len(y_all) >= 100))
    return booster


# ═══════════════════════════════════════════════════════
#  FŐPROGRAM — tesztelés
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-xgb", action="store_true",
                        help="XGBoost modell betanítása")
    parser.add_argument("--demo", action="store_true",
                        help="Portfolió optimalizálás demo futtatása")
    args = parser.parse_args()

    if args.train_xgb:
        train_xgboost_from_saved_data()

    if args.demo:
        # Demo: szimulált jelzések + valós árfolyamok
        DEMO_TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN",
                        "GOOGL", "META", "NFLX"]

        # Szimulált AI jelzések (élő botban az aibotv3 adja)
        demo_signals = {t: {"bull_pct": float(np.random.uniform(35, 85))}
                        for t in DEMO_TICKERS}

        result = run_portfolio_from_bot_signals(
            signals=demo_signals,
            watchlist=DEMO_TICKERS,
            period="1y",
            total_capital=10000.0,
        )
