"""
╔══════════════════════════════════════════════════════════════════╗
║         AI LAYER — Lokális ML Elemző Modul                       ║
║         Random Forest + Score-History tábla + Teljes Riport      ║
╠══════════════════════════════════════════════════════════════════╣
║  MIT CSINÁL EZ A FÁJL?                                           ║
║                                                                  ║
║  1. TANÍTÁS (train):                                             ║
║     A backtest eredményeiből feature-öket és labeleket           ║
║     gyárt, majd betanít egy Random Forest modellt.               ║
║     Menti a modellt egy .pkl fájlba (nem kell újratanítani       ║
║     minden futásnál).                                            ║
║                                                                  ║
║  2. SCORE-HISTORY TÁBLA:                                         ║
║     Tárolja, hogy melyik bull_score értéknél historikusan        ║
║     mennyi volt a win rate részvényenként. Ez az ötleted         ║
║     kibővített, granulált változata.                             ║
║                                                                  ║
║  3. PREDIKCIÓ:                                                   ║
║     Az élő bot átadja a 9 stratégia értékeit →                   ║
║     a modell visszaad egy bullish% valószínűséget.               ║
║                                                                  ║
║  4. TELJES RIPORT generálás:                                     ║
║     - Bullish valószínűség (ML + score-history együtt)           ║
║     - Szöveges indoklás (melyik feature húzta fel/le)            ║
║     - Kockázati figyelmeztetések (volatilitás, ADX, RSI)         ║
║     - Alternatív szcenárió (mi kellene a fordulóhoz)             ║
║                                                                  ║
║  HOW TO RUN:                                                     ║
║    1. Backtest futtatása: python backtest.py                     ║
║    2. Model tanítás:      python ai_layer.py --train             ║
║    3. Élő bot:            python advanced_bot_v3.py              ║
║       (automatikusan betölti a modellt)                          ║
║                                                                  ║
║  FÜGGŐSÉGEK:                                                     ║
║    pip install scikit-learn joblib pandas numpy                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from collections import defaultdict

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import TimeSeriesSplit, cross_val_score
from sklearn.metrics           import classification_report, roc_auc_score
from sklearn.pipeline          import Pipeline
from sklearn.calibration        import CalibratedClassifierCV


# ═══════════════════════════════════════════════════════
#  KONFIGURÁCIÓ
# ═══════════════════════════════════════════════════════

MODEL_PATH        = "models/rf_model.pkl"
SCORE_HISTORY_PATH= "models/score_history.json"
FEATURE_NAMES_PATH= "models/feature_names.json"

# A trade "nyertes" ha legalább ennyit hozott (%)
WIN_THRESHOLD_PCT = 5.0

# Előretekintési ablak: ennyi napon belül kell teljesíteni a hozamot
FORWARD_DAYS = 63   # ~3 hónap

# Random Forest hiperparaméterek
RF_PARAMS = {
    "n_estimators":  300,
    "max_depth":     6,
    "min_samples_leaf": 8,    # véd az overfitting ellen kis adathalmazon
    "max_features":  "sqrt",
    "random_state":  42,
    "class_weight":  "balanced",
    "n_jobs":        -1,
}


# ═══════════════════════════════════════════════════════
#  FEATURE ENGINEERING
#  Ezeket a jellemzőket kapja meg a modell döntéshez.
#  A 9 stratégia értékein + technikai kontextuson alapul.
# ═══════════════════════════════════════════════════════

FEATURE_NAMES = [
    # ── A 9 stratégia pontszámai (0/1/2/3/4) ─────────────
    "s1_trend_score",      # Golden/Death Cross + SMA iránya
    "s2_macd_score",       # MACD histogram konszolidáció
    "s3_rsi_score",        # RSI zóna
    "s3_div_bull",         # RSI bullish divergencia (0/1)
    "s3_div_bear",         # RSI bearish divergencia (0/1)
    "s4_bb_score",         # Bollinger Band helyzet
    "s5_adx_score",        # ADX trend erősség + DI irány
    "s6_obv_score",        # OBV regresszió iránya
    "s7_rs_score",         # Relatív erő vs S&P500
    "s8_52w_score",        # 52 hetes pozíció
    "s9_vam_score",        # Volatility-Adjusted Momentum
    # ── Összesített pontszámok ────────────────────────────
    "bull_score_total",    # Teljes bull pontszám
    "bear_score_total",    # Teljes bear pontszám
    "net_score",           # bull - bear
    # ── Technikai kontextus (piac állapota) ──────────────
    "rsi_raw",             # RSI tényleges értéke (nem csak zóna)
    "adx_raw",             # ADX tényleges értéke
    "bb_width_raw",        # BB szélesség (relatív volatilitás)
    "obv_slope_raw",       # OBV regresszió meredeksége
    "vam_raw",             # VAM tényleges értéke
    # ── Score-History tábla értéke ───────────────────────
    "hist_win_rate",       # Historikus win rate ennél a bull_score-nál
    "hist_sample_count",   # Hány trade alapja a win rate (megbízhatóság)
    # ── Piaci rezsimet leíró feature-ök ──────────────────
    "above_sma200",        # 1 ha az ár az SMA200 felett van
    "sma50_slope",         # SMA50 meredeksége (emelkedő/csökkenő trend)
    "volatility_30d",      # 30 napos historikus volatilitás
]


# ═══════════════════════════════════════════════════════
#  SCORE-HISTORY TÁBLA
# ═══════════════════════════════════════════════════════

class ScoreHistoryTable:
    """
    Tárolja és lekéri, hogy egy adott bull_score értéknél
    historikusan mennyi volt a nyertes trade aránya.

    Ez az általad javasolt PHT ötlet kibővített változata:
    ahelyett hogy 2 bitbe tömörítenénk az állapotot (elveszítve
    az információt), megőrizzük a teljes bull_score-t (0-20)
    és részvényenként tároljuk a historikus win rate-et.

    Struktúra:
        {
          "NVDA": {
            "10": {"wins": 5, "total": 7},
            "11": {"wins": 3, "total": 4},
            ...
          },
          "GLOBAL": { ... }   ← összes részvény összesítve
        }
    """

    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))

    def update(self, ticker: str, bull_score: int, was_win: bool):
        """Egy lezárt trade eredményét rögzíti."""
        key = str(bull_score)
        self.table[ticker][key]["total"] += 1
        self.table["GLOBAL"][key]["total"] += 1
        if was_win:
            self.table[ticker][key]["wins"] += 1
            self.table["GLOBAL"][key]["wins"] += 1

    def get_win_rate(self, ticker: str, bull_score: int) -> tuple[float, int]:
        """
        Visszaadja a historikus win rate-et és a minta számát.
        Ha a részvénynek nincs elég adata (< 3 trade), a globális
        táblát használja fallbackként.
        """
        key = str(bull_score)

        # Részvény-specifikus adat
        ticker_data = self.table.get(ticker, {}).get(key, {})
        ticker_total = ticker_data.get("total", 0)

        if ticker_total >= 3:
            wins  = ticker_data["wins"]
            total = ticker_total
        else:
            # Fallback: globális tábla (Bayesian prior)
            global_data = self.table.get("GLOBAL", {}).get(key, {})
            wins  = global_data.get("wins",  0)
            total = global_data.get("total", 0)

        if total == 0:
            return 0.5, 0   # Nincs adat → 50% (semleges prior)

        # Laplace simítás: elkerüli a 0% és 100% extrémeket kis mintáknál
        # Formula: (wins + 1) / (total + 2)
        smoothed_rate = (wins + 1) / (total + 2)
        return smoothed_rate, total

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # defaultdict-et sima dict-be konvertáljuk JSON-hoz
        serializable = {k: dict(v) for k, v in self.table.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for ticker, scores in data.items():
            for score_key, counts in scores.items():
                self.table[ticker][score_key] = counts


# ═══════════════════════════════════════════════════════
#  FEATURE EXTRACTION — backtest trade-ből
# ═══════════════════════════════════════════════════════

def extract_features_from_trade(trade_context: dict,
                                  score_table: ScoreHistoryTable) -> np.ndarray:
    """
    Egy trade bejegyzésből (amit a backtest generál) feature vektort készít.

    A trade_context szótárnak tartalmaznia kell:
        - strategy_scores: dict a 9 stratégia pontszámával
        - bull_score, bear_score, net_score
        - rsi, adx, bb_width, obv_slope, vam
        - above_sma200, sma50_slope, volatility_30d
        - ticker
    """
    s  = trade_context.get("strategy_scores", {})
    wr, count = score_table.get_win_rate(
        trade_context.get("ticker", "GLOBAL"),
        int(trade_context.get("bull_score", 0))
    )

    features = [
        # 9 stratégia
        s.get("s1_trend",   0),
        s.get("s2_macd",    0),
        s.get("s3_rsi",     0),
        s.get("s3_div_bull",0),
        s.get("s3_div_bear",0),
        s.get("s4_bb",      0),
        s.get("s5_adx",     0),
        s.get("s6_obv",     0),
        s.get("s7_rs",      0),
        s.get("s8_52w",     0),
        s.get("s9_vam",     0),
        # Összesített
        trade_context.get("bull_score",    0),
        trade_context.get("bear_score",    0),
        trade_context.get("net_score",     0),
        # Technikai kontextus
        trade_context.get("rsi",           50.0),
        trade_context.get("adx",           20.0),
        trade_context.get("bb_width",      0.0),
        trade_context.get("obv_slope",     0.0),
        trade_context.get("vam",           0.0),
        # Score-history
        wr,
        min(count / 20.0, 1.0),   # normalizált minta szám (0-1)
        # Piac kontextus
        float(trade_context.get("above_sma200",  1)),
        trade_context.get("sma50_slope",   0.0),
        trade_context.get("volatility_30d",0.2),
    ]

    return np.array(features, dtype=float)


# ═══════════════════════════════════════════════════════
#  SZINTETIKUS TANÍTÓADAT GENERÁTOR
#  (backtest adatok hiányában vagy kiegészítésképpen)
# ═══════════════════════════════════════════════════════

def generate_synthetic_training_data(n_samples: int = 2000,
                                      seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Szintetikus tanítóadatot generál a modell hidegindításához.

    FONTOS MEGJEGYZÉS:
    ──────────────────
    Ez csak azért kell, mert a backtestből esetleg csak 80-150
    trade keletkezik — ez kevés egy robusztus modellhez.
    A szintetikus adatok a valós piac statisztikai tulajdonságait
    imitálják (nem konkrét historikus adatokat hamisítanak).

    Ahogy gyűlnek a valós backtest trade-ek, a súlyuk nő,
    a szintetikusoké csökken az ensemble-ben.

    A labelek generálásának logikája (empirikus alapon):
    - Magas bull_score + emelkedő trend + jó momentum → nagy eséllyel nyertes
    - Alacsony bull_score + gyenge trend + magas volatilitás → vesztes
    - Zaj: valós piaci bizonytalanságot szimulál
    """
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []

    for _ in range(n_samples):
        # Véletlenszerű feature értékek realisztikus eloszlással
        bull_total = rng.randint(0, 18)
        bear_total = rng.randint(0, 14)
        net        = bull_total - bear_total
        rsi        = rng.uniform(25, 80)
        adx        = rng.uniform(10, 50)
        bb_width   = rng.uniform(0.5, 8.0)
        obv_slope  = rng.uniform(-0.05, 0.05)
        vam        = rng.uniform(-1.5, 2.0)
        above_sma200 = float(rng.random() > 0.4)
        sma50_slope  = rng.uniform(-0.003, 0.003)
        vol_30d      = rng.uniform(0.1, 0.7)

        # Stratégiánkénti pontok (0-4 arányban)
        s1 = min(4, max(0, int(rng.normal(bull_total * 0.28, 0.8))))
        s2 = min(3, max(0, int(rng.normal(bull_total * 0.20, 0.6))))
        s3 = min(2, max(0, int(rng.normal(bull_total * 0.13, 0.5))))
        s4 = min(2, max(0, int(rng.normal(bull_total * 0.13, 0.5))))
        s5 = min(2, max(0, int(rng.normal(bull_total * 0.11, 0.4))))
        s6 = min(2, max(0, int(rng.normal(bull_total * 0.10, 0.4))))
        s7 = min(2, max(0, int(rng.normal(bull_total * 0.07, 0.4))))
        s8 = min(1, max(0, int(rng.normal(bull_total * 0.04, 0.3))))
        s9 = min(2, max(0, int(rng.normal(bull_total * 0.09, 0.4))))

        div_bull = int(rng.random() < 0.15)
        div_bear = int(rng.random() < 0.12)

        hist_wr    = rng.uniform(0.35, 0.75)
        hist_count = min(1.0, rng.randint(0, 25) / 20.0)

        features = [
            s1, s2, s3, div_bull, div_bear, s4, s5, s6, s7, s8, s9,
            bull_total, bear_total, net,
            rsi, adx, bb_width, obv_slope, vam,
            hist_wr, hist_count,
            above_sma200, sma50_slope, vol_30d
        ]

        # Label generálás — empirikus valószínűségi modell
        # Az egyes faktorok hozzájárulása a nyerési valószínűséghez:
        p_win = 0.42   # base rate (átlagos piaci hozam)
        p_win += net           * 0.018   # nettó pontszám hatása
        p_win += above_sma200  * 0.08    # SMA200 felett erős plusz
        p_win += (rsi - 50)    * 0.003   # RSI hatása
        p_win += obv_slope     * 3.0     # OBV trend erős hatása
        p_win += vam           * 0.06    # VAM hatása
        p_win += (hist_wr - 0.5) * 0.25  # historikus win rate hatása
        p_win += sma50_slope   * 50.0    # SMA50 meredeksége
        p_win -= vol_30d       * 0.10    # magasabb vol = több kockázat
        p_win  = max(0.05, min(0.95, p_win))   # 5%-95% közé klippelés
        # Piaci zaj szimulálása
        p_win += rng.normal(0, 0.08)
        p_win  = max(0.05, min(0.95, p_win))

        label = int(rng.random() < p_win)
        X_list.append(features)
        y_list.append(label)

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


# ═══════════════════════════════════════════════════════
#  TANÍTÓADAT BACKTEST TRADE-EKBŐL
# ═══════════════════════════════════════════════════════

def build_training_data_from_backtest(backtest_results: list[dict],
                                       score_table: ScoreHistoryTable,
                                       win_threshold: float = WIN_THRESHOLD_PCT
                                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    A backtest_results listából (amit a backtest.py generál)
    feature mátrixot és label vektort épít.

    Minden trade-hez:
    1. Kiszámolja hogy nyertes volt-e (pnl_pct > win_threshold)
    2. Frissíti a score_table-t (self-supervised update)
    3. Elkészíti a feature vektort
    """
    X_list, y_list = [], []

    for result in backtest_results:
        ticker = result.get("ticker", "UNKNOWN")
        for trade in result.get("trades", []):
            pnl_pct   = trade.get("pnl_pct", 0)
            is_win    = pnl_pct > win_threshold
            bull_score= trade.get("bull_score_at_entry", result.get("bull_score", 9))

            # Score-history tábla frissítése
            score_table.update(ticker, bull_score, is_win)

            # Trade kontextus (a backtest.py-nak kell ezt tárolnia)
            ctx = trade.get("context", {})
            if not ctx:
                continue

            ctx["ticker"] = ticker
            features = extract_features_from_trade(ctx, score_table)

            X_list.append(features)
            y_list.append(int(is_win))

    if not X_list:
        return np.array([]).reshape(0, len(FEATURE_NAMES)), np.array([])

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


# ═══════════════════════════════════════════════════════
#  MODEL BETANÍTÁS
# ═══════════════════════════════════════════════════════

def train_model(backtest_results: list[dict] = None) -> tuple:
    """
    Betanítja a modellt és elmenti a fájlrendszerbe.

    Pipeline:
    1. Backtest trade-ekből valós adatok (ha vannak)
    2. Szintetikus adatokkal kiegészítés (mindig van elég adat)
    3. Random Forest + Platt scaling (valószínűség kalibráláshoz)
    4. TimeSeriesSplit cross-validation (nem szivárog jövőbeli adat)
    5. Mentés pkl fájlba
    """
    print("\n" + "═"*55)
    print("  AI LAYER — Model Tanítás")
    print("═"*55)

    score_table = ScoreHistoryTable()

    # ── 1. Valós backtest adatok ──────────────────────────
    X_real, y_real = np.array([]).reshape(0, len(FEATURE_NAMES)), np.array([])
    if backtest_results:
        X_real, y_real = build_training_data_from_backtest(
            backtest_results, score_table
        )
        print(f"  Valós trade-ek:     {len(y_real)} db")
        if len(y_real) > 0:
            print(f"  Valós win rate:     {y_real.mean()*100:.1f}%")

    # ── 2. Szintetikus adatok ─────────────────────────────
    # Minél több valós adat van, annál kevesebb szintetikus kell
    n_synth = max(500, 2000 - len(y_real) * 10)
    X_synth, y_synth = generate_synthetic_training_data(n_samples=n_synth)
    print(f"  Szintetikus adatok: {n_synth} db")

    # ── 3. Összefűzés ─────────────────────────────────────
    if len(y_real) > 0:
        # A valós adatok 3x akkora súlyt kapnak mint a szintetikusak
        X_real_rep = np.repeat(X_real, 3, axis=0)
        y_real_rep = np.repeat(y_real, 3)
        X_all = np.vstack([X_real_rep, X_synth])
        y_all = np.concatenate([y_real_rep, y_synth])
    else:
        X_all, y_all = X_synth, y_synth

    print(f"  Teljes tanítóhalmaz: {len(y_all)} db  "
          f"(win rate: {y_all.mean()*100:.1f}%)")

    # ── 4. Pipeline felépítés ─────────────────────────────
    # StandardScaler: normalizálja a feature-öket (fontos a Logistic Regression-nek)
    # CalibratedClassifierCV: a Random Forest valószínűségeit kalibrálja
    #   → a "73% bullish" valóban azt jelenti, hogy 73% az esélye
    base_model = RandomForestClassifier(**RF_PARAMS)
    pipeline   = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  CalibratedClassifierCV(base_model, cv=3, method="sigmoid"))
    ])

    # ── 5. TimeSeriesSplit cross-validation ───────────────
    # FONTOS: sima k-fold nem használható idősoroknál!
    # A TimeSeriesSplit mindig csak múltbeli adatot lát tanításkor.
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X_all, y_all,
                             cv=tscv, scoring="roc_auc", n_jobs=-1)
    print(f"\n  Cross-validation ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  (0.5 = véletlen találgatás, 1.0 = tökéletes)")

    # ── 6. Végleges tanítás ───────────────────────────────
    pipeline.fit(X_all, y_all)

    # Feature importancia (a kalibrált modellből kibontva)
    try:
        rf = pipeline.named_steps["model"].calibrated_classifiers_[0].estimator
        importances = rf.feature_importances_
        top_features = sorted(zip(FEATURE_NAMES, importances),
                               key=lambda x: x[1], reverse=True)[:8]
        print("\n  Top 8 legfontosabb feature:")
        for fname, imp in top_features:
            bar = "█" * int(imp * 100)
            print(f"    {fname:25s} {imp:.3f} {bar}")
    except Exception:
        pass

    # ── 7. Mentés ─────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    score_table.save(SCORE_HISTORY_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f)

    print(f"\n  ✅ Modell elmentve: {MODEL_PATH}")
    print(f"  ✅ Score-history:   {SCORE_HISTORY_PATH}")
    print("═"*55 + "\n")

    return pipeline, score_table


# ═══════════════════════════════════════════════════════
#  AI ELEMZŐ — ÉLŐBEN HASZNÁLT OSZTÁLY
# ═══════════════════════════════════════════════════════

class AIAnalyzer:
    """
    Ezt az osztályt használja az advanced_bot_v3.py.
    Betölti a modellt és a score-history táblát,
    majd minden jelzéshez teljes riportot generál.
    """

    def __init__(self):
        self.model       = None
        self.score_table = ScoreHistoryTable()
        self._load()

    def _load(self):
        """Betölti a modellt és a score-history táblát."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("  🤖 AI modell betöltve.")
        else:
            print("  ⚠️  AI modell nem található. "
                  "Futtasd: python ai_layer.py --train")

        if os.path.exists(SCORE_HISTORY_PATH):
            self.score_table.load(SCORE_HISTORY_PATH)
            print("  📊 Score-history tábla betöltve.")

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, context: dict) -> dict:
        """
        Egy részvény aktuális állapotából teljes AI riportot generál.

        context: ugyanaz a dict amit az advanced_bot_v3.py analyze()
                 függvénye visszaad, kiegészítve a strategy_scores-szal.

        Visszatér: riport dict (ld. lentebb)
        """
        if not self.is_ready():
            return self._fallback_report(context)

        features = extract_features_from_trade(context, self.score_table)

        # NaN csere 0-ra (robusztusság)
        features = np.where(np.isnan(features), 0.0, features)
        features = features.reshape(1, -1)

        # ML valószínűség
        proba      = self.model.predict_proba(features)[0]
        ml_bull_p  = float(proba[1])

        # Score-history win rate
        hist_wr, hist_count = self.score_table.get_win_rate(
            context.get("ticker", "GLOBAL"),
            int(context.get("bull_score", 0))
        )

        # ── Ensemble: ML + score-history súlyozva ─────────
        # Ha sok historikus adat van (> 10 trade), annak nagyobb
        # súlyt adunk, mert részvény-specifikus viselkedést tükröz.
        hist_weight = min(hist_count * 20, 0.35)   # max 35% súly
        ml_weight   = 1.0 - hist_weight
        final_p     = ml_weight * ml_bull_p + hist_weight * hist_wr

        # ── Teljes riport összeállítása ────────────────────
        return self._build_report(context, final_p, ml_bull_p, hist_wr,
                                   hist_count, features)

    def _build_report(self, ctx: dict, final_p: float, ml_p: float,
                       hist_wr: float, hist_count: int,
                       features: np.ndarray) -> dict:
        """
        Összerakja a teljes szöveges + numerikus riportot.
        """
        ticker     = ctx.get("ticker", "?")
        bull_score = ctx.get("bull_score", 0)
        bear_score = ctx.get("bear_score", 0)
        rsi        = ctx.get("rsi", 50)
        adx        = ctx.get("adx", 20)
        vol        = ctx.get("volatility_30d", 0.2)
        vam        = ctx.get("vam", 0)
        net        = ctx.get("net_score", 0)

        bull_pct   = round(final_p * 100, 1)
        bear_pct   = round((1 - final_p) * 100, 1)

        # ── Konfidencia szint ──────────────────────────────
        if hist_count >= 10:
            data_quality = f"Magas (historikus adat: {int(hist_count*20)} trade)"
        elif hist_count >= 3:
            data_quality = f"Közepes (historikus adat: {int(hist_count*20)} trade)"
        else:
            data_quality = "Alacsony — szintetikus prior alapján"

        # ── Fő irány és erősség ────────────────────────────
        if bull_pct >= 75:
            direction_label = "ERŐSEN BULLISH"
            direction_emoji = "🟢🟢"
        elif bull_pct >= 60:
            direction_label = "BULLISH"
            direction_emoji = "🟢"
        elif bull_pct >= 45:
            direction_label = "ENYHÉN BULLISH"
            direction_emoji = "🟡"
        elif bull_pct >= 35:
            direction_label = "ENYHÉN BEARISH"
            direction_emoji = "🟠"
        elif bull_pct >= 25:
            direction_label = "BEARISH"
            direction_emoji = "🔴"
        else:
            direction_label = "ERŐSEN BEARISH"
            direction_emoji = "🔴🔴"

        # ── Szöveges indoklás ─────────────────────────────
        explanations = []
        s = ctx.get("strategy_scores", {})

        if s.get("s1_trend", 0) >= 3:
            explanations.append("a trend struktúra erősen bullish (Golden Cross zóna)")
        elif s.get("s1_trend", 0) <= 0:
            explanations.append("a fő trend bearish (SMA struktúra negatív)")

        if s.get("s2_macd", 0) >= 2:
            explanations.append("MACD momentum megerősített és gyorsuló")
        elif s.get("s2_macd", 0) < 0:
            explanations.append("MACD momentum negatív irányban gyorsul")

        if rsi > 60:
            explanations.append(f"RSI erős bullish zónában ({rsi:.0f})")
        elif rsi < 40:
            explanations.append(f"RSI gyenge területen ({rsi:.0f}), de potenciális visszapattanás")

        if s.get("s6_obv", 0) >= 2:
            explanations.append("intézményi akkumuláció jelei látszanak (OBV trend)")

        if vam > 0.5:
            explanations.append(f"a részvény kockázat-arányos hozama kiemelkedő (VAM={vam:.2f})")

        if hist_wr > 0.65 and int(hist_count*20) >= 5:
            explanations.append(
                f"historikusan {hist_wr*100:.0f}%-os win rate "
                f"ennél a score szintnél ({int(hist_count*20)} trade alapján)"
            )

        if not explanations:
            explanations.append("a jelek vegyesek, nincs egyértelmű domináns faktor")

        explanation_text = " és ".join(explanations[:3])

        # ── Kockázati figyelmeztetések ─────────────────────
        risks = []
        if adx > 40:
            risks.append(
                f"⚠️  ADX={adx:.0f} — a trend érett, fordulat közeledhet"
            )
        if rsi > 72:
            risks.append(
                f"⚠️  RSI={rsi:.0f} — túlvett állapot, korrekció lehetséges"
            )
        if vol > 0.45:
            risks.append(
                f"⚠️  Magas volatilitás ({vol*100:.0f}% éves) — "
                f"a stop-loss távolabb kell legyen"
            )
        if bear_score >= 6:
            risks.append(
                f"⚠️  Erős bearish ellennyomás (bear_score={bear_score}) — "
                f"érdemes kisebb pozíciómérettel belépni"
            )
        if hist_count < 3:
            risks.append(
                "⚠️  Kevés historikus adat ehhez a jelzéstípushoz — "
                "az ML becslés szintetikus adaton alapul"
            )
        if not risks:
            risks.append("✅ Nem azonosítottak jelentős kockázati jelzőket")

        # ── Alternatív szcenárió ───────────────────────────
        if bull_pct >= 60:
            # Bullish irányban vagyunk — mi rontaná el?
            alt_scenario = self._build_bearish_scenario(ctx)
        else:
            # Bearish irányban — mi változtatná meg a képet?
            alt_scenario = self._build_bullish_scenario(ctx)

        # ── Pozícióméret javaslat ──────────────────────────
        # Kelly-kritérium közelítés: f = (p*b - q) / b
        # ahol p = win_p, q = loss_p, b = átlagos nyerés/veszteség arány
        kelly_b = 1.5   # 3:2 arány feltételezve (konzervatív)
        kelly_f = (final_p * kelly_b - (1 - final_p)) / kelly_b
        kelly_f = max(0, min(0.25, kelly_f))   # max 25% pozícióméret
        position_pct = round(kelly_f * 100, 1)

        return {
            "ticker":           ticker,
            "bull_pct":         bull_pct,
            "bear_pct":         bear_pct,
            "direction_label":  direction_label,
            "direction_emoji":  direction_emoji,
            "ml_probability":   round(ml_p * 100, 1),
            "hist_win_rate":    round(hist_wr * 100, 1),
            "data_quality":     data_quality,
            "explanation":      explanation_text,
            "risks":            risks,
            "alt_scenario":     alt_scenario,
            "position_size_pct":position_pct,
        }

    def _build_bearish_scenario(self, ctx: dict) -> str:
        """Mi kellene ahhoz, hogy a bullish kép bearishre forduljon?"""
        lines = ["Ha az alábbiak bekövetkeznek, a bullish kép érvényét vesztheti:"]
        rsi = ctx.get("rsi", 50)
        adx = ctx.get("adx", 20)

        if rsi < 65:
            lines.append("  • RSI átlépi a 70-et és divergencia jelenik meg")
        lines.append("  • Az árfolyam visszaesik az SMA50 alá és ott zár")
        if adx < 30:
            lines.append("  • ADX csökken 20 alá (trend elvesztése)")
        lines.append("  • OBV 3 egymást követő napon csökken magas volumen mellett")
        return "\n".join(lines)

    def _build_bullish_scenario(self, ctx: dict) -> str:
        """Mi kellene ahhoz, hogy a bearish kép bullishre forduljon?"""
        lines = ["A következők javítanák a bullish kilátásokat:"]
        rsi = ctx.get("rsi", 50)

        lines.append("  • Az árfolyam visszatér az SMA50 fölé és ott konszolidál")
        if rsi < 50:
            lines.append(f"  • RSI visszaemelkedik 50 fölé ({rsi:.0f} → 50+)")
        lines.append("  • MACD histogram 3 egymást követő napig pozitív")
        lines.append("  • OBV emelkedő trendet mutat növekvő volumennel")
        return "\n".join(lines)

    def _fallback_report(self, ctx: dict) -> dict:
        """Ha nincs betanított modell, score-alapú egyszerű becslés."""
        bull  = ctx.get("bull_score", 0)
        bear  = ctx.get("bear_score", 0)
        total = bull + bear or 1
        p     = bull / total
        return {
            "ticker":           ctx.get("ticker", "?"),
            "bull_pct":         round(p * 100, 1),
            "bear_pct":         round((1-p) * 100, 1),
            "direction_label":  "BULLISH" if p > 0.5 else "BEARISH",
            "direction_emoji":  "🟡",
            "ml_probability":   round(p * 100, 1),
            "hist_win_rate":    50.0,
            "data_quality":     "Nincs betanított modell — score arány alapján",
            "explanation":      f"bull_score={bull}, bear_score={bear}",
            "risks":            ["⚠️ Modell nem elérhető, futtasd: python ai_layer.py --train"],
            "alt_scenario":     "N/A",
            "position_size_pct":5.0,
        }

    def format_discord_report(self, report: dict) -> str:
        """Discord-ra formatált teljes AI riport."""
        risks_text = "\n".join(report["risks"])
        return (
            f"\n🤖 **AI ELEMZÉS — {report['ticker']}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{report['direction_emoji']} **{report['direction_label']}**\n"
            f"**Bullish valószínűség: `{report['bull_pct']}%`** "
            f"| Bearish: `{report['bear_pct']}%`\n"
            f"_(ML: {report['ml_probability']}% | "
            f"Historikus win rate: {report['hist_win_rate']}% | "
            f"Adat: {report['data_quality']})_\n\n"
            f"**📝 Indoklás:**\n"
            f"A modell szerint {report['explanation']}.\n\n"
            f"**⚠️ Kockázati figyelmeztetések:**\n"
            f"```\n{risks_text}\n```\n"
            f"**🔄 Alternatív szcenárió:**\n"
            f"```\n{report['alt_scenario']}\n```\n"
            f"**💼 Javasolt pozícióméret:** `{report['position_size_pct']}%` "
            f"_(Kelly-kritérium, konzervatív)_\n"
        )


# ═══════════════════════════════════════════════════════
#  PARANCSSORI FUTTATÁS
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Layer — Model Tanítás")
    parser.add_argument("--train", action="store_true",
                        help="Modell betanítása (backtest nélkül: szintetikus adaton)")
    parser.add_argument("--test",  action="store_true",
                        help="Teszt predikció futtatása")
    args = parser.parse_args()

    if args.train:
        pipeline, score_table = train_model(backtest_results=None)
        print("✅ Tanítás kész! Az advanced_bot_v3.py mostantól használhatja.")

    if args.test:
        analyzer = AIAnalyzer()
        if analyzer.is_ready():
            # Teszt kontextus
            test_ctx = {
                "ticker": "NVDA", "bull_score": 12, "bear_score": 3,
                "net_score": 9,   "rsi": 61.5,      "adx": 32.1,
                "bb_width": 2.8,  "obv_slope": 0.012,"vam": 0.78,
                "above_sma200": 1,"sma50_slope": 0.002,"volatility_30d": 0.35,
                "strategy_scores": {
                    "s1_trend": 3, "s2_macd": 2, "s3_rsi": 2,
                    "s3_div_bull": 1, "s3_div_bear": 0,
                    "s4_bb": 1, "s5_adx": 2, "s6_obv": 2,
                    "s7_rs": 2, "s8_52w": 1, "s9_vam": 2,
                }
            }
            report = analyzer.predict(test_ctx)
            print("\n=== TESZT RIPORT ===")
            print(f"Részvény:         {report['ticker']}")
            print(f"Irány:            {report['direction_emoji']} {report['direction_label']}")
            print(f"Bullish valószínűség: {report['bull_pct']}%")
            print(f"ML:               {report['ml_probability']}%")
            print(f"Hist. win rate:   {report['hist_win_rate']}%")
            print(f"Indoklás:         {report['explanation']}")
            print(f"Pozícióméret:     {report['position_size_pct']}%")
            print(f"\nKockázatok:")
            for r in report["risks"]: print(f"  {r}")
            print(f"\nAlternatív szcenárió:\n{report['alt_scenario']}")
