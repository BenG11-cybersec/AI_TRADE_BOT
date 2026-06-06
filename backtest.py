"""
╔══════════════════════════════════════════════════════════════════╗
║         BACKTEST v2.0 — advanced_bot_v3 + AI Layer               ║
║         3 éves historikus teszt, 10 részvény, $1000/db           ║
╠══════════════════════════════════════════════════════════════════╣
║  MIT ÚJ A v2.0-BAN?                                              ║
║                                                                  ║
║  1. Az advanced_bot_v3 ÖSSZES stratégiáját teszteli (S1–S9)      ║
║  2. Az AI layer is fut: minden trade-nél bullish%-ot számol      ║
║  3. AI-SZŰRT mód: csak akkor vesz, ha AI bullish% >= küszöb      ║
║  4. Párhuzamos összehasonlítás:                                  ║
║       • Csak Quant stratégia (v3, AI nélkül)                     ║
║       • Quant + AI szűrő (csak magas AI konfidenciánál vesz)     ║
║       • Buy & Hold                                               ║
║  5. AI modell önfejlesztése: minden lezárt trade visszakerül     ║
║     a score-history táblába → a modell tanul a backtestből       ║
║                                                                  ║
║  HOW TO RUN:                                                     ║
║    1. python ai_layer.py --train   (első alkalommal)             ║
║    2. python backtest_v2.py                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

# Saját modulok
from ai_layer import AIAnalyzer, ScoreHistoryTable, FEATURE_NAMES, save_trade_data
from aibotv3 import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    _aligned_relative_strength,
    _obv_slope,
)


# ═══════════════════════════════════════════════════════
#  KONFIGURÁCIÓ
# ═══════════════════════════════════════════════════════

WATCHLIST = [
    # Tech & AI
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "PANW", "DELL",
    # Pénzügy
    "JPM", "V", "MA", "BAC", "GS",
    # Egészségügy
    "JNJ", "UNH", "LLY", "PFE", "MRK",
    # Ipar & Védelem
    "LMT", "RTX", "NOC", "BA", "CAT", "RHM.DE",
    # Egyéb befektetett cégek
    "NOW", "CVS", "VCEL", "AXON", "MC", "AIR", "COHR",
    "BLK", "BYD", "LCID", "DOCS", "S", "VST", "KRNT",
]

CAPITAL_PER_STOCK = 1000.0
BACKTEST_PERIOD   = "7y"
COMMISSION_PCT    = 0.001

BUY_THRESHOLD     = 5    # felülírja az importált értéket

# AI szűrő: csak akkor vesz, ha az AI bullish valószínűsége legalább ennyi
AI_BUY_FILTER_PCT  = 60.0
AI_SELL_FILTER_PCT = 55.0

WIN_THRESHOLD_PCT  = 5.0    # trade "nyertes" ha > 5% hozam


# ═══════════════════════════════════════════════════════
#  STANDALONE FÜGGVÉNYEK — importálva a v3_ai-ból
#  (Ezeket a backtest közvetlenül hívja naponként)
# ═══════════════════════════════════════════════════════

def _get(row, key, default):
    try:
        v = float(row.get(key, default))
        return default if (v != v) else v
    except:
        return default


def build_context(ticker, curr, prev, data_slice, benchmark_slice,
                  bull_score, bear_score, net_score, reasons,
                  s_scores, vam_val):
    """
    Összerakja azt a context dict-et amit az AIAnalyzer.predict() vár.
    """
    close   = _get(curr, "Close", 0)
    rsi     = _get(curr, "RSI_14", 50)
    adx     = _get(curr, "ADX_14", 20)
    sma50   = _get(curr, "SMA_50",  close)
    sma200  = _get(curr, "SMA_200", close)
    bb_w    = _get(curr, "BBB_20_2.0", 0)

    # OBV slope
    try:
        obv_slope_val = float(_obv_slope(data_slice["OBV"]))
    except:
        obv_slope_val = 0.0

    # 30 napos vol
    try:
        ret30  = data_slice["Close"].pct_change().tail(30).dropna()
        vol30  = float(ret30.std() * np.sqrt(252)) if len(ret30) >= 10 else 0.25
    except:
        vol30  = 0.25

    # SMA50 meredeksége
    try:
        sma50_series = data_slice["SMA_50"].dropna()
        sma50_slope  = float(
            (sma50_series.iloc[-1] - sma50_series.iloc[-6]) /
            (abs(sma50_series.iloc[-6]) + 1e-9)
        ) if len(sma50_series) >= 6 else 0.0
    except:
        sma50_slope = 0.0

    return {
        "ticker":         ticker,
        "bull_score":     bull_score,
        "bear_score":     bear_score,
        "net_score":      net_score,
        "rsi":            rsi,
        "adx":            adx,
        "bb_width":       bb_w,
        "obv_slope":      obv_slope_val,
        "vam":            vam_val,
        "above_sma200":   int(close > sma200),
        "sma50_slope":    sma50_slope,
        "volatility_30d": vol30,
        "strategy_scores": s_scores,
        "reasons":        reasons,
    }


# ═══════════════════════════════════════════════════════
#  INDIKÁTOR SZÁMÍTÁS (backtest-specifikus, önálló)
# ═══════════════════════════════════════════════════════

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.ta.sma(length=50,  append=True)
    data.ta.sma(length=200, append=True)
    data.ta.rsi(length=14,  append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    data.ta.bbands(length=20, std=2, append=True)
    data.ta.adx(length=14,  append=True)
    data.ta.obv(append=True)
    data["Vol_SMA20"] = data["Volume"].rolling(window=20).mean()

    bb_map = {
        "BBL": "BBL_20_2.0", "BBM": "BBM_20_2.0",
        "BBU": "BBU_20_2.0", "BBB": "BBB_20_2.0", "BBP": "BBP_20_2.0",
    }
    rename = {}
    for col in data.columns:
        for prefix, std_name in bb_map.items():
            if col.startswith(prefix + "_20_2.0") and col != std_name:
                rename[col] = std_name
    if rename:
        data.rename(columns=rename, inplace=True)
    return data


# ═══════════════════════════════════════════════════════
#  SCORING MOTOR (v3 logika, backtest-specifikus)
# ═══════════════════════════════════════════════════════

def score_day(curr, prev, data_tail, benchmark_tail) -> tuple[int, int, dict, float]:
    """
    Egy napra kiszámolja a bull/bear score-t + stratégia részpontszámokat.
    Visszatér: (bull_score, bear_score, strategy_scores_dict, vam_value)

    Ez a v3 stratégiákat hűen reprodukálja a backtest számára,
    de részletesebb kimenetet ad (per-stratégia pontszámok az AI-hoz).
    """
    def g(row, key, default=0.0):
        try:
            v = float(row.get(key, default))
            return default if (v != v) else v
        except:
            return default

    close       = g(curr, "Close")
    rsi         = g(curr, "RSI_14",          50.0)
    sma50       = g(curr, "SMA_50",          close)
    sma200      = g(curr, "SMA_200",         close)
    prev_sma50  = g(prev, "SMA_50",          sma50)
    prev_sma200 = g(prev, "SMA_200",         sma200)
    bb_upper    = g(curr, "BBU_20_2.0",      close)
    bb_lower    = g(curr, "BBL_20_2.0",      close)
    bb_mid      = g(curr, "BBM_20_2.0",      close)
    bb_width    = g(curr, "BBB_20_2.0",      0)
    prev_bb_mid = g(prev, "BBM_20_2.0",      bb_mid)
    adx         = g(curr, "ADX_14",          20.0)
    dmp         = g(curr, "DMP_14",          0.0)
    dmn         = g(curr, "DMN_14",          0.0)
    volume      = g(curr, "Volume",          0)
    avg_volume  = g(curr, "Vol_SMA20",       volume)

    bull = 0
    bear = 0
    s    = {}   # per-stratégia pontszámok

    # ── S1: Trend / Golden-Death Cross ───────────────────
    golden = prev_sma50 < prev_sma200 and sma50 > sma200
    death  = prev_sma50 > prev_sma200 and sma50 < sma200
    if golden:
        bull += 4; s["s1_trend"] = 4
    elif death:
        bear += 4; s["s1_trend"] = -4
    elif sma50 > sma200:
        pts = 2 if sma200 > prev_sma200 else 1
        bull += pts; s["s1_trend"] = pts
    else:
        pts = 2 if sma200 < prev_sma200 else 1
        bear += pts; s["s1_trend"] = -pts

    # ── S2: MACD Histogram 3 napos konszolidáció ─────────
    macd_h_series = data_tail["MACDh_12_26_9"].tail(5).dropna()
    s2_pts = 0
    if len(macd_h_series) >= 4:
        last4 = macd_h_series.values[-4:]
        if last4[0] < 0 and last4[1] < 0 and last4[2] > 0 and last4[3] > last4[2] > 0:
            bull += 3; s2_pts = 3
        elif last4[0] > 0 and last4[1] > 0 and last4[2] < 0 and last4[3] < last4[2] < 0:
            bear += 3; s2_pts = -3
        elif all(macd_h_series.values[-3:] > 0) and \
             macd_h_series.values[-1] > macd_h_series.values[-2] > macd_h_series.values[-3]:
            bull += 2; s2_pts = 2
        elif all(macd_h_series.values[-3:] < 0) and \
             macd_h_series.values[-1] < macd_h_series.values[-2] < macd_h_series.values[-3]:
            bear += 2; s2_pts = -2
        elif len(macd_h_series) and macd_h_series.values[-1] > 0:
            bull += 1; s2_pts = 1
        elif len(macd_h_series) and macd_h_series.values[-1] < 0:
            bear += 1; s2_pts = -1
    s["s2_macd"] = s2_pts

    # ── S3: RSI zóna + divergencia ────────────────────────
    s3_pts = 0
    if rsi > 55:   bull += 2; s3_pts = 2
    elif rsi > 50: bull += 1; s3_pts = 1
    elif rsi < 45: bear += 2; s3_pts = -2
    elif rsi < 50: bear += 1; s3_pts = -1
    s["s3_rsi"] = s3_pts

    div_bull = div_bear = 0
    rsi_w = data_tail["RSI_14"].dropna()
    px_w  = data_tail["Close"]
    if len(rsi_w) >= 10:
        px_std = px_w.std()
        if close <= px_w.min() + px_std * 0.05 and rsi > float(rsi_w.min()) + 5:
            bull += 2; div_bull = 1
        if close >= px_w.max() - px_std * 0.05 and rsi < float(rsi_w.max()) - 5:
            bear += 2; div_bear = 1
    s["s3_div_bull"] = div_bull
    s["s3_div_bear"] = div_bear

    # ── S4: Bollinger Bands ───────────────────────────────
    s4_pts = 0
    bb_col = "BBB_20_2.0"
    if bb_col in data_tail.columns:
        bb_s = data_tail[bb_col].dropna()
        if len(bb_s) >= 30:
            rolling_low = float(bb_s.quantile(0.20))
            bb_squeeze  = bb_width < rolling_low and bb_width > 0
        else:
            bb_squeeze = False
    else:
        bb_squeeze = False

    if close > bb_upper:
        pts = 2 if adx > 25 else 1; bull += pts; s4_pts = pts
    elif close < bb_lower:
        if close > float(prev["Close"]):
            bull += 2; s4_pts = 2
        else:
            bear += 1; s4_pts = -1
    elif close > bb_mid and float(prev["Close"]) < prev_bb_mid:
        bull += 1; s4_pts = 1
    s["s4_bb"] = s4_pts

    # ── S5: ADX + DI irány ────────────────────────────────
    s5_pts = 0
    if adx > 25 and adx <= 40:
        if dmp > dmn:
            pts = 2 if dmp - dmn > 10 else 1; bull += pts; s5_pts = pts
        elif dmn > dmp:
            pts = 2 if dmn - dmp > 10 else 1; bear += pts; s5_pts = -pts
    s["s5_adx"] = s5_pts

    # ── S6: OBV regresszió ────────────────────────────────
    s6_pts = 0
    try:
        obv_slope_val = float(_obv_slope(data_tail["OBV"]))
        if obv_slope_val > 0.005:
            pts = 2 if bull > bear else 1; bull += pts; s6_pts = pts
        elif obv_slope_val < -0.005:
            pts = 2 if bear > bull else 1; bear += pts; s6_pts = -pts
    except:
        obv_slope_val = 0.0
    s["s6_obv"] = s6_pts

    # ── S7: Relatív erő vs benchmark ─────────────────────
    s7_pts = 0
    try:
        rs_score, _ = _aligned_relative_strength(data_tail, benchmark_tail)
        if rs_score > 0:   bull += rs_score; s7_pts = rs_score
        elif rs_score < 0: bear += abs(rs_score); s7_pts = rs_score
    except:
        pass
    s["s7_rs"] = s7_pts

    # ── S8: 52 hetes pozíció ──────────────────────────────
    s8_pts = 0
    try:
        one_yr_ago = data_tail.index[-1] - timedelta(days=365)
        idx_tz = data_tail.index.tz_localize(None) if data_tail.index.tz else data_tail.index
        data_52w = data_tail[idx_tz >= one_yr_ago]
        if len(data_52w) >= 20:
            h52 = float(data_52w["Close"].max())
            l52 = float(data_52w["Close"].min())
            if h52 > 0 and (close - h52) / h52 * 100 > -10:
                bull += 1; s8_pts = 1
            elif l52 > 0 and (close - l52) / l52 * 100 < 20:
                bear += 1; s8_pts = -1
    except:
        pass
    s["s8_52w"] = s8_pts

    # ── S9: VAM ───────────────────────────────────────────
    s9_pts = 0
    vam_val = 0.0
    try:
        # FONTOS: data_tail.iloc[0]-t használunk, nem iloc[-63]-t.
        # Ha data_tail 63 napos szelet, az iloc[-63] == iloc[0].
        # Ha rövidebb (periódus eleje), az iloc[-63] negatív irányból
        # indexelne → rossz értéket adna. Az iloc[0] mindig biztonságos.
        ret63 = data_tail["Close"].pct_change().dropna()
        if len(data_tail) >= 63 and len(ret63) >= 30:
            pr = float(data_tail["Close"].iloc[-1] / data_tail["Close"].iloc[0] - 1)
            av = float(ret63.std() * np.sqrt(252))
            if av > 0.01:
                vam_val = pr / av
                if vam_val > 0.6:    bull += 2; s9_pts = 2
                elif vam_val > 0.25: bull += 1; s9_pts = 1
                elif vam_val < -0.6: bear += 2; s9_pts = -2
                elif vam_val < -0.25: bear += 1; s9_pts = -1
    except Exception:
        pass
    s["s9_vam"] = s9_pts

    return bull, bear, s, vam_val


# ═══════════════════════════════════════════════════════
#  BACKTEST MOTOR
# ═══════════════════════════════════════════════════════

def backtest_ticker(ticker: str, data: pd.DataFrame,
                    benchmark: pd.DataFrame,
                    capital: float,
                    ai_analyzer: AIAnalyzer,
                    score_table: ScoreHistoryTable) -> dict:
    """
    Lefuttatja a v3 stratégiát + AI szűrőt egy részvényre.
    """
    if data.empty or len(data) < 220:
        return None

    # 1. INDIKÁTOROK KISZÁMÍTÁSA
    data = calculate_indicators(data)
    
    # 2. IDŐZÓNÁK EGYSZERI ELDOBÁSA ÉS BENCHMARK ILLESZTÉS (A GYORSÍTÁS LELKE!)
    # Ezt a ciklus előtt csináljuk meg egyszer, hogy ne kelljen iterációnként searchsorted-et hívni.
    data.index = data.index.tz_localize(None)
    if not benchmark.empty:
        benchmark.index = benchmark.index.tz_localize(None)
        # Az asof merge összeköti az S&P500 záróárát a részvény dátumával, kitöltve a hiányokat.
        data = pd.merge_asof(data, benchmark[['Close']].rename(columns={'Close': 'SPY_Close'}), 
                             left_index=True, right_index=True, direction='backward')
    else:
        data['SPY_Close'] = np.nan

    data = data.dropna(subset=["SMA_200"])

    quant = {"cash": capital, "position": None, "trades": [], "equity_curve": []}
    ai_f  = {"cash": capital, "position": None, "trades": [], "equity_curve": []}

    # Cash a változók eléréséhez (hogy ne kelljen a Pandas Series-ben lassan keresgélni)
    close_vals = data["Close"].values
    open_vals = data["Open"].values
    dates = data.index

    for i in range(1, len(data) - 1):
        curr_row  = data.iloc[i]
        prev_row  = data.iloc[i - 1]
        date      = dates[i]
        
        # Gyorsabb C-szintű tömb elérés (NumPy) a Pandas Series helyett ahol lehet
        close     = float(close_vals[i])
        next_open = float(open_vals[i + 1])

        # Adatszeletek (Ezt sajnos muszáj meghagyni a score_day kompatibilitás miatt, de a benchmark keresést megúsztuk!)
        tail_w    = data.iloc[max(0, i - 63): i + 1]
        
        # A bench_sub helyett most már csak a tail_w-t használjuk, mert abban benne van az 'SPY_Close'
        bench_sub = pd.DataFrame()
        if 'SPY_Close' in data.columns and not pd.isna(data['SPY_Close'].iloc[0]):
             bench_sub = pd.DataFrame({'Close': tail_w['SPY_Close'].values}, index=tail_w.index)

        # Pontszámítás
        bull, bear, s_scores, vam_val = score_day(
            curr_row, prev_row, tail_w, bench_sub
        )
        net = bull - bear

        # AI context
        ctx = build_context(
            ticker, curr_row, prev_row, tail_w, bench_sub,
            bull, bear, net, [], s_scores, vam_val
        )

        ai_bull_pct = 50.0
        if ai_analyzer and ai_analyzer.is_ready():
            try:
                report      = ai_analyzer.predict(ctx)
                ai_bull_pct = report.get("bull_pct", 50.0)
            except:
                pass

        buy_signal  = bull >= BUY_THRESHOLD
        sell_signal = bear >= SELL_THRESHOLD

        _apply_signals(quant, buy_signal, sell_signal,
                       date, next_open, capital, bull, ctx, score_table, ticker,
                       update_score_table=(not ai_f["position"])) 

        ai_buy  = buy_signal  and ai_bull_pct >= AI_BUY_FILTER_PCT
        ai_sell = sell_signal and ai_bull_pct <  AI_SELL_FILTER_PCT
        _apply_signals(ai_f, ai_buy, ai_sell,
                       date, next_open, capital, bull, ctx, None, ticker,
                       update_score_table=False)

        for port in (quant, ai_f):
            eq = port["position"]["shares"] * close if port["position"] else port["cash"]
            port["equity_curve"].append({"date": date, "equity": eq})

    final_price = float(close_vals[-1])
    for port in (quant, ai_f):
        if port["position"]:
            comm     = final_price * COMMISSION_PCT
            proceeds = port["position"]["shares"] * (final_price - comm)
            port["trades"].append({
                "entry_date":  port["position"]["entry_date"],
                "exit_date":   dates[-1],
                "entry_price": port["position"]["entry_price"],
                "exit_price":  final_price,
                "pnl_usd":     proceeds - capital,
                "pnl_pct":     (proceeds / capital - 1) * 100,
                "holding_days":(dates[-1] - port["position"]["entry_date"]).days,
                "still_open":  True,
                "bull_score":  bull,
            })
            port["cash"] = proceeds

    bh_shares = capital / float(close_vals[0])
    bh_final  = bh_shares * final_price * (1 - COMMISSION_PCT)
    bh_pct    = (bh_final / capital - 1) * 100

    def _stats(port):
        trades  = port["trades"]
        wins    = [t for t in trades if t["pnl_usd"] > 0]
        return {
            "final_value": port["cash"],
            "total_pct":   (port["cash"] / capital - 1) * 100,
            "num_trades":  len(trades),
            "win_rate":    len(wins) / len(trades) * 100 if trades else 0,
            "trades":      trades,
            "equity_curve":port["equity_curve"],
        }

    return {
        "ticker":       ticker,
        "capital":      capital,
        "quant":        _stats(quant),
        "ai_filtered":  _stats(ai_f),
        "bh_final":     bh_final,
        "bh_pct":       bh_pct,
        "start_price":  float(close_vals[0]),
        "end_price":    final_price,
    }


def _apply_signals(port, buy_signal, sell_signal,
                   date, next_open, capital, bull_score,
                   ctx, score_table, ticker,
                   update_score_table=False):
    """
    Segédfüggvény: egy portfólión alkalmazza a vétel/eladás logikát.

    JAVÍTÁS — Entry vs Exit score:
    ───────────────────────────────
    A score_table-ba és a trade rekordba az ENTRY pillanatában
    mért bull_score kerül, nem a kilépéskor aktuális érték.
    Miért fontos: a Random Forest azt tanulja meg, hogy melyik
    BELÉPÉSI score-nál milyen az eredmény. Ha a kilépési score-t
    mentenénk, az összekeverné a kauzalitást — a modell olyan
    adatból tanulna, ami a döntés UTÁN keletkezett.
    """
    if port["position"] is None and buy_signal:
        comm   = next_open * COMMISSION_PCT
        shares = port["cash"] / (next_open + comm)
        port["cash"] = 0.0
        port["position"] = {
            "entry_date":  date,
            "entry_price": next_open,
            "shares":      shares,
            "bull_score":  bull_score,   # ← ENTRY score mentése
            "context":     ctx,          # ← ENTRY context mentése
        }

    elif port["position"] is not None and sell_signal:
        comm     = next_open * COMMISSION_PCT
        proceeds = port["position"]["shares"] * (next_open - comm)
        pnl_pct  = (proceeds / capital - 1) * 100
        is_win   = pnl_pct > WIN_THRESHOLD_PCT

        # ← ENTRY adatok kinyerése — nem a jelenlegi bull_score!
        entry_bull_score = port["position"]["bull_score"]
        entry_ctx        = port["position"]["context"]

        trade = {
            "entry_date":   port["position"]["entry_date"],
            "exit_date":    date,
            "entry_price":  port["position"]["entry_price"],
            "exit_price":   next_open,
            "pnl_usd":      proceeds - capital,
            "pnl_pct":      pnl_pct,
            "holding_days": (date - port["position"]["entry_date"]).days,
            "bull_score":   entry_bull_score,   # ← ENTRY score
            "context":      entry_ctx,           # ← ENTRY context
        }
        port["trades"].append(trade)
        port["cash"] = proceeds

        # Score-history frissítése az ENTRY score alapján
        if update_score_table and score_table:
            score_table.update(ticker, entry_bull_score, is_win)

        port["position"] = None


# ═══════════════════════════════════════════════════════
#  ADATLETÖLTÁS / FALLBACK
# ═══════════════════════════════════════════════════════

def download_or_mock(ticker: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(period=BACKTEST_PERIOD)
        if not data.empty:
            print(f"    ✅ Valós: {ticker} ({len(data)} nap)")
            return data
    except:
        pass

    # Szimulált fallback
    PARAMS = {
        "NVDA":  (165.0, 0.85, 0.58), "AAPL":  (138.0, 0.28, 0.28),
        "MSFT":  (258.0, 0.32, 0.26), "TSLA":  (300.0, 0.05, 0.72),
        "AMZN":  (114.0, 0.38, 0.34), "GOOGL": (112.0, 0.31, 0.29),
        "META":  (174.0, 0.62, 0.48), "NFLX":  (190.0, 0.41, 0.42),
        "PANW":  (155.0, 0.45, 0.40), "DELL":  (42.0,  0.52, 0.38),
    }
    if ticker not in PARAMS:
        return pd.DataFrame()

    np.random.seed(abs(hash(ticker)) % (2**31))
    sp, ar, av = PARAMS[ticker]
    dates  = pd.bdate_range(end=datetime(2025, 6, 1), periods=756)
    days   = len(dates)
    dr, dv = ar / 252, av / np.sqrt(252)
    px     = [sp]
    for _ in range(days - 1):
        px.append(max(px[-1] * (1 + np.random.normal(dr, dv)), 1.0))
    px   = np.array(px)
    rets = np.diff(px, prepend=px[0]) / px
    vol  = np.random.randint(5_000_000, 50_000_000, days).astype(float) * (1 + 3*np.abs(rets))
    df   = pd.DataFrame({
        "Open":   np.roll(px, 1), "High": px * (1 + np.abs(np.random.normal(0, 0.008, days))),
        "Low":    px * (1 - np.abs(np.random.normal(0, 0.008, days))),
        "Close":  px, "Volume": vol
    }, index=dates)
    df["Open"].iloc[0] = px[0]
    print(f"    📊 Szimulált: {ticker}  ${sp:.0f} → ${px[-1]:.0f}")
    return df


# ═══════════════════════════════════════════════════════
#  ÖSSZEFOGLALÓ RIPORT
# ═══════════════════════════════════════════════════════

def print_summary(results: list[dict], capital: float):
    total_q  = sum(r["quant"]["final_value"]       for r in results)
    total_ai = sum(r["ai_filtered"]["final_value"]  for r in results)
    total_bh = sum(r["bh_final"]                    for r in results)
    invested = capital * len(results)

    print("\n" + "═"*100)
    print("  BACKTEST v2.0 ÖSSZEFOGLALÓ — 3 ÉV  |  $1000/részvény  |  10 részvény")
    print("═"*100)
    print(f"\n  {'Ticker':6s}  "
          f"{'Quant végérték':>14s}  {'Quant%':>8s}  "
          f"{'AI-szűrt végérték':>17s}  {'AI%':>8s}  "
          f"{'B&H végérték':>13s}  {'B&H%':>7s}  "
          f"{'Quant T.':>8s}  {'AI T.':>6s}  {'Q win%':>6s}  {'AI win%':>7s}")
    print("  " + "─"*98)

    for r in sorted(results, key=lambda x: x["ai_filtered"]["total_pct"], reverse=True):
        q  = r["quant"]
        af = r["ai_filtered"]
        bh_pct = r["bh_pct"]
        best = max(q["total_pct"], af["total_pct"], bh_pct)
        icon = "🏆" if af["total_pct"] == best else ("🟢" if af["total_pct"] > 0 else "🔴")

        print(f"  {r['ticker']:6s}  "
              f"${q['final_value']:>13.2f}  {q['total_pct']:>+7.1f}%  "
              f"${af['final_value']:>16.2f}  {af['total_pct']:>+7.1f}%  "
              f"${r['bh_final']:>12.2f}  {bh_pct:>+6.1f}%  "
              f"{q['num_trades']:>8d}  {af['num_trades']:>6d}  "
              f"{q['win_rate']:>5.0f}%  {af['win_rate']:>6.0f}%  {icon}")

    print("  " + "─"*98)
    q_pct  = (total_q  / invested - 1) * 100
    ai_pct = (total_ai / invested - 1) * 100
    bh_pct = (total_bh / invested - 1) * 100
    print(f"  {'ÖSSZESEN':6s}  "
          f"${total_q:>13.2f}  {q_pct:>+7.1f}%  "
          f"${total_ai:>16.2f}  {ai_pct:>+7.1f}%  "
          f"${total_bh:>12.2f}  {bh_pct:>+6.1f}%")

    print(f"""
  ╔══════════════════════════════════════════════════╗
  ║  VÉGEREDMÉNY ($10 000 befektetve)                ║
  ║                                                  ║
  ║  🤖 Quant-only stratégia:  ${total_q:>8,.2f}  ({q_pct:>+6.1f}%)  ║
  ║  🧠 Quant + AI szűrő:      ${total_ai:>8,.2f}  ({ai_pct:>+6.1f}%)  ║
  ║  📊 Buy & Hold:             ${total_bh:>8,.2f}  ({bh_pct:>+6.1f}%)  ║
  ╚══════════════════════════════════════════════════╝""")

    # Legjobb módszer kijelölése
    methods = [("Quant-only", q_pct), ("AI-szűrt", ai_pct), ("Buy&Hold", bh_pct)]
    winner  = max(methods, key=lambda x: x[1])
    print(f"\n  🏆 Legjobb módszer: {winner[0]}  ({winner[1]:+.1f}%)")

    # Összes trade statisztika
    all_q_trades  = [t for r in results for t in r["quant"]["trades"]]
    all_ai_trades = [t for r in results for t in r["ai_filtered"]["trades"]]
    q_wins  = sum(1 for t in all_q_trades  if t["pnl_usd"] > 0)
    ai_wins = sum(1 for t in all_ai_trades if t["pnl_usd"] > 0)

    print(f"\n  Quant:   {len(all_q_trades)} trade, "
          f"win rate: {q_wins/len(all_q_trades)*100:.1f}%" if all_q_trades else "\n  Quant: 0 trade")
    print(f"  AI-szűrt: {len(all_ai_trades)} trade, "
          f"win rate: {ai_wins/len(all_ai_trades)*100:.1f}%" if all_ai_trades else "  AI-szűrt: 0 trade")

    print(f"""
  ⚠️  Szimulált adaton futott (ha a Yahoo Finance nem volt elérhető).
  ⚠️  Az AI szűrő küszöbe: {AI_BUY_FILTER_PCT}% bullish valószínűség.
  ⚠️  Múltbeli eredmény nem garancia a jövőre.
  ⚠️  Csak tájékoztató jellegű, nem befektetési tanács.
{"═"*100}
""")


# ═══════════════════════════════════════════════════════
#  FŐPROGRAM
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  BACKTEST v2.0 — Indul")
    print(f"  {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}")
    print(f"  Részvények: {', '.join(WATCHLIST)}")
    print(f"  AI szűrő küszöb: {AI_BUY_FILTER_PCT}%")
    print("═"*60)

    # AI modell betöltése
    ai_analyzer = AIAnalyzer()
    if not ai_analyzer.is_ready():
        print("\n  ⚠️  AI modell nem található!")
        print("  Futtasd először: python ai_layer.py --train")
        print("  A backtest AI szűrő nélkül fut tovább...\n")

    # Score-history tábla — a backtest során feltöltjük
    score_table = ScoreHistoryTable()
    if ai_analyzer.is_ready():
        score_table = ai_analyzer.score_table

    # Benchmark
    print("\n[Benchmark: SPY]")
    benchmark = pd.DataFrame()
    try:
        benchmark = yf.Ticker("SPY").history(period=BACKTEST_PERIOD)
        print(f"    ✅ SPY betöltve ({len(benchmark)} nap)")
    except:
        print("    📊 SPY szimulálva")
        np.random.seed(999)
        dates = pd.bdate_range(end=datetime(2025, 6, 1), periods=756)
        p = [430.0]
        for _ in range(len(dates)-1):
            p.append(max(p[-1] * (1 + np.random.normal(0.00038, 0.010)), 1.0))
        benchmark = pd.DataFrame({
            "Close": p, "Open": p, "High": p, "Low": p,
            "Volume": [50_000_000.0]*len(dates)
        }, index=dates)

    results = []
    for ticker in WATCHLIST:
        print(f"\n[{ticker}]")
        data = download_or_mock(ticker)
        if data is None or data.empty:
            print("    ❌ Adat nem elérhető, kihagyva.")
            continue

        result = backtest_ticker(
            ticker, data, benchmark, CAPITAL_PER_STOCK,
            ai_analyzer, score_table
        )
        if result is None:
            print("    ⚠️  Nem elég adat.")
            continue

        results.append(result)
        q  = result["quant"]
        af = result["ai_filtered"]
        print(f"    Quant:    ${q['final_value']:>8.2f}  ({q['total_pct']:>+.1f}%)  "
              f"| {q['num_trades']} trade  | win: {q['win_rate']:.0f}%")
        print(f"    AI-szűrt: ${af['final_value']:>8.2f}  ({af['total_pct']:>+.1f}%)  "
              f"| {af['num_trades']} trade  | win: {af['win_rate']:.0f}%")
        print(f"    B&H:      ${result['bh_final']:>8.2f}  ({result['bh_pct']:>+.1f}%)")

    if results:
        print_summary(results, CAPITAL_PER_STOCK)

        # ── Score-history mentése ─────────────────────────────
        # FONTOS: a score_table-t MINDIG mentjük, függetlenül attól
        # hogy az ai_analyzer be volt-e töltve. A backtest során
        # az _apply_signals() feltöltötte a memóriában lévő
        # score_table-t — ezt kell fájlba írni.
        import os
        os.makedirs("models", exist_ok=True)
        score_table.save("models/score_history.json")
        # Ellenőrzés: hány bejegyzés van
        total_entries = sum(
            len(v) for k, v in score_table.table.items() if k != "GLOBAL"
        )
        global_entries = len(score_table.table.get("GLOBAL", {}))
        print(f"  ✅ Score-history mentve: {total_entries} részvény-score bejegyzés, "
              f"{global_entries} globális score érték.")

        # ── Valós trade adatok mentése az ai_layer tanításához ─
        # Összegyűjtjük az összes quant trade context+label párját.
        # Ezeket a save_trade_data() fájlba írja (append mód),
        # hogy a következő --train futtatás felhasználhassa.
        #
        # MIÉRT CSAK A QUANT TRADE-EK?
        # Az AI-szűrt trade-ek körkörös logikát alkotnának:
        # az AI szűrte őket → az ő adatukból tanulna → önmaga
        # megerősítése (confirmation bias). A quant trade-ek
        # az AI előtt keletkeztek, ezért objektív tanítóadatok.
        all_trade_contexts = []
        all_trade_labels   = []

        for r in results:
            ticker = r["ticker"]
            for trade in r["quant"]["trades"]:
                ctx = trade.get("context", {})
                if not ctx:
                    continue
                ctx["ticker"] = ticker
                is_win = trade.get("pnl_pct", 0) > WIN_THRESHOLD_PCT
                all_trade_contexts.append(ctx)
                all_trade_labels.append(int(is_win))

        if all_trade_contexts:
            save_trade_data(all_trade_contexts, all_trade_labels)
            wins  = sum(all_trade_labels)
            total = len(all_trade_labels)
            print(f"  ✅ {total} trade elmentve tanításhoz "
                  f"(win rate: {wins/total*100:.1f}%)")
            print(f"  ℹ️  Újratanításhoz futtasd: python ai_layer.py --train")
        else:
            print("  ⚠️  Nem sikerült trade context-et menteni.")
            print("       Ellenőrizd hogy a build_context() visszaad-e adatot.")
    else:
        print("\n❌ Nincs eredmény.")
