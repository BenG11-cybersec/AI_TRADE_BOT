"""
╔══════════════════════════════════════════════════════════════════╗
║         BACKTEST READONLY — Csak olvas, semmit nem ír felül      ║
╠══════════════════════════════════════════════════════════════════╣
║  KÜLÖNBSÉG a backtest.py-tól:                                    ║
║    • NEM ír real_trade_data.json-t                               ║
║    • NEM frissíti a score_history.json-t                         ║
║    • NEM módosítja a betanított modellt                          ║
║    • Úgy viselkedik mint az aibotv3, csak múltbeli adatokon      ║
║                                                                  ║
║  TELJESÍTMÉNY JAVÍTÁSOK:                                         ║
║    1. AI predict() csak jelzés napján fut (nem minden napra)     ║
║    2. NumPy tömbök a Pandas iloc helyett a belső ciklusban       ║
║    3. OBV slope előre kiszámítva rolling window-val              ║
║    4. bench_sub DataFrame újralétrehozása megszűnt               ║
║                                                                  ║
║  HOW TO RUN:                                                     ║
║    python backtest_readonly.py                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

from ai_layer import AIAnalyzer, ScoreHistoryTable
from aibotv3  import BUY_THRESHOLD, SELL_THRESHOLD, _aligned_relative_strength, _obv_slope


# ═══════════════════════════════════════════════════════
#  KONFIGURÁCIÓ
# ═══════════════════════════════════════════════════════

"""WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "PANW", "DELL",
    "JPM", "V", "MA", "BAC", "GS",
    "JNJ", "UNH", "LLY", "PFE", "MRK",
    "LMT", "RTX", "NOC", "BA", "CAT", "RHM.DE",
    "NOW", "CVS", "VCEL", "AXON", "MC", "AIR", "COHR",
    "BLK", "BYD", "LCID", "DOCS", "S", "VST", "KRNT",
]"""
WATCHLIST = [ "ARM", "AMD", "SHOP", "BSX", "MU", "MCD"]

CAPITAL_PER_STOCK  = 1000.0
BACKTEST_PERIOD    = "4y"
COMMISSION_PCT     = 0.001
BUY_THRESHOLD_OVERRIDE = 5   # felülírja az importált értéket
AI_BUY_FILTER_PCT  = 60.0
AI_SELL_FILTER_PCT = 55.0
WIN_THRESHOLD_PCT  = 5.0


# ═══════════════════════════════════════════════════════
#  INDIKÁTOR SZÁMÍTÁS
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
    # BB oszlopnév normalizálás
    bb_map = {"BBL":"BBL_20_2.0","BBM":"BBM_20_2.0",
              "BBU":"BBU_20_2.0","BBB":"BBB_20_2.0","BBP":"BBP_20_2.0"}
    rename = {}
    for col in data.columns:
        for prefix, std_name in bb_map.items():
            if col.startswith(prefix + "_20_2.0") and col != std_name:
                rename[col] = std_name
    if rename:
        data.rename(columns=rename, inplace=True)
    return data


# ═══════════════════════════════════════════════════════
#  ELŐRE KISZÁMÍTOTT ROLLING ÉRTÉKEK
#  A fő bottleneck az volt hogy minden napra újra sliceltük
#  a DataFrame-et és újra számoltuk az OBV slope-ot, BB
#  percentilisét stb. Most ezeket EGYSZER számoljuk ki
#  az egész adatsorra, NumPy tömbökként.
# ═══════════════════════════════════════════════════════

def precompute_rolling(data: pd.DataFrame) -> dict:
    """
    Előre kiszámítja azokat az értékeket amelyek a score_day()-ben
    rolling window-t igényelnek. Eredmény: NumPy tömbök,
    indexelés: arr[i] = az i-edik napra vonatkozó érték.

    Ez a legfontosabb teljesítmény javítás:
    - OBV slope: 1750× külön polyfit helyett rolling korreláció
    - BB percentilis: 1750× quantile helyett rolling min/max közelítés
    - RSI divergencia: 1750× min/max helyett rolling min/max
    - VAM: 1750× std számítás helyett rolling std
    """
    n = len(data)

    # ── OBV slope előre számítás ──────────────────────────
    # Az _obv_slope 20 napos lineáris regressziót futtat.
    # Közelítés: rolling 20 napos korreláció az idővel — gyors és elegendő.
    obv = data["OBV"].values.astype(float)
    obv_slope_arr = np.zeros(n)
    win = 20
    for i in range(win, n):
        y = obv[i-win:i]
        y_mean = np.mean(np.abs(y)) or 1.0
        x = np.arange(win, dtype=float)
        # polyfit helyett: slope = cov(x,y) / var(x)
        xm = x - x.mean()
        ym = (y / y_mean) - (y / y_mean).mean()
        denom = (xm * xm).sum()
        obv_slope_arr[i] = float((xm * ym).sum() / denom) if denom > 0 else 0.0

    # ── BB rolling percentilis (21 napos ablak, 20. percentilis) ─
    bb_width = data["BBB_20_2.0"].values.astype(float) if "BBB_20_2.0" in data.columns \
               else np.zeros(n)
    bb_pct20_arr = np.zeros(n)
    for i in range(30, n):
        window_bb = bb_width[max(0, i-252):i]
        valid = window_bb[~np.isnan(window_bb)]
        bb_pct20_arr[i] = float(np.percentile(valid, 20)) if len(valid) >= 10 else 0.0

    # ── RSI divergencia: rolling 21 napos min/max ────────
    rsi_vals  = data["RSI_14"].values.astype(float)
    close_vals= data["Close"].values.astype(float)
    DIV_WIN   = 21

    rsi_min_arr   = np.zeros(n)
    rsi_max_arr   = np.zeros(n)
    close_min_arr = np.zeros(n)
    close_max_arr = np.zeros(n)
    close_std_arr = np.zeros(n)

    for i in range(DIV_WIN, n):
        w_rsi   = rsi_vals[i-DIV_WIN:i]
        w_close = close_vals[i-DIV_WIN:i]
        rsi_min_arr[i]   = np.nanmin(w_rsi)
        rsi_max_arr[i]   = np.nanmax(w_rsi)
        close_min_arr[i] = np.nanmin(w_close)
        close_max_arr[i] = np.nanmax(w_close)
        close_std_arr[i] = np.nanstd(w_close)

    # ── VAM: rolling 63 napos std és return ──────────────
    vam_arr = np.zeros(n)
    for i in range(63, n):
        ret63 = np.diff(close_vals[i-63:i+1]) / close_vals[i-63:i]
        av    = float(np.std(ret63) * np.sqrt(252))
        if av > 0.01:
            pr = (close_vals[i] / close_vals[i-63]) - 1
            vam_arr[i] = pr / av

    # ── 52 hetes high/low (rolling 252 napos) ─────────────
    high52_arr = np.zeros(n)
    low52_arr  = np.zeros(n)
    for i in range(252, n):
        w = close_vals[i-252:i]
        high52_arr[i] = np.max(w)
        low52_arr[i]  = np.min(w)

    return {
        "obv_slope":  obv_slope_arr,
        "bb_pct20":   bb_pct20_arr,
        "rsi_min":    rsi_min_arr,
        "rsi_max":    rsi_max_arr,
        "close_min":  close_min_arr,
        "close_max":  close_max_arr,
        "close_std":  close_std_arr,
        "vam":        vam_arr,
        "high52":     high52_arr,
        "low52":      low52_arr,
    }


# ═══════════════════════════════════════════════════════
#  GYORS SCORING — CSAK NUMPY TÖMBÖKKEL
#  Nincs pandas .iloc slice, nincs DataFrame újralétrehozás
# ═══════════════════════════════════════════════════════

def score_day_fast(i: int, data: pd.DataFrame, pre: dict,
                   bench_rs_arr: np.ndarray) -> tuple[int, int, dict, float]:
    """
    Egy napra kiszámolja a bull/bear score-t.
    Az összes rolling érték már előre ki van számítva a `pre` dict-ben.
    Így minden iteráció O(1) — nincs slice, nincs rolling számítás.
    """
    def gv(arr, idx, default=0.0):
        try:
            v = float(arr[idx])
            return default if np.isnan(v) else v
        except:
            return default

    # Közvetlen NumPy tömb elérés — ez a leggyorsabb
    cols = data.columns.tolist()
    row  = data.iloc[i]
    prev = data.iloc[i-1]

    def g(key, default=0.0):
        try:
            v = float(row[key])
            return default if np.isnan(v) else v
        except:
            return default
    def gp(key, default=0.0):
        try:
            v = float(prev[key])
            return default if np.isnan(v) else v
        except:
            return default

    close       = g("Close")
    rsi         = g("RSI_14",       50.0)
    sma50       = g("SMA_50",       close)
    sma200      = g("SMA_200",      close)
    prev_sma50  = gp("SMA_50",      sma50)
    prev_sma200 = gp("SMA_200",     sma200)
    bb_upper    = g("BBU_20_2.0",   close)
    bb_lower    = g("BBL_20_2.0",   close)
    bb_mid      = g("BBM_20_2.0",   close)
    bb_width    = g("BBB_20_2.0",   0)
    prev_bb_mid = gp("BBM_20_2.0",  bb_mid)
    adx         = g("ADX_14",       20.0)
    dmp         = g("DMP_14",       0.0)
    dmn         = g("DMN_14",       0.0)

    bull = 0; bear = 0; s = {}

    # S1: Trend
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

    # S2: MACD Histogram (utolsó 5 érték közvetlen tömb elérés)
    s2_pts = 0
    macd_col = "MACDh_12_26_9"
    if macd_col in cols:
        macd_vals = data[macd_col].values
        window5 = macd_vals[max(0,i-4):i+1]
        valid5  = window5[~np.isnan(window5)]
        if len(valid5) >= 4:
            last4 = valid5[-4:]
            if last4[0]<0 and last4[1]<0 and last4[2]>0 and last4[3]>last4[2]>0:
                bull += 3; s2_pts = 3
            elif last4[0]>0 and last4[1]>0 and last4[2]<0 and last4[3]<last4[2]<0:
                bear += 3; s2_pts = -3
            elif all(valid5[-3:]>0) and valid5[-1]>valid5[-2]>valid5[-3]:
                bull += 2; s2_pts = 2
            elif all(valid5[-3:]<0) and valid5[-1]<valid5[-2]<valid5[-3]:
                bear += 2; s2_pts = -2
            elif valid5[-1] > 0:
                bull += 1; s2_pts = 1
            elif valid5[-1] < 0:
                bear += 1; s2_pts = -1
    s["s2_macd"] = s2_pts

    # S3: RSI zóna
    s3_pts = 0
    if rsi > 55:   bull += 2; s3_pts = 2
    elif rsi > 50: bull += 1; s3_pts = 1
    elif rsi < 45: bear += 2; s3_pts = -2
    elif rsi < 50: bear += 1; s3_pts = -1
    s["s3_rsi"] = s3_pts

    # S3: RSI divergencia (előre számított értékek)
    div_bull = div_bear = 0
    c_std = gv(pre["close_std"], i)
    if c_std > 0:
        if close <= gv(pre["close_min"],i) + c_std*0.05 and rsi > gv(pre["rsi_min"],i) + 5:
            bull += 2; div_bull = 1
        if close >= gv(pre["close_max"],i) - c_std*0.05 and rsi < gv(pre["rsi_max"],i) - 5:
            bear += 2; div_bear = 1
    s["s3_div_bull"] = div_bull
    s["s3_div_bear"] = div_bear

    # S4: Bollinger Bands (előre számított BB percentilis)
    s4_pts = 0
    bb_pct20   = gv(pre["bb_pct20"], i)
    bb_squeeze = bb_width < bb_pct20 and bb_width > 0 and bb_pct20 > 0
    if close > bb_upper:
        pts = 2 if adx > 25 else 1; bull += pts; s4_pts = pts
    elif close < bb_lower:
        prev_close = gp("Close")
        if close > prev_close:
            bull += 2; s4_pts = 2
        else:
            bear += 1; s4_pts = -1
    elif close > bb_mid and gp("Close") < prev_bb_mid:
        bull += 1; s4_pts = 1
    s["s4_bb"] = s4_pts

    # S5: ADX + DI
    s5_pts = 0
    if 25 < adx <= 40:
        if dmp > dmn:
            pts = 2 if dmp-dmn > 10 else 1; bull += pts; s5_pts = pts
        elif dmn > dmp:
            pts = 2 if dmn-dmp > 10 else 1; bear += pts; s5_pts = -pts
    s["s5_adx"] = s5_pts

    # S6: OBV slope (előre számított)
    s6_pts = 0
    obv_slope_val = gv(pre["obv_slope"], i)
    if obv_slope_val > 0.005:
        pts = 2 if bull > bear else 1; bull += pts; s6_pts = pts
    elif obv_slope_val < -0.005:
        pts = 2 if bear > bull else 1; bear += pts; s6_pts = -pts
    s["s6_obv"] = s6_pts

    # S7: Relatív erő (előre számított)
    s7_pts = 0
    rs = gv(bench_rs_arr, i)
    if rs > 10:   bull += 2; s7_pts = 2
    elif rs > 3:  bull += 1; s7_pts = 1
    elif rs < -10: bear += 2; s7_pts = -2
    elif rs < -3:  bear += 1; s7_pts = -1
    s["s7_rs"] = s7_pts

    # S8: 52 hetes pozíció (előre számított)
    s8_pts = 0
    h52 = gv(pre["high52"], i)
    l52 = gv(pre["low52"],  i)
    if h52 > 0:
        if (close - h52) / h52 * 100 > -10:
            bull += 1; s8_pts = 1
        elif l52 > 0 and (close - l52) / l52 * 100 < 20:
            bear += 1; s8_pts = -1
    s["s8_52w"] = s8_pts

    # S9: VAM (előre számított)
    s9_pts  = 0
    vam_val = gv(pre["vam"], i)
    if   vam_val >  0.6:  bull += 2; s9_pts =  2
    elif vam_val >  0.25: bull += 1; s9_pts =  1
    elif vam_val < -0.6:  bear += 2; s9_pts = -2
    elif vam_val < -0.25: bear += 1; s9_pts = -1
    s["s9_vam"] = s9_pts

    return bull, bear, s, vam_val


# ═══════════════════════════════════════════════════════
#  RELATÍV ERŐ ELŐRE SZÁMÍTÁS
# ═══════════════════════════════════════════════════════

def precompute_relative_strength(data: pd.DataFrame,
                                  benchmark: pd.DataFrame,
                                  lookback: int = 63) -> np.ndarray:
    """
    Előre kiszámolja a részvény vs SPY relatív hozamot
    minden napra (rolling lookback ablak).
    Visszatér: NumPy tömb, arr[i] = relatív hozam % az i. napon.
    """
    n = len(data)
    rs_arr = np.zeros(n)

    if benchmark.empty:
        return rs_arr

    close_s = data["Close"].values.astype(float)
    spy_col = "SPY_Close" if "SPY_Close" in data.columns else None
    if spy_col is None:
        return rs_arr

    spy_s = data[spy_col].values.astype(float)

    for i in range(lookback, n):
        s_start = close_s[i - lookback]
        s_end   = close_s[i]
        b_start = spy_s[i - lookback]
        b_end   = spy_s[i]
        if s_start > 0 and b_start > 0:
            sr = (s_end / s_start - 1) * 100
            br = (b_end / b_start - 1) * 100
            rs_arr[i] = sr - br

    return rs_arr


# ═══════════════════════════════════════════════════════
#  BACKTEST MOTOR — READONLY, GYORS
# ═══════════════════════════════════════════════════════

def backtest_ticker_readonly(ticker: str, data: pd.DataFrame,
                              benchmark: pd.DataFrame,
                              capital: float,
                              ai_analyzer: AIAnalyzer) -> dict | None:
    """
    Csak olvasó backtest — semmit nem ír felül.
    Gyors: minden rolling értéket egyszer számít elő,
    az AI predict csak jelzés napján fut.
    """
    if data.empty or len(data) < 220:
        return None

    # Indikátorok kiszámítása
    data = calculate_indicators(data)
    data.index = data.index.tz_localize(None)

    # Benchmark merge (egyszer, a ciklus előtt)
    if not benchmark.empty:
        bm = benchmark.copy()
        bm.index = bm.index.tz_localize(None) if bm.index.tz else bm.index
        data = pd.merge_asof(
            data,
            bm[["Close"]].rename(columns={"Close": "SPY_Close"}),
            left_index=True, right_index=True, direction="backward"
        )
    else:
        data["SPY_Close"] = np.nan

    data = data.dropna(subset=["SMA_200"])
    n    = len(data)
    if n < 10:
        return None

    # ── Előre számítás (ez a gyorsítás lelke) ────────────
    pre      = precompute_rolling(data)
    rs_arr   = precompute_relative_strength(data, benchmark)

    # NumPy tömbök a belső ciklushoz
    close_vals = data["Close"].values.astype(float)
    open_vals  = data["Open"].values.astype(float)
    dates      = data.index

    # Két párhuzamos portfólió
    quant = {"cash": capital, "position": None, "trades": [], "equity_curve": []}
    ai_f  = {"cash": capital, "position": None, "trades": [], "equity_curve": []}

    for i in range(1, n - 1):
        close     = close_vals[i]
        next_open = open_vals[i + 1]
        date      = dates[i]

        # Gyors scoring — nincs pandas slice
        bull, bear, s_scores, vam_val = score_day_fast(i, data, pre, rs_arr)
        net = bull - bear

        buy_signal  = bull >= BUY_THRESHOLD_OVERRIDE
        sell_signal = bear >= SELL_THRESHOLD

        # ── AI predict CSAK jelzés napján ────────────────
        # Ez az eredeti backtest legfőbb bottleneckje volt:
        # minden napra futott (~1750×). Most csak akkor hívódik
        # meg ha ténylegesen dönteni kell (~10-30× részvényenként).
        ai_bull_pct = 50.0
        if (buy_signal or sell_signal) and ai_analyzer and ai_analyzer.is_ready():
            ctx = {
                "ticker":      ticker,
                "bull_score":  bull,
                "bear_score":  bear,
                "net_score":   net,
                "rsi":         float(data["RSI_14"].values[i]),
                "adx":         float(data["ADX_14"].values[i]),
                "bb_width":    float(data["BBB_20_2.0"].values[i])
                               if "BBB_20_2.0" in data.columns else 0.0,
                "obv_slope":   float(pre["obv_slope"][i]),
                "vam":         float(pre["vam"][i]),
                "above_sma200":int(close > float(data["SMA_200"].values[i])),
                "sma50_slope": 0.0,
                "volatility_30d": 0.25,
                "strategy_scores": s_scores,
                "reasons": [],
            }
            try:
                report      = ai_analyzer.predict(ctx)
                ai_bull_pct = report.get("bull_pct", 50.0)
            except Exception:
                pass

        # Quant portfólió
        _apply_ro(quant, buy_signal, sell_signal,
                  date, next_open, capital, bull)

        # AI-szűrt portfólió
        ai_buy  = buy_signal  and ai_bull_pct >= AI_BUY_FILTER_PCT
        ai_sell = sell_signal and ai_bull_pct <  AI_SELL_FILTER_PCT
        _apply_ro(ai_f, ai_buy, ai_sell,
                  date, next_open, capital, bull)

        # Napi equity
        for port in (quant, ai_f):
            eq = port["position"]["shares"] * close if port["position"] else port["cash"]
            port["equity_curve"].append({"date": str(date.date()), "equity": round(eq, 2)})

    # Nyitott pozíciók zárása
    final_price = float(close_vals[-1])
    for port in (quant, ai_f):
        if port["position"]:
            comm     = final_price * COMMISSION_PCT
            proceeds = port["position"]["shares"] * (final_price - comm)
            port["trades"].append({
                "entry_date":  str(port["position"]["entry_date"].date()),
                "exit_date":   str(dates[-1].date()),
                "entry_price": port["position"]["entry_price"],
                "exit_price":  final_price,
                "pnl_usd":     proceeds - capital,
                "pnl_pct":     (proceeds / capital - 1) * 100,
                "holding_days":(dates[-1] - port["position"]["entry_date"]).days,
                "still_open":  True,
            })
            port["cash"] = proceeds

    # Buy & Hold
    bh_final = (capital / close_vals[0]) * final_price * (1 - COMMISSION_PCT)
    bh_pct   = (bh_final / capital - 1) * 100

    def _stats(port):
        trades = port["trades"]
        wins   = [t for t in trades if t["pnl_usd"] > 0]
        return {
            "final_value": round(port["cash"], 2),
            "total_pct":   round((port["cash"] / capital - 1) * 100, 2),
            "num_trades":  len(trades),
            "win_rate":    round(len(wins)/len(trades)*100, 1) if trades else 0.0,
            "trades":      trades,
        }

    return {
        "ticker":      ticker,
        "quant":       _stats(quant),
        "ai_filtered": _stats(ai_f),
        "bh_final":    round(bh_final, 2),
        "bh_pct":      round(bh_pct, 2),
    }


def _apply_ro(port, buy_signal, sell_signal, date, next_open, capital, bull_score):
    """Portfólió logika — readonly verzió, nem frissít score_table-t."""
    if port["position"] is None and buy_signal:
        comm   = next_open * COMMISSION_PCT
        shares = port["cash"] / (next_open + comm)
        port["cash"] = 0.0
        port["position"] = {
            "entry_date":  date,
            "entry_price": next_open,
            "shares":      shares,
            "bull_score":  bull_score,
        }
    elif port["position"] is not None and sell_signal:
        comm     = next_open * COMMISSION_PCT
        proceeds = port["position"]["shares"] * (next_open - comm)
        port["trades"].append({
            "entry_date":   str(port["position"]["entry_date"].date()),
            "exit_date":    str(date.date()),
            "entry_price":  port["position"]["entry_price"],
            "exit_price":   next_open,
            "pnl_usd":      round(proceeds - capital, 2),
            "pnl_pct":      round((proceeds / capital - 1) * 100, 2),
            "holding_days": (date - port["position"]["entry_date"]).days,
            "bull_score":   port["position"]["bull_score"],
        })
        port["cash"]     = proceeds
        port["position"] = None


# ═══════════════════════════════════════════════════════
#  ADATLETÖLTÁS
# ═══════════════════════════════════════════════════════

def download_or_mock(ticker: str) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(period=BACKTEST_PERIOD)
        if not data.empty:
            print(f"    ✅ {ticker}: {len(data)} nap")
            return data
    except Exception:
        pass
    print(f"    ❌ {ticker}: letöltés sikertelen")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  ÖSSZEFOGLALÓ
# ═══════════════════════════════════════════════════════

def print_summary(results: list[dict], capital: float):
    total_q  = sum(r["quant"]["final_value"]       for r in results)
    total_ai = sum(r["ai_filtered"]["final_value"]  for r in results)
    total_bh = sum(r["bh_final"]                    for r in results)
    invested = capital * len(results)

    print("\n" + "═"*96)
    print("  READONLY BACKTEST — ÖSSZEFOGLALÓ")
    print("═"*96)
    print(f"  {'Ticker':7s}  {'Quant':>10s}  {'Q%':>7s}  "
          f"{'AI-szűrt':>10s}  {'AI%':>7s}  "
          f"{'B&H':>10s}  {'BH%':>7s}  "
          f"{'Q T.':>5s}  {'AI T.':>5s}  {'Q win%':>6s}")
    print("  " + "─"*94)

    for r in sorted(results, key=lambda x: x["ai_filtered"]["total_pct"], reverse=True):
        q  = r["quant"]
        af = r["ai_filtered"]
        best = max(q["total_pct"], af["total_pct"], r["bh_pct"])
        icon = "🏆" if af["total_pct"] == best else ("🟢" if af["total_pct"] > 0 else "🔴")
        print(f"  {r['ticker']:7s}  "
              f"${q['final_value']:>9.2f}  {q['total_pct']:>+6.1f}%  "
              f"${af['final_value']:>9.2f}  {af['total_pct']:>+6.1f}%  "
              f"${r['bh_final']:>9.2f}  {r['bh_pct']:>+6.1f}%  "
              f"{q['num_trades']:>5d}  {af['num_trades']:>5d}  "
              f"{q['win_rate']:>5.0f}%  {icon}")

    print("  " + "─"*94)
    q_pct  = (total_q  / invested - 1) * 100
    ai_pct = (total_ai / invested - 1) * 100
    bh_pct = (total_bh / invested - 1) * 100

    print(f"\n  {'ÖSSZESEN':7s}  "
          f"${total_q:>9.2f}  {q_pct:>+6.1f}%  "
          f"${total_ai:>9.2f}  {ai_pct:>+6.1f}%  "
          f"${total_bh:>9.2f}  {bh_pct:>+6.1f}%")

    winner = max([("Quant", q_pct), ("AI-szűrt", ai_pct), ("B&H", bh_pct)],
                 key=lambda x: x[1])
    print(f"\n  🏆 Legjobb: {winner[0]}  ({winner[1]:+.1f}%)")
    print(f"  💰 Befektetve: ${invested:,.0f}  →  "
          f"Quant: ${total_q:,.2f}  |  AI: ${total_ai:,.2f}  |  B&H: ${total_bh:,.2f}")
    print(f"\n  ⚠️  Ez a backtest NEM ír semmit — modellek, score_history érintetlen.")
    print("═"*96 + "\n")


# ═══════════════════════════════════════════════════════
#  FŐPROGRAM
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    print("\n" + "═"*60)
    print("  READONLY BACKTEST — Indul")
    print(f"  {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}")
    print(f"  Részvények: {len(WATCHLIST)} db  |  Periódus: {BACKTEST_PERIOD}")
    print(f"  ✅ Semmit nem ír felül (modell, score_history, trade_data)")
    print("═"*60)

    ai_analyzer = AIAnalyzer()
    if not ai_analyzer.is_ready():
        print("  ⚠️  AI modell nincs betanítva — csak quant mód fut.")

    # Benchmark
    print("\n[Benchmark: SPY]")
    benchmark = pd.DataFrame()
    try:
        benchmark = yf.Ticker("SPY").history(period=BACKTEST_PERIOD)
        benchmark.index = benchmark.index.tz_localize(None)
        print(f"    ✅ SPY: {len(benchmark)} nap")
    except Exception:
        print("    ⚠️  SPY nem elérhető")

    results   = []
    t_total   = time.time()

    for ticker in WATCHLIST:
        print(f"\n[{ticker}]")
        t0   = time.time()
        data = download_or_mock(ticker)
        if data is None or data.empty:
            continue

        result = backtest_ticker_readonly(
            ticker, data, benchmark, CAPITAL_PER_STOCK, ai_analyzer
        )
        elapsed = time.time() - t0

        if result is None:
            print(f"    ⚠️  Nem elég adat.")
            continue

        results.append(result)
        q  = result["quant"]
        af = result["ai_filtered"]
        print(f"    Quant:    ${q['final_value']:>8.2f}  ({q['total_pct']:>+.1f}%)  "
              f"| {q['num_trades']} trade | win: {q['win_rate']:.0f}%  "
              f"| ⏱ {elapsed:.1f}s")
        print(f"    AI-szűrt: ${af['final_value']:>8.2f}  ({af['total_pct']:>+.1f}%)  "
              f"| {af['num_trades']} trade")
        print(f"    B&H:      ${result['bh_final']:>8.2f}  ({result['bh_pct']:>+.1f}%)")

    print(f"\n  ⏱  Teljes futásidő: {time.time()-t_total:.1f}s")

    if results:
        print_summary(results, CAPITAL_PER_STOCK)
