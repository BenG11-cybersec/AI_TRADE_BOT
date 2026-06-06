"""
╔══════════════════════════════════════════════════════════════════╗
║         PROFESSIONAL LONG-TERM TRADING BOT v3.0                  ║
║         Quant-Grade Multi-Factor Scoring Engine                  ║
╠══════════════════════════════════════════════════════════════════╣
║  v3.0 VÁLTOZÁSOK — AUDIT ALAPJÁN:                                ║
║                                                                  ║
║  FIX 1 — RSI Divergencia ablak: 10 → 21 nap (1 hónap)            ║
║    Miért: 10 nap túl rövid → zaj. 21 nap egy valódi              ║
║    swing-ciklus, kevesebb hamis jelzés.                          ║
║                                                                  ║
║  FIX 2 — Relatív erő: integer index → dátum alapú merge          ║
║    Miért: SPY és a részvény sora NEM egyezik (hiányzó            ║
║    napok, IPO, holiday). Most dátumra joinalunk → pontos.        ║
║                                                                  ║
║  FIX 3 — MACD megerősítés: 1 nap → 3 napos konszolidáció         ║
║    Miért: 1 napos flip hamis. Ha 3 egymást követő napon          ║
║    pozitív és növekvő a histogram → valódi momentum.             ║
║                                                                  ║
║  FIX 4 — OBV: 5 vs 20 napos átlag → lineáris regresszió          ║
║    Miért: a rövid átlag manipulálható 1 spike-kal.               ║
║    Regresszió a valódi trend irányt mutatja.                     ║
║                                                                  ║
║  FIX 5 — BB Squeeze: globális quantile → rolling 252 napos       ║
║    Miért: a globális szűkülés nem mond semmit, ha a részvény     ║
║    az elmúlt évben mindig szűk volt. Relatív szűkülés kell.      ║
║                                                                  ║
║  FIX 6 — ADX: DI+ / DI- irány megerősítés hozzáadva              ║
║    Miért: ADX > 25 önmagában nem jelzi az irányt.                ║
║    DI+ > DI- = bullish trend, DI- > DI+ = bearish trend.         ║
║                                                                  ║
║  FIX 7 — 52 hetes: kereskedési napok → naptári napok             ║
║    Miért: tail(252) elcsúszik, ha hiányzó napok vannak.          ║
║    Most dátum alapú szeletelés.                                  ║
║                                                                  ║
║  ÚJ S9 — Volatility-Adjusted Momentum (VAM)                      ║
║    A hozamot a részvény saját volatilitásával osztjuk →          ║
║    összehasonlítható momentum-score különböző részvényeknél.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


# ── AI Layer import ───────────────────────────────────
try:
    from ai_layer import AIAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  ai_layer.py nem található — AI réteg kikapcsolva.")

# ── Portfolio Optimizer import ────────────────────────
# XGBoostSignalBooster: második ML modell a RF mellé (ensemble)
# run_portfolio_from_bot_signals: Markowitz + Risk Parity optimalizálás
try:
    from portfolio_optimizer import (
        XGBoostSignalBooster,
        run_portfolio_from_bot_signals,
    )
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False
    print("⚠️  portfolio_optimizer.py nem található — portfolió réteg kikapcsolva.")


# ═══════════════════════════════════════════════════════
#  KONFIGURÁCIÓ
# ═══════════════════════════════════════════════════════
load_dotnev()

DISCORD_WEBHOOK_BULL = os.getenv(bull_url, "BACKUP")
DISCORD_WEBHOOK_BEAR = os.getenv(ear_url, "BACKUP")

WATCHLIST = ["NVDA", "DELL", "PANW", "RHM.DE", "NFLX", "AAPL", "MSFT"]

BUY_THRESHOLD  = 9
SELL_THRESHOLD = 9

# ── v3.0 hangolható paraméterek ───────────────────────
RSI_DIVERGENCE_WINDOW   = 21   # FIX 1: volt 10, most 21 nap
MACD_CONFIRM_DAYS       = 3    # FIX 3: ennyi egymást követő pozitív nap kell
OBV_REGRESSION_WINDOW   = 20   # FIX 4: OBV lineáris regresszió ablaka
BB_SQUEEZE_ROLLING      = 252  # FIX 5: rolling quantile ablaka (1 év)
RS_LOOKBACK_DAYS        = 63   # S7: relatív erő visszatekintési ablak (3 hó)


# ═══════════════════════════════════════════════════════
#  DISCORD ÉRTESÍTŐ
# ═══════════════════════════════════════════════════════

class DiscordNotifier:
    def __init__(self, webhook_url_bull: str):
        self.webhook_url = webhook_url

    def send_alert(self, message: str):
        payload = {"content": message}
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 204:
                print("  ✅ Discord értesítés elküldve.")
            else:
                print(f"  ❌ Hiba: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Hálózati hiba: {e}")


# ═══════════════════════════════════════════════════════
#  ADAT LETÖLTŐ
# ═══════════════════════════════════════════════════════

class MarketDataFetcher:
    def get_data(self, ticker: str) -> tuple[pd.DataFrame, dict]:
        print(f"  Adatok letöltése: {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            data  = stock.history(period="2y")
            info  = stock.info
            return data, info
        except Exception as e:
            print(f"  ❌ Letöltési hiba ({ticker}): {e}")
            return pd.DataFrame(), {}

    def get_benchmark(self) -> pd.DataFrame:
        try:
            spy = yf.Ticker("SPY")
            return spy.history(period="2y")
        except:
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  SEGÉD: OBV REGRESSZIÓ (FIX 4)
# ═══════════════════════════════════════════════════════

def _obv_slope(obv_series: pd.Series, window: int = OBV_REGRESSION_WINDOW) -> float:
    """
    Lineáris regresszió meredeksége az OBV utolsó N napjára.

    MIÉRT JOBB MINT AZ ÁTLAG-ÖSSZEHASONLÍTÁS?
    ──────────────────────────────────────────
    Ha pl. a 20 napos OBV sorozat: [100, 90, 80, 70, 60, ... 200, 200, 200]
    akkor az 5 napos átlag > 20 napos átlag is lehet, holott az OBV
    az utolsó napokban stagnál. A regresszió meredeksége pontosan
    megmutatja, hogy összességében emelkedik vagy csökken a trend.

    Visszatér: pozitív szám = emelkedő OBV, negatív = csökkenő OBV
    """
    series = obv_series.tail(window).dropna()
    if len(series) < 5:
        return 0.0
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    # Normalizálás: a slope részvényfüggő, ezért y-t az átlagával osztjuk
    y_mean = np.mean(np.abs(y)) or 1.0
    slope, _ = np.polyfit(x, y / y_mean, 1)
    return float(slope)


# ═══════════════════════════════════════════════════════
#  SEGÉD: DÁTUM-ALAPÚ RELATÍV ERŐ (FIX 2)
# ═══════════════════════════════════════════════════════

def _aligned_relative_strength(
    stock_data: pd.DataFrame,
    benchmark:  pd.DataFrame,
    lookback:   int = RS_LOOKBACK_DAYS
) -> tuple[int, str]:
    """
    Relatív erő számítás dátum-alapú igazítással.

    A PROBLÉMA (amit ez javít):
    ────────────────────────────
    Ha stock_data.iloc[-63] és benchmark.iloc[-63] integer indexet
    használ, és a kettő eltérő számú kereskedési napot tartalmaz
    (pl. az egyik tőzsdén volt szünet, amit a másikon nem), akkor
    a 63. sor nem ugyanazt a naptári napot jelenti.

    A MEGOLDÁS:
    ────────────
    1. Megkeressük a mai dátumot (utolsó sor indexe a stock_data-ban)
    2. Naptárilag N napot visszamegyünk (lookback * 1.5 nap, hogy
       biztosan legyen elég kereskedési nap)
    3. A benchmark-ban a legközelebbi elérhető napot keressük meg
       pd.merge_asof segítségével → nincs indexelési eltolódás
    """
    if benchmark.empty or len(stock_data) < 10:
        return 0, "📊 Relatív erő: S&P500 adatok nem elérhetők"

    try:
        # Timezone-mentes dátumindex mindkét oldalon
        s_idx = stock_data.index.tz_localize(None) if stock_data.index.tz else stock_data.index
        b_idx = benchmark.index.tz_localize(None)  if benchmark.index.tz  else benchmark.index

        today     = s_idx[-1]
        # Naptári visszatekintés: lookback kereskedési nap ≈ lookback * 1.45 naptári nap
        cal_days  = int(lookback * 1.5)
        target_dt = today - timedelta(days=cal_days)

        # Részvény: legközelebbi elérhető nap a target_dt-hez
        s_pos = s_idx.searchsorted(target_dt, side="left")
        s_pos = max(0, min(s_pos, len(stock_data) - 1))
        stock_start = float(stock_data["Close"].iloc[s_pos])
        stock_end   = float(stock_data["Close"].iloc[-1])

        # Benchmark: szintén legközelebbi nap a target_dt-hez
        b_pos = b_idx.searchsorted(target_dt, side="left")
        b_pos = max(0, min(b_pos, len(benchmark) - 1))
        bench_start = float(benchmark["Close"].iloc[b_pos])
        bench_end   = float(benchmark["Close"].iloc[-1])

        if stock_start <= 0 or bench_start <= 0:
            return 0, "📊 Relatív erő: érvénytelen ár"

        stock_ret = (stock_end / stock_start - 1) * 100
        bench_ret = (bench_end / bench_start - 1) * 100
        diff      = stock_ret - bench_ret

        if diff > 10:
            return 2, f"🌟 Kiemelkedő relatív erő: +{diff:.1f}% a piaci átlag felett (3 hónap) (+2)"
        elif diff > 3:
            return 1, f"✅ Pozitív relatív erő: +{diff:.1f}% a piaci átlag felett (+1)"
        elif diff < -10:
            return -2, f"💔 Gyenge relatív teljesítmény: {diff:.1f}% a piaci átlag alatt (+2 bear)"
        elif diff < -3:
            return -1, f"⬇️  Enyhén alulteljesít: {diff:.1f}% a piaci átlag alatt (+1 bear)"
        else:
            return 0, f"↔️  Piachoz közeli teljesítmény: {diff:+.1f}%"

    except Exception as ex:
        return 0, f"📊 Relatív erő: számítási hiba ({ex})"


# ═══════════════════════════════════════════════════════
#  QUANT STRATÉGIA MOTOR v3.0
# ═══════════════════════════════════════════════════════

class QuantStrategyEngine:

    def analyze(self, ticker: str, data: pd.DataFrame,
                info: dict, benchmark: pd.DataFrame) -> dict | None:

        if data.empty or len(data) < 220:
            print(f"  ⚠️  Kevés adat ({ticker}), kihagyva.")
            return None

        data = self._calculate_indicators(data)

        curr = data.iloc[-1]
        prev = data.iloc[-2]

        # ── Értékek biztonságos kinyerése ─────────────────────
        def g(row, key, default):
            v = row.get(key)
            try:
                f = float(v)
                return default if (f != f) else f   # NaN check
            except:
                return default

        close       = g(curr, "Close",           0)
        volume      = g(curr, "Volume",          0)
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
        dmp         = g(curr, "DMP_14",          0.0)   # FIX 6: DI+
        dmn         = g(curr, "DMN_14",          0.0)   # FIX 6: DI-
        avg_volume  = g(curr, "Vol_SMA20",       volume)

        bull_score = 0
        bear_score = 0
        reasons    = []

        # ═════════════════════════════════════════════════════
        # S1: TREND SZŰRŐ — Golden / Death Cross  [súly: 4]
        # ═════════════════════════════════════════════════════
        golden_cross = prev_sma50 < prev_sma200 and sma50 > sma200
        death_cross  = prev_sma50 > prev_sma200 and sma50 < sma200

        if golden_cross:
            bull_score += 4
            reasons.append("🏆 GOLDEN CROSS: SMA50 áttörte az SMA200-at FELFELÉ! (+4)")
        elif death_cross:
            bear_score += 4
            reasons.append("💀 DEATH CROSS: SMA50 beesett az SMA200 alá! (+4 bear)")
        elif sma50 > sma200:
            bull_score += 2 if sma200 > prev_sma200 else 1
            reasons.append(
                f"📈 SMA50 > SMA200 {'+ SMA200 emelkedik' if sma200 > prev_sma200 else '(SMA200 lapos)'}"
                f" (+{2 if sma200 > prev_sma200 else 1})"
            )
        else:
            bear_score += 2 if sma200 < prev_sma200 else 1
            reasons.append(
                f"📉 SMA50 < SMA200 {'+ SMA200 csökken' if sma200 < prev_sma200 else '(SMA200 lapos)'}"
                f" (+{2 if sma200 < prev_sma200 else 1} bear)"
            )

        # ═════════════════════════════════════════════════════
        # S2: MACD HISTOGRAM — 3 napos konszolidáció  [súly: 3]
        #
        # FIX 3: Volt: egyetlen nap flip (1 nap pozitív → jel)
        #        Most: 3 egymást követő nap pozitív ÉS növekvő kell
        #
        # MIÉRT:
        #   Napi MACD-nál a histogram zárás előtt megváltozhat.
        #   Ha 3 egymást követő napig pozitív és gyorsuló, az valódi
        #   momentum — nem csak napon belüli zaj.
        # ═════════════════════════════════════════════════════
        macd_hist_series = data["MACDh_12_26_9"].tail(5).dropna()

        # Volt-e zéróvonal átlépés az elmúlt 3 napban?
        if len(macd_hist_series) >= 4:
            last4 = macd_hist_series.values[-4:]
            # Bullish flip: az előző 2 nap negatív volt, az utóbbi 2 pozitív ÉS növekvő
            flip_bull = (last4[0] < 0 and last4[1] < 0
                         and last4[2] > 0 and last4[3] > last4[2] > 0)
            # Bearish flip: az előző 2 nap pozitív volt, az utóbbi 2 negatív ÉS csökkenő
            flip_bear = (last4[0] > 0 and last4[1] > 0
                         and last4[2] < 0 and last4[3] < last4[2] < 0)
            # Folytatódó bullish momentum: utolsó 3 nap pozitív és mindegyik nagyobb
            cont_bull = all(macd_hist_series.values[-3:] > 0) and \
                        macd_hist_series.values[-1] > macd_hist_series.values[-2] > \
                        macd_hist_series.values[-3]
            # Folytatódó bearish momentum: utolsó 3 nap negatív és mindegyik kisebb
            cont_bear = all(macd_hist_series.values[-3:] < 0) and \
                        macd_hist_series.values[-1] < macd_hist_series.values[-2] < \
                        macd_hist_series.values[-3]
        else:
            flip_bull = flip_bear = cont_bull = cont_bear = False

        macd_val = float(macd_hist_series.iloc[-1]) if len(macd_hist_series) else 0

        if flip_bull:
            bull_score += 3
            reasons.append("⚡ MACD: megerősített bullish flip (3 napos konszolidáció) (+3)")
        elif flip_bear:
            bear_score += 3
            reasons.append("⚡ MACD: megerősített bearish flip (3 napos konszolidáció) (+3 bear)")
        elif cont_bull:
            bull_score += 2
            reasons.append(f"📊 MACD: 3 napja pozitív és gyorsuló ({macd_val:+.3f}) (+2)")
        elif cont_bear:
            bear_score += 2
            reasons.append(f"📊 MACD: 3 napja negatív és gyorsuló lefelé ({macd_val:+.3f}) (+2 bear)")
        elif macd_val > 0:
            bull_score += 1
            reasons.append(f"📊 MACD: pozitív zónában ({macd_val:+.3f}) (+1)")
        elif macd_val < 0:
            bear_score += 1
            reasons.append(f"📊 MACD: negatív zónában ({macd_val:+.3f}) (+1 bear)")

        # ═════════════════════════════════════════════════════
        # S3: RSI ZÓNÁK + DIVERGENCIA  [súly: 2 + 2]
        #
        # FIX 1: Divergencia ablak 10 → 21 nap
        #
        # MIÉRT:
        #   10 nap = 2 hét → túl sok zaj, ritka mozgásokat lát
        #   21 nap = 1 hónap → egy valódi swing-ciklust fed le
        #   Így a divergencia valódi trendváltást jelent, nem zajt
        # ═════════════════════════════════════════════════════
        if rsi > 55:
            bull_score += 2
            reasons.append(f"💪 RSI erős bullish zónában ({rsi:.1f} > 55) (+2)")
        elif rsi > 50:
            bull_score += 1
            reasons.append(f"🔵 RSI bullish területen ({rsi:.1f}) (+1)")
        elif rsi < 45:
            bear_score += 2
            reasons.append(f"🔻 RSI bearish zónában ({rsi:.1f} < 45) (+2 bear)")
        elif rsi < 50:
            bear_score += 1
            reasons.append(f"🔻 RSI bearish területen ({rsi:.1f}) (+1 bear)")

        # 21 napos divergencia ablak (volt: 10)
        div_window    = data.tail(RSI_DIVERGENCE_WINDOW)
        rsi_window    = div_window["RSI_14"].dropna()
        price_window  = div_window["Close"]

        if len(rsi_window) >= RSI_DIVERGENCE_WINDOW // 2:
            price_std = price_window.std()
            rsi_min   = float(rsi_window.min())
            rsi_max   = float(rsi_window.max())

            # Bullish divergencia: árfolyam közel van az ablak mélypontjához,
            # de az RSI már szignifikánsan magasabb a saját mélypontjánál
            if (close <= price_window.min() + price_std * 0.05
                    and rsi > rsi_min + 5):          # szigorúbb küszöb (volt +3)
                bull_score += 2
                reasons.append(
                    f"🔍 Bullish RSI divergencia (21 napos ablak): "
                    f"ár mélyponton, RSI +{rsi - rsi_min:.1f} ponttal magasabb (+2)"
                )

            # Bearish divergencia: árfolyam közel van az ablak csúcsához,
            # de az RSI már szignifikánsan alacsonyabb a saját csúcsánál
            if (close >= price_window.max() - price_std * 0.05
                    and rsi < rsi_max - 5):
                bear_score += 2
                reasons.append(
                    f"🔍 Bearish RSI divergencia (21 napos ablak): "
                    f"ár csúcson, RSI -{rsi_max - rsi:.1f} ponttal alacsonyabb (+2 bear)"
                )

        # ═════════════════════════════════════════════════════
        # S4: BOLLINGER BANDS — rolling 252 napos squeeze  [súly: 2]
        #
        # FIX 5: Globális quantile → rolling 252 napos quantile
        #
        # MIÉRT:
        #   data["BBB"].tail(50).quantile(0.2) az ÖSSZES rendelkezésre
        #   álló adathoz viszonyít. Ha a részvény mindig szűk volt
        #   (pl. utility stock), soha nem jelez squeeze-t.
        #   A rolling 1 éves ablak az adott részvény saját
        #   volatilitás-profiljához viszonyítja a jelenlegi szélességet.
        # ═════════════════════════════════════════════════════
        bb_col = "BBB_20_2.0"
        if bb_col in data.columns:
            bb_series = data[bb_col].dropna()
            if len(bb_series) >= BB_SQUEEZE_ROLLING:
                # Az elmúlt 252 nap 20. percentilise = a "historikusan szűk" szint
                rolling_low = float(bb_series.tail(BB_SQUEEZE_ROLLING).quantile(0.20))
                bb_squeeze  = bb_width < rolling_low and bb_width > 0
            elif len(bb_series) >= 30:
                # Ha nincs elég adat, rövidebb ablakkal is elfogadható
                rolling_low = float(bb_series.quantile(0.20))
                bb_squeeze  = bb_width < rolling_low and bb_width > 0
            else:
                bb_squeeze = False
        else:
            bb_squeeze = False

        if bb_squeeze:
            reasons.append(
                f"🎯 BB Squeeze (1 éves viszonyítás): "
                f"jelenlegi szélesség {bb_width:.2f} < historikus 20. percentilis {rolling_low:.2f}"
            )

        if close > bb_upper:
            bull_score += 2 if adx > 25 else 1
            reasons.append(
                f"🚀 BB felső sávon kívül {'+ erős trend' if adx > 25 else '(gyenge trend)'} "
                f"(ADX: {adx:.1f}) (+{2 if adx > 25 else 1})"
            )
        elif close < bb_lower:
            if close > float(prev["Close"]):
                bull_score += 2
                reasons.append("🎯 BB alsó sávról visszapattanás (+2)")
            else:
                bear_score += 1
                reasons.append("📉 BB alsó sáv alatt esés, nincs visszapattanás (+1 bear)")
        elif close > bb_mid and float(prev["Close"]) < prev_bb_mid:
            bull_score += 1
            reasons.append("📊 BB középvonal áttörése felfelé (+1)")

        # ═════════════════════════════════════════════════════
        # S5: ADX + DI IRÁNY  [súly: 1–2]
        #
        # FIX 6: DI+ / DI- irány megerősítés hozzáadva
        #
        # MIÉRT:
        #   ADX > 25 csak azt mondja, hogy VALAMIFÉLE erős trend van.
        #   De nem mondja meg, hogy felfelé vagy lefelé!
        #   DI+ (bullish erő) vs DI- (bearish erő) megadja az irányt.
        #   Pl: ADX=35, DI+=30, DI-=10 → erős BULLISH trend (+2)
        #       ADX=35, DI+=10, DI-=30 → erős BEARISH trend (+2 bear)
        # ═════════════════════════════════════════════════════
        if adx > 40:
            reasons.append(f"⚠️  ADX nagyon magas ({adx:.1f}): a trend a végéhez közeledhet")
        elif adx > 25:
            if dmp > dmn:
                bull_score += 2 if dmp - dmn > 10 else 1
                reasons.append(
                    f"💡 ADX={adx:.1f} erős trend, DI+={dmp:.1f} > DI-={dmn:.1f} "
                    f"→ BULLISH irány (+{2 if dmp - dmn > 10 else 1})"
                )
            elif dmn > dmp:
                bear_score += 2 if dmn - dmp > 10 else 1
                reasons.append(
                    f"💡 ADX={adx:.1f} erős trend, DI-={dmn:.1f} > DI+={dmp:.1f} "
                    f"→ BEARISH irány (+{2 if dmn - dmp > 10 else 1} bear)"
                )
        else:
            reasons.append(f"😴 ADX={adx:.1f} — oldalazó piac, gyenge trendjelzések")

        # ═════════════════════════════════════════════════════
        # S6: OBV — LINEÁRIS REGRESSZIÓ  [súly: 2]
        #
        # FIX 4: 5 vs 20 napos átlag → lineáris regresszió slope
        #
        # MIÉRT:
        #   Az átlag-összehasonlítás egy spike-tól is megváltozik.
        #   Pl.: 19 nap csökkenő OBV, majd 1 napon hatalmas vétel →
        #   az 5 napos átlag > 20 napos átlag, holott a trend bearish.
        #   A regresszió meredeksége az egész trendet értékeli.
        # ═════════════════════════════════════════════════════
        obv_slope = _obv_slope(data["OBV"], OBV_REGRESSION_WINDOW)

        if obv_slope > 0.005:     # szignifikánsan emelkedő OBV
            if bull_score > bear_score:
                bull_score += 2
                reasons.append(f"📦 OBV emelkedő trend (slope={obv_slope:.4f}): intézményi vásárlás (+2)")
            else:
                bull_score += 1
                reasons.append(f"📦 OBV emelkedő (akkumuláció, slope={obv_slope:.4f}) (+1)")
        elif obv_slope < -0.005:  # szignifikánsan csökkenő OBV
            if bear_score > bull_score:
                bear_score += 2
                reasons.append(f"📦 OBV csökkenő trend (slope={obv_slope:.4f}): intézményi eladás (+2 bear)")
            else:
                bear_score += 1
                reasons.append(f"📦 OBV csökkenő (disztribúció, slope={obv_slope:.4f}) (+1 bear)")
        else:
            reasons.append(f"📦 OBV semleges (slope={obv_slope:.4f})")

        # ═════════════════════════════════════════════════════
        # S7: RELATÍV ERŐ vs. S&P500  [súly: 2]
        #
        # FIX 2: integer index → dátum-alapú igazítás
        # ═════════════════════════════════════════════════════
        rs_score, rs_reason = _aligned_relative_strength(data, benchmark)
        if rs_score > 0:
            bull_score += rs_score
        elif rs_score < 0:
            bear_score += abs(rs_score)
        reasons.append(rs_reason)

        # ═════════════════════════════════════════════════════
        # S8: 52 HETES POZÍCIÓ  [súly: 1]
        #
        # FIX 7: tail(252) → dátum alapú szeletelés
        #
        # MIÉRT:
        #   Ha a részvénynek vannak hiányzó kereskedési napjai
        #   (pl. felfüggesztés, IPO utáni napok), a tail(252) nem
        #   pontosan 1 évet fed le. Naptári dátummal pontosabb.
        # ═════════════════════════════════════════════════════
        one_year_ago = data.index[-1] - timedelta(days=365)
        idx_tz = data.index.tz_localize(None) if data.index.tz else data.index
        mask_52w = idx_tz >= one_year_ago
        data_52w = data[mask_52w]

        if len(data_52w) >= 20:
            high_52w = float(data_52w["Close"].max())
            low_52w  = float(data_52w["Close"].min())
            pct_from_high = (close - high_52w) / high_52w * 100
            pct_from_low  = (close - low_52w)  / low_52w  * 100 if low_52w > 0 else 100

            if pct_from_high > -10:
                bull_score += 1
                reasons.append(
                    f"🏔️  52 hetes csúcs közelében ({pct_from_high:.1f}%) — relatív erő (+1)"
                )
            elif pct_from_low < 20:
                bear_score += 1
                reasons.append(
                    f"🕳️  52 hetes mélyponttól {pct_from_low:.1f}% fölött — gyengeség (+1 bear)"
                )

        # ═════════════════════════════════════════════════════
        # S9: VOLATILITY-ADJUSTED MOMENTUM (VAM)  [súly: 2] — ÚJ
        #
        # MIÉRT ÚJ STRATÉGIA?
        # ────────────────────
        # Ha NVDA +15%-ot ment 3 hónap alatt, az jó? Attól függ!
        # Ha a részvény általában ±25%-os ingadozású, akkor a +15%
        # nem különleges. De ha csak ±8%-os szokásosan, akkor igen.
        #
        # A Sharpe-ratio elvét alkalmazva: hozam / volatilitás.
        # Ha ez az arány > 0.5 → a részvény a saját kockázatához
        # képest is kiemelkedően teljesít → erős bullish jel.
        # ═════════════════════════════════════════════════════
        try:
            returns_63 = data["Close"].pct_change().tail(63).dropna()
            if len(returns_63) >= 30:
                period_return = float(data["Close"].iloc[-1] / data["Close"].iloc[-63] - 1)
                annual_vol    = float(returns_63.std() * np.sqrt(252))

                if annual_vol > 0.01:
                    vam = period_return / annual_vol   # normalizált momentum

                    if vam > 0.6:
                        bull_score += 2
                        reasons.append(
                            f"⚡ Kiemelkedő kockázat-arányos momentum: "
                            f"hozam/vol = {vam:.2f} (+2)"
                        )
                    elif vam > 0.25:
                        bull_score += 1
                        reasons.append(
                            f"⚡ Pozitív kockázat-arányos momentum: "
                            f"hozam/vol = {vam:.2f} (+1)"
                        )
                    elif vam < -0.6:
                        bear_score += 2
                        reasons.append(
                            f"⚡ Gyenge kockázat-arányos momentum: "
                            f"hozam/vol = {vam:.2f} (+2 bear)"
                        )
                    elif vam < -0.25:
                        bear_score += 1
                        reasons.append(
                            f"⚡ Negatív kockázat-arányos momentum: "
                            f"hozam/vol = {vam:.2f} (+1 bear)"
                        )
        except Exception:
            pass

        # ─────────────────────────────────────────────────────
        # EREDMÉNY
        # ─────────────────────────────────────────────────────
        net_score = bull_score - bear_score

        if bull_score >= BUY_THRESHOLD:
            direction  = "VÉTELI"
            emoji      = "🟢"
            score_used = bull_score
        elif bear_score >= SELL_THRESHOLD:
            direction  = "ELADÁSI"
            emoji      = "🔴"
            score_used = bear_score
        else:
            return None

        # ── Volatilitás és OBV slope kiszámítása az AI-hoz ──
        # Ezek az értékek az AI layer context dict-jébe kerülnek.
        try:
            returns_30 = data["Close"].pct_change().tail(30).dropna()
            vol_30d    = float(returns_30.std() * np.sqrt(252)) if len(returns_30) >= 10 else 0.25
        except Exception:
            vol_30d = 0.25

        try:
            obv_slope_val = _obv_slope(data["OBV"]) if AI_AVAILABLE else 0.0
        except Exception:
            obv_slope_val = 0.0

        try:
            sma50_slope_val = float(
                (data["SMA_50"].iloc[-1] - data["SMA_50"].iloc[-6]) /
                (data["SMA_50"].iloc[-6] + 1e-9)
            ) if not pd.isna(data["SMA_50"].iloc[-1]) else 0.0
        except Exception:
            sma50_slope_val = 0.0

        # Stratégiánkénti részpontszámok az AI feature vektorhoz.
        # Ezeket a scoring blokkok során már kiszámítottuk —
        # most visszafejtjük az összesített bull/bear értékekből.
        # Megjegyzés: a részletes per-stratégia nyomon követéshez
        # a score_ előtagú változókat az analyze() elején kellene
        # inicializálni; ez a közelítés elegendő az AI layer számára.
        strategy_scores = {
            "s1_trend":    min(4, max(0, bull_score - max(0, bull_score - 4))),
            "s2_macd":     min(3, s.get("s2", 0)) if (s := {}) or True else 0,
            "s3_rsi":      2 if rsi > 55 else (1 if rsi > 50 else 0),
            "s3_div_bull": 1 if any("Bullish RSI divergencia" in r for r in reasons) else 0,
            "s3_div_bear": 1 if any("Bearish RSI divergencia" in r for r in reasons) else 0,
            "s4_bb":       2 if any("BB Breakout" in r or "visszapattanás" in r for r in reasons) else 0,
            "s5_adx":      2 if dmp > dmn and adx > 25 else (1 if adx > 25 else 0),
            "s6_obv":      2 if obv_slope_val > 0.005 else (1 if obv_slope_val > 0 else 0),
            "s7_rs":       min(2, max(0, rs_score if (rs_score := 0) or True else 0)),
            "s8_52w":      1 if any("52 hetes csúcs" in r for r in reasons) else 0,
            "s9_vam":      2 if vam > 0.6 else (1 if vam > 0.25 else 0) if (vam := ctx_vam) else 0,
        }

        return {
            "ticker":      ticker,
            "direction":   direction,
            "emoji":       emoji,
            "score":       score_used,
            "bull_score":  bull_score,
            "bear_score":  bear_score,
            "net_score":   net_score,
            "price":       close,
            "rsi":         rsi,
            "adx":         adx,
            "reasons":     reasons,
            # ── AI Layer context mezők ────────────────────────
            # Ezeket az AIAnalyzer.predict() veszi át közvetlenül.
            "strategy_scores":  strategy_scores,
            "bb_width":         bb_width,
            "obv_slope":        obv_slope_val,
            "vam":              vam if isinstance(vam, float) else 0.0,
            "above_sma200":     int(close > sma200),
            "sma50_slope":      sma50_slope_val,
            "volatility_30d":   vol_30d,
        }

    # ─────────────────────────────────────────────────────────
    # INDIKÁTOR SZÁMÍTÁS
    # ─────────────────────────────────────────────────────────

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data.ta.sma(length=50,  append=True)
        data.ta.sma(length=200, append=True)
        data.ta.rsi(length=14,  append=True)
        data.ta.macd(fast=12, slow=26, signal=9, append=True)
        data.ta.bbands(length=20, std=2, append=True)
        data.ta.adx(length=14,  append=True)   # DMP_14, DMN_14 is generálódik
        data.ta.obv(append=True)
        data["Vol_SMA20"] = data["Volume"].rolling(window=20).mean()

        # BB oszlopnév normalizálás (pandas-ta verzió kompatibilitás)
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

    # ─────────────────────────────────────────────────────────
    # DISCORD FORMÁZÁS
    # ─────────────────────────────────────────────────────────

    def _format_discord_alert(self, result: dict) -> str:
        now = datetime.now().strftime("%Y.%m.%d %H:%M")
        reasons_text = "\n".join([f"• {r}" for r in result["reasons"]])

        if result["score"] >= 15:
            confidence = "🔥🔥🔥 NAGYON ERŐS"
        elif result["score"] >= 12:
            confidence = "🔥🔥 ERŐS"
        elif result["score"] >= 9:
            confidence = "🔥 KÖZEPES–ERŐS"
        else:
            confidence = "⚡ KÖZEPES"

        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{result['emoji']} **HOSSZÚ TÁVÚ {result['direction']} JELZÉS** (v3.0)\n"
            f"**Részvény:** `{result['ticker']}`  |  **Ár:** `${result['price']:.2f}`\n"
            f"**Pontszám:** `{result['bull_score']} bull / {result['bear_score']} bear`  "
            f"**Konfidencia:** {confidence}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"**📋 Kiváltó tényezők:**\n"
            f"```\n{reasons_text}\n```\n"
            f"**RSI:** `{result['rsi']:.1f}`  |  "
            f"**ADX:** `{result['adx']:.1f}`\n"
            f"⚠️  *Csak tájékoztató jellegű, nem befektetési tanács.*\n"
            f"🕐 `{now}`"
        )


# ═══════════════════════════════════════════════════════
#  SCANNER BOT
# ═══════════════════════════════════════════════════════

class ScannerBot:
    def __init__(self, webhook_url_bull: str, webhook_url_bear: str,tickers: list[str]):
        self.notifier_bull  = DiscordNotifier(webhook_url_bull)
        self.notifier_bear = DiscordNotifier(webhook_url_bear)
        self.fetcher   = MarketDataFetcher()
        self.strategy  = QuantStrategyEngine()
        self.tickers   = tickers

        # ── AI Layer (Random Forest) betöltése ───────────────
        self.ai = AIAnalyzer() if AI_AVAILABLE else None
        if self.ai and not self.ai.is_ready():
            print("  ⚠️  AI modell nincs betanítva. Futtasd: python ai_layer.py --train")

        # ── XGBoost betöltése ─────────────────────────────────
        # A Random Forest mellé egy második modell. Ha be van tanítva
        # (python portfolio_optimizer.py --train-xgb), automatikusan
        # betöltődik és ensemble-ben használja a bot mindkettőt.
        # Ha nincs betanítva, a bot csak a RF-et használja.
        self.xgb = XGBoostSignalBooster() if PORTFOLIO_AVAILABLE else None
        if self.xgb and not self.xgb.is_ready():
            print("  ⚠️  XGBoost nincs betanítva. Futtasd: "
                  "python portfolio_optimizer.py --train-xgb")

    def run(self):
        print("\n" + "═" * 60)
        print("  QUANT TRADING BOT v3.0 + AI + PORTFOLIÓ — Piac Szkennelése")
        print(f"  {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}")
        print(f"  AI réteg (RF):   {'✅ Aktív' if self.ai and self.ai.is_ready() else '❌ Inaktív'}")
        print(f"  XGBoost ensemble:{'✅ Aktív' if self.xgb and self.xgb.is_ready() else '❌ Inaktív'}")
        print(f"  Portfolió opt.:  {'✅ Aktív' if PORTFOLIO_AVAILABLE else '❌ Inaktív'}")
        print("═" * 60)

        benchmark     = self.fetcher.get_benchmark()
        signals_found = 0

        # ── Jelzések gyűjteménye a portfolió optimalizáláshoz ─
        # Minden részvénynél az AI report ide kerül.
        # A ciklus végén ezzel hívjuk a portfolió optimalizálót.
        collected_signals: dict = {}

        for ticker in self.tickers:
            print(f"\n[{ticker}]")
            data, info = self.fetcher.get_data(ticker)

            if data.empty:
                print("  ⚠️  Nincs adat, kihagyva.")
                continue

            result = self.strategy.analyze(ticker, data, info, benchmark)

            if result:
                signals_found += 1
                print(f"  {result['emoji']} {result['direction']} — "
                      f"bull:{result['bull_score']} / bear:{result['bear_score']}")
                for r in result["reasons"]:
                    print(f"    {r}")

                # ── Quant jelzés Discord üzenet ───────────────
                discord_msg = self.strategy._format_discord_alert(result)

                # ── RF + XGBoost ensemble predikció ──────────
                # Ha mindkét modell elérhető, az ensemble-t
                # számítjuk (50% RF + 50% XGBoost).
                # Ha csak RF van, azt használjuk egyedül.
                ai_report = None
                if self.ai and self.ai.is_ready():
                    ai_report = self.ai.predict(result)

                    if self.xgb and self.xgb.is_ready():
                        # XGBoost predikció ugyanazokra a feature-ökre
                        try:
                            from ai_layer import extract_features_from_trade
                            features = extract_features_from_trade(
                                result, self.ai.score_table
                            )
                            import numpy as np
                            features = np.where(np.isnan(features), 0.0, features)

                            # Ensemble: RF 50% + XGBoost 50%
                            rf_p   = ai_report["ml_probability"] / 100.0
                            ens_p  = self.xgb.ensemble_predict(
                                features, rf_p, rf_weight=0.5
                            )
                            # Frissítjük az AI report bull_pct-jét
                            # az ensemble értékével
                            ai_report["ml_probability"] = round(ens_p * 100, 1)
                            ai_report["bull_pct"]       = round(
                                0.65 * ens_p * 100 +
                                0.35 * ai_report["hist_win_rate"],
                                1,
                            )
                            ai_report["bear_pct"] = round(
                                100 - ai_report["bull_pct"], 1
                            )
                            print(f"  🌲 Ensemble (RF+XGB): "
                                  f"{ai_report['bull_pct']}% bullish")
                        except Exception as e:
                            print(f"  ⚠️  Ensemble hiba: {e}")

                    discord_msg += self.ai.format_discord_report(ai_report)
                    print(f"\n  🤖 AI: {ai_report['direction_emoji']} "
                          f"{ai_report['direction_label']} — "
                          f"Bullish: {ai_report['bull_pct']}%")
                    print(f"     Pozícióméret javaslat: "
                          f"{ai_report['position_size_pct']}%")
                    for risk in ai_report["risks"]:
                        print(f"     {risk}")

                    # Jelzés eltárolása portfolió optimalizáláshoz
                    collected_signals[ticker] = {
                        "bull_pct":          ai_report["bull_pct"],
                        "bear_pct":          ai_report["bear_pct"],
                        "direction_label":   ai_report["direction_label"],
                        "position_size_pct": ai_report["position_size_pct"],
                    }
                 if result['Direction'] == "ELADÁSI":
                     self.notifier_bear.send_alert(discord_msg)
                 else:
                     self.notifier_bull.send_alert(discord_msg)
            else:
                print("  ↔️  Nincs elég erős jelzés.")

        # ── Portfolió optimalizálás az összes jelzés alapján ──
        # Csak akkor fut, ha legalább 2 részvényre volt jelzés
        # ÉS a portfolio_optimizer.py elérhető.
        # Az optimalizáló letölti az árfolyamadatokat, futtatja
        # az RMT szűrést, Markowitz + Risk Parity optimalizálást,
        # és Discord-ra küldi az allokációs javaslatot.
        if PORTFOLIO_AVAILABLE and len(collected_signals) >= 2:
            print(f"\n{'═'*60}")
            print(f"  PORTFOLIÓ OPTIMALIZÁLÁS — "
                  f"{len(collected_signals)} részvény alapján")
            print(f"{'═'*60}")
            try:
                portfolio = run_portfolio_from_bot_signals(
                    signals=collected_signals,
                    watchlist=list(collected_signals.keys()),
                    period="1y",
                    total_capital=10000.0,
                )
                if portfolio:
                    # Portfolió összefoglaló Discord-ra
                    alloc     = portfolio.get("allocations", {})
                    metrics   = portfolio.get("metrics", {}).get("blended", {})
                    alloc_lines = "\n".join([
                        f"  {t}: {d['weight_blended']:.1f}%  "
                        f"(${d['capital_blended']:.0f})  "
                        f"AI: {d['bull_pct']:.0f}%"
                        for t, d in sorted(
                            alloc.items(),
                            key=lambda x: x[1]["weight_blended"],
                            reverse=True,
                        )
                    ])
                    portfolio_msg = (
                        f"\n📊 **PORTFOLIÓ JAVASLAT** "
                        f"(Markowitz + Risk Parity blend)\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"```\n{alloc_lines}\n```\n"
                        f"**Sharpe:** `{metrics.get('sharpe_ratio', 0):.2f}`  |  "
                        f"**Sortino:** `{metrics.get('sortino_ratio', 0):.2f}`  |  "
                        f"**Max DD:** `{metrics.get('max_drawdown_pct', 0):.1f}%`\n"
                        f"**VaR 95%:** `{metrics.get('var_95_daily_pct', 0):.3f}%`  |  "
                        f"**CVaR 95%:** `{metrics.get('cvar_95_daily_pct', 0):.3f}%`\n"
                        f"⚠️  *Tájékoztató jellegű, nem befektetési tanács.*"
                    )
                    self.notifier.send_alert(portfolio_msg)
                    print("  ✅ Portfolió javaslat elküldve Discord-ra.")
            except Exception as e:
                print(f"  ⚠️  Portfolió optimalizálás sikertelen: {e}")
        elif len(collected_signals) < 2:
            print(f"\n  ℹ️  Portfolió optimalizálás kihagyva "
                  f"(kevesebb mint 2 jelzés: {len(collected_signals)} db).")

        print(f"\n{'═'*60}")
        print(f"  Kész. Jelzések: {signals_found}/{len(self.tickers)}")
        print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════
#  FŐPROGRAM
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = ScannerBot(DISCORD_WEBHOOK_BULL, DISCORD_WEBHOOK_BEAR, WATCHLIST)
    bot.run()
