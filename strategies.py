"""Kereskedési stratégiák — Session-aware, dinamikus SL/TP-vel.
v3: VWAP Scalp (original proven), Session Breakout/EMA (proven), új Momentum strat.
TrendContinuation és VWAPReversal eltávolítva — 0% WR minden futtatásban.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np


class Signal(Enum):
    HOLD = 0
    BUY = 1
    SELL = -1


@dataclass
class TradeSetup:
    """Trade belépési adatok SL/TP specifikációval."""
    signal: Signal
    entry_price: float
    sl_price: float
    tp1_price: float       # Első cél (50% partial)
    tp2_price: float       # Második cél (50%)
    partial_ratio: float = 0.5
    trailing_activate: float = 0.0
    trailing_distance: float = 0.0
    breakeven_after_r: float = 0.0


class BaseStrategy:
    """Alap stratégia — session beállításokkal."""
    def __init__(self, name):
        self.name = name
        self.params = {}

    def get_session(self, ts) -> str:
        t_str = ts.strftime("%H:%M") if hasattr(ts, 'strftime') else str(ts)
        h = int(t_str.split(":")[0])
        m = int(t_str.split(":")[1])
        mins = h * 60 + m

        if mins < 570:
            return "premarket"
        elif mins < 630:
            return "open"
        elif mins < 750:
            return "mid_morning"
        elif mins < 810:
            return "lunch"
        elif mins < 960:
            return "afternoon"
        else:
            return "afterhours"

    def get_session_params(self, ts, atr) -> dict:
        """Session-alapú SL/TP paraméterek. v3: optimalizált SL/TP arányok."""
        session = self.get_session(ts)
        atr = max(atr, 5)

        params = {
            "open": {
                "sl_atr": 2.0, "tp_rr": 2.5, "trail_atr": 0,
                "min_trade_atr": 8,
            },
            "mid_morning": {
                "sl_atr": 2.0, "tp_rr": 2.5, "trail_atr": 0,
                "min_trade_atr": 6,
            },
            "lunch": {
                "sl_atr": 1.8, "tp_rr": 2.0, "trail_atr": 3.0,
                "min_trade_atr": 5,
            },
            "afternoon": {
                "sl_atr": 2.0, "tp_rr": 2.5, "trail_atr": 0,
                "min_trade_atr": 7,
            },
            "premarket": {
                "sl_atr": 2.5, "tp_rr": 3.0, "trail_atr": 4.0,
                "min_trade_atr": 10,
            },
            "afterhours": {
                "sl_atr": 2.0, "tp_rr": 2.0, "trail_atr": 0,
                "min_trade_atr": 6,
            },
        }

        return params.get(session, params["mid_morning"])

    def generate_signals(self, df):
        raise NotImplementedError


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout — napi ORB reset.
    v3: Nagyobb SL (3x ATR), volume filter relaxálva.
    """

    def __init__(self, orb_window=15, sl_atr_mult=3.0, tp_rr=3.0):
        super().__init__("ORB_15min")
        self.params = {"orb_window": orb_window,
                       "sl_atr_mult": sl_atr_mult, "tp_rr": tp_rr}

    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index, dtype=object)
        last_date = None
        orb_high = None
        orb_low = None
        orb_active = False
        orb_end_idx = 0

        for i in range(len(df)):
            t = df.index[i]
            current_date = t.date()

            if current_date != last_date:
                orb_high = None
                orb_low = None
                orb_active = False
                orb_end_idx = 0
                last_date = current_date

            # ORB zóna 09:30-09:45 ET
            if t.hour == 9 and t.minute >= 29 and orb_high is None:
                orb_end_idx = i
                for j in range(i, len(df)):
                    tj = df.index[j]
                    if tj.hour == 9 and tj.minute >= 45:
                        orb_end_idx = j
                        break
                orb_range = df.iloc[i:orb_end_idx + 1]
                if not orb_range.empty:
                    orb_high = orb_range["High"].max()
                    orb_low = orb_range["Low"].min()
                    orb_active = True

            if orb_active and i > orb_end_idx:
                # Csak az első 60 percben ORB után (09:45-10:45)
                t_minutes = t.hour * 60 + t.minute
                if t_minutes > 645:  # 10:45 után nincs ORB trade
                    continue

                atr = df.get("ATR_14", pd.Series(10)).iloc[i] if "ATR_14" in df.columns else 10
                if pd.isna(atr) or atr <= 0:
                    atr = 10

                session_p = self.get_session_params(t, atr)
                if atr < session_p["min_trade_atr"]:
                    continue

                close = df["Close"].iloc[i]
                vwap = df["VWAP"].iloc[i] if "VWAP" in df.columns else close
                vol = df["Volume"].iloc[i]
                vol_avg = df["Volume"].iloc[i-5:i].mean() if i >= 5 else df["Volume"].iloc[:i].mean()

                # BUY: breakout above ORB high + VWAP felett + volume
                if (close > orb_high and
                    close > vwap and
                    vol > vol_avg * 1.1):
                    signals.iloc[i] = Signal.BUY
                    orb_active = False

                # SELL: breakdown below ORB low + VWAP alatt + volume
                elif (close < orb_low and
                      close < vwap and
                      vol > vol_avg * 1.1):
                    signals.iloc[i] = Signal.SELL
                    orb_active = False

        return signals


class VWAPScalpStrategy(BaseStrategy):
    """VWAP Scalping — RSI + BB mean reversion.
    v3: Visszaállítva az eredeti működő paraméterekre (RSI 25/75, SL 1.2, TP 2.0).
    """

    def __init__(self, rsi_period=7, rsi_oversold=25, rsi_overbought=75,
                 sl_atr=1.2, tp_rr=2.0, max_hold=30):
        super().__init__("VWAP_Scalp")
        self.params = {"rsi_period": rsi_period, "rsi_oversold": rsi_oversold,
                       "rsi_overbought": rsi_overbought, "sl_atr": sl_atr,
                       "tp_rr": tp_rr, "max_hold": max_hold}

    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index, dtype=object)
        rsi_col = f"RSI_{self.params['rsi_period']}"
        bb_low = "BB_20_2.0_l"
        bb_high = "BB_20_2.0_u"

        for i in range(1, len(df)):
            if rsi_col not in df.columns or bb_low not in df.columns:
                continue

            close = df["Close"].iloc[i]
            vwap = df["VWAP"].iloc[i] if "VWAP" in df.columns else close
            atr = df.get("ATR_14", pd.Series(10)).iloc[i] if "ATR_14" in df.columns else 10
            if pd.isna(atr) or atr <= 0:
                atr = 10

            session_p = self.get_session_params(df.index[i], atr)
            if atr < session_p["min_trade_atr"]:
                continue

            # LONG: VWAP alatt, BB alsó sáv, RSI túladott + bullish candle
            if (close < vwap and
                close <= df[bb_low].iloc[i] * 1.005 and
                df[rsi_col].iloc[i] < self.params["rsi_oversold"] and
                df["Close"].iloc[i] > df["Close"].iloc[i-1]):
                signals.iloc[i] = Signal.BUY

            # SHORT: VWAP felett, BB felső sáv, RSI túlvett + bearish candle
            elif (close > vwap and
                  close >= df[bb_high].iloc[i] * 0.995 and
                  df[rsi_col].iloc[i] > self.params["rsi_overbought"] and
                  df["Close"].iloc[i] < df["Close"].iloc[i-1]):
                signals.iloc[i] = Signal.SELL

        return signals


class EMACrossStrategy(BaseStrategy):
    """9/21 EMA Crossover — session-aware SL/TP.
    v3: Szélesebb SL/TP (3R), MACD filter.
    """

    def __init__(self, fast=9, slow=21, sl_atr=2.5, tp_rr=3.0):
        super().__init__("EMA_Cross")
        self.params = {"fast": fast, "slow": slow,
                       "sl_atr": sl_atr, "tp_rr": tp_rr}

    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index, dtype=object)
        fast_col = f"EMA_{self.params['fast']}"
        slow_col = f"EMA_{self.params['slow']}"

        for i in range(1, len(df)):
            if fast_col not in df.columns or slow_col not in df.columns:
                continue

            vwap = df["VWAP"].iloc[i] if "VWAP" in df.columns else df["Close"].iloc[i]
            close = df["Close"].iloc[i]
            atr = df.get("ATR_14", pd.Series(10)).iloc[i] if "ATR_14" in df.columns else 10
            if pd.isna(atr) or atr <= 0:
                atr = 10

            session_p = self.get_session_params(df.index[i], atr)
            if atr < session_p["min_trade_atr"]:
                continue

            macd_hist = df["MACD_hist"].iloc[i] if "MACD_hist" in df.columns else 0

            # Bullish crossover + VWAP filter + MACD momentum
            if (df[fast_col].iloc[i] > df[slow_col].iloc[i] and
                df[fast_col].iloc[i-1] <= df[slow_col].iloc[i-1] and
                close > vwap and
                macd_hist > 0):
                signals.iloc[i] = Signal.BUY

            # Bearish crossover + VWAP filter + MACD momentum
            elif (df[fast_col].iloc[i] < df[slow_col].iloc[i] and
                  df[fast_col].iloc[i-1] >= df[slow_col].iloc[i-1] and
                  close < vwap and
                  macd_hist < 0):
                signals.iloc[i] = Signal.SELL

        return signals


class EnhancedSessionEMA(BaseStrategy):
    """Továbbfejlesztett EMA Cross — session-aware belépés, RSI + MACD filter.
    v3: Megtartva a bevált lazább paramétereket.
    """
    def __init__(self, fast=9, slow=21, rsi_period=14,
                 min_ema_spread=0.15, vol_mult=0.8):
        super().__init__("Session_EMA")
        self.params = {
            "fast": fast, "slow": slow,
            "rsi_period": rsi_period,
            "min_ema_spread": min_ema_spread,
            "vol_mult": vol_mult,
        }

    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index, dtype=object)
        fast_col = f"EMA_{self.params['fast']}"
        slow_col = f"EMA_{self.params['slow']}"
        rsi_col = f"RSI_{self.params['rsi_period']}"
        macd_hist = "MACD_hist"

        for i in range(2, len(df)):
            ts = df.index[i]
            session = self.get_session(ts)

            if session in ("premarket", "afterhours"):
                continue

            if fast_col not in df.columns or slow_col not in df.columns:
                continue

            close = df["Close"].iloc[i]
            fast_val = df[fast_col].iloc[i]
            slow_val = df[slow_col].iloc[i]
            fast_prev = df[fast_col].iloc[i-1]
            slow_prev = df[slow_col].iloc[i-1]
            atr = df.get("ATR_14", pd.Series(10)).iloc[i] if "ATR_14" in df.columns else 10
            if pd.isna(atr) or atr <= 0:
                atr = 10

            session_p = self.get_session_params(ts, atr)
            if atr < session_p["min_trade_atr"]:
                continue

            ema_spread = abs(fast_val - slow_val)
            if ema_spread < self.params["min_ema_spread"]:
                continue

            if i >= 5:
                avg_vol = df["Volume"].iloc[i-5:i].mean()
                if avg_vol > 0 and df["Volume"].iloc[i] < avg_vol * self.params["vol_mult"]:
                    continue

            rsi_val = df[rsi_col].iloc[i] if rsi_col in df.columns else 50
            macd_val = df[macd_hist].iloc[i] if macd_hist in df.columns else 0
            macd_prev = df[macd_hist].iloc[i-1] if macd_hist in df.columns else 0

            vwap = df["VWAP"].iloc[i] if "VWAP" in df.columns else close

            # Bullish
            rsi_bullish = rsi_val > 50
            macd_bullish = macd_val > 0 and macd_val > macd_prev

            if (fast_val > slow_val and fast_prev <= slow_prev and
                close > vwap and
                (rsi_bullish or macd_bullish)):
                signals.iloc[i] = Signal.BUY

            # Bearish
            rsi_bearish = rsi_val < 50
            macd_bearish = macd_val < 0 and macd_val < macd_prev

            if (fast_val < slow_val and fast_prev >= slow_prev and
                close < vwap and
                (rsi_bearish or macd_bearish)):
                signals.iloc[i] = Signal.SELL

        return signals


class SessionBreakoutStrategy(BaseStrategy):
    """Session alapú breakout trading.
    v3: Megtartva a bevált paramétereket (62.5% WR, Sharpe 5.12).
    """
    def __init__(self, lookback_bars=30, vol_mult=1.15, sl_atr=2.0, tp_rr=2.5):
        super().__init__("Session_Breakout")
        self.params = {"lookback_bars": lookback_bars,
                       "vol_mult": vol_mult,
                       "sl_atr": sl_atr,
                       "tp_rr": tp_rr}

    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index, dtype=object)
        lookback = self.params["lookback_bars"]
        vol_mult = self.params["vol_mult"]

        last_session = None
        session_range_high = None
        session_range_low = None

        for i in range(lookback + 1, len(df)):
            ts = df.index[i]
            session = self.get_session(ts)

            if session in ("premarket", "afterhours"):
                last_session = None
                continue

            atr = df.get("ATR_14", pd.Series(10)).iloc[i] if "ATR_14" in df.columns else 10
            if pd.isna(atr) or atr <= 0:
                atr = 10

            session_p = self.get_session_params(ts, atr)
            if atr < session_p["min_trade_atr"]:
                continue

            # Session range reset
            if session != last_session:
                session_range_high = None
                session_range_low = None
                last_session = session

            if session_range_high is None:
                lookback_window = df.iloc[max(0, i-lookback):i+1]
                session_range_high = lookback_window["High"].max()
                session_range_low = lookback_window["Low"].min()
                continue

            lookback_window = df.iloc[i-lookback:i]
            range_high = lookback_window["High"].max()
            range_low = lookback_window["Low"].min()
            range_size = range_high - range_low

            if range_size < atr * 0.5:
                continue

            close = df["Close"].iloc[i]
            vol = df["Volume"].iloc[i]
            avg_vol = lookback_window["Volume"].mean() if i >= lookback else vol
            vwap = df["VWAP"].iloc[i] if "VWAP" in df.columns else close

            if (close > range_high and
                vol > avg_vol * vol_mult and
                close > vwap):
                signals.iloc[i] = Signal.BUY

            elif (close < range_low and
                  vol > avg_vol * vol_mult and
                  close < vwap):
                signals.iloc[i] = Signal.SELL

        return signals


class VWAPMomentumStrategy(BaseStrategy):
    """VWAP Momentum — trend követés VWAP breakout-ra.
    ÚJ: Nem mean reversion, hanem trend. VWAP-tól való elszakadás + momentum.
    - VWAP deviation > 0.3% = erős trend
    - MACD növekvő = momentum megerősítés
    - Széles SL/TP (2.5x ATR / 3R) trend trade-hez
    - Trailing stop aktiválás 2R után
    """
    def __init__(self, dev_pct=0.3, rsi_period=7, sl_atr=2.5, tp_rr=3.0):
        super().__init__("VWAP_Momentum")
        self.params = {"dev_pct": dev_pct, "rsi_period": rsi_period,
                       "sl_atr": sl_atr, "tp_rr": tp_rr}

    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index, dtype=object)
        rsi_col = f"RSI_{self.params['rsi_period']}"
        dev_pct = self.params["dev_pct"]

        for i in range(3, len(df)):
            ts = df.index[i]
            session = self.get_session(ts)

            if session in ("premarket", "afterhours"):
                continue

            if "VWAP" not in df.columns or rsi_col not in df.columns:
                continue

            close = df["Close"].iloc[i]
            vwap = df["VWAP"].iloc[i]
            atr = df.get("ATR_14", pd.Series(10)).iloc[i] if "ATR_14" in df.columns else 10
            if pd.isna(atr) or atr <= 0:
                atr = 10

            session_p = self.get_session_params(ts, atr)
            if atr < session_p["min_trade_atr"]:
                continue

            macd_hist = df["MACD_hist"].iloc[i] if "MACD_hist" in df.columns else 0
            macd_prev = df["MACD_hist"].iloc[i-1] if "MACD_hist" in df.columns else 0
            rsi = df[rsi_col].iloc[i]

            deviation_pct = (close - vwap) / vwap * 100

            # LONG: price above VWAP by dev_pct, MACD rising, RSI > 50
            if (deviation_pct > dev_pct and
                macd_hist > macd_prev and
                macd_hist > 0 and
                rsi > 55):
                signals.iloc[i] = Signal.BUY

            # SHORT: price below VWAP by dev_pct, MACD falling, RSI < 50
            elif (deviation_pct < -dev_pct and
                  macd_hist < macd_prev and
                  macd_hist < 0 and
                  rsi < 45):
                signals.iloc[i] = Signal.SELL

        return signals
