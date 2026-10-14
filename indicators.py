"""Indikátor számítások — saját implementáció pandas segítségével.
(pandas-ta helyett, mert az nem kompatibilis a pandas 3.0-val)"""

import pandas as pd
import numpy as np


def ema(series, period):
    """Exponenciális mozgóátlag."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    """Egyszerű mozgóátlag."""
    return series.rolling(window=period).mean()


def vwap(high, low, close, volume):
    """Volume-Weighted Average Price — kumulatív (teljes ablak, nincs reset).

    Figyelem: daytradinghez a `vwap_session`-t használd, ez a változat
    az egész sorozatra egyetlen folytonos VWAP-ot ad (nincs napi reset).
    """
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def vwap_session(high, low, close, volume, index):
    """Session/napi anchorált VWAP — minden kereskedési nap elején resetel.

    A daytrade VWAP naponta (session-openkor, itt: napi dátumváltáskor)
    újraindul, így nem „csúszik át” az előző napok forgalma. A bemenő
    adat már RTH-ra szűrt (09:30–16:00 ET), ezért a dátumváltás = napi
    session határ.
    """
    tp = (high + low + close) / 3
    pv = (tp * volume).astype(float)
    vol = volume.astype(float)
    # Napi kulcs a DatetimeIndex-ből; timezone-tudatos Timestamp.date()-et ad.
    dates = pd.Series([d.date() for d in index], index=index)
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    out = cum_pv / cum_vol
    return out


def rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Kezdeti átlagok után EMA smoothing
    for i in range(period, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series, fast=12, slow=26, signal=9):
    """MACD indikátor."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(series, period=20, std=2.0):
    """Bollinger sávok."""
    mid = sma(series, period)
    std_dev = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def atr(high, low, close, period=14):
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_all(df):
    """Az összes indikátor kiszámítása a DataFrame-en."""
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # EMA
    df["EMA_9"] = ema(c, 9)
    df["EMA_21"] = ema(c, 21)
    df["EMA_50"] = ema(c, 50)
    df["EMA_100"] = ema(c, 100)

    # VWAP — daytrade-hez session/napi anchorált (naponta resetel),
    # a régi kumulatív érték `VWAP_CUM` néven megmarad audit/kontroll célra.
    df["VWAP"] = vwap_session(h, l, c, v, df.index)
    df["VWAP_CUM"] = vwap(h, l, c, v)

    # RSI
    df["RSI_7"] = rsi(c, 7)
    df["RSI_14"] = rsi(c, 14)

    # MACD
    ml, sl, hist = macd(c, 12, 26, 9)
    df["MACD_12_26_9"] = ml
    df["MACD_signal"] = sl
    df["MACD_hist"] = hist

    # Bollinger
    bb_u, bb_m, bb_l = bollinger(c, 20, 2.0)
    df["BB_20_2.0_u"] = bb_u
    df["BB_20_2.0_m"] = bb_m
    df["BB_20_2.0_l"] = bb_l

    # ATR
    df["ATR_14"] = atr(h, l, c, 14)

    return df
