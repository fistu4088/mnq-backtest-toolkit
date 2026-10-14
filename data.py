"""Adat letöltés és tisztítás — MNQ historikus adatok timezone fix-szel."""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_mnq(days=365, interval="5m"):
    """Letölti az MNQ historikus adatokat yfinance-ból.
    Args:
        days: Hány napnyi adat
        interval: '1m', '5m', '15m', '30m', '1h'
    Returns:
        DataFrame OHLCV adatokkal
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # yfinance 7 nap limit 1m-es/5m-es intervallumra
    all_data = []
    if interval in ["1m", "2m", "5m"]:
        chunk_days = 7
    elif interval == "15m":
        chunk_days = 60
    else:
        chunk_days = min(days, 730)

    # Több chunk-ban töltünk le, ha kell
    for i in range(0, min(days, 730), chunk_days):
        end = datetime.now() - timedelta(days=i)
        start = end - timedelta(days=chunk_days)
        try:
            ticker = yf.Ticker("MNQ=F")
            df = ticker.history(start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"),
                                interval=interval)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Hiba a letöltéskor ({i} nap): {e}")

    if all_data:
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
    else:
        df = pd.DataFrame()

    return _clean_df(df)


def _clean_df(df):
    """Adattisztítás: hiányzó értékek, duplikált index, üres sorok.
    Minden időpontot US/Eastern-re konvertál (yfinance UTC-ben adja)."""
    if df.empty:
        return df

    # yfinance UTC-t ad, át kell tenni US/Eastern-re
    try:
        if df.index.tz is None:
            # Ha nincs timezone, feltételezzük hogy UTC (yfinance default)
            df.index = df.index.tz_localize("UTC", ambiguous="infer")
        df.index = df.index.tz_convert("US/Eastern")
    except Exception as e:
        print(f"Timezone konverzió hiba: {e}")

    # Csak a kereskedési órák (09:30-16:00 ET) — használjuk a .time() metódust
    try:
        start_time = pd.Timestamp("09:30").time()
        end_time = pd.Timestamp("16:00").time()
        mask = (df.index.time >= start_time) & (df.index.time <= end_time)
        df = df[mask].copy()
    except Exception as e:
        print(f"Óra filter hiba: {e}")

    df.dropna(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    rename_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }
    # Csak a meglévő oszlopokat nevezzük át
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Numeric konverzió
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    df.dropna(inplace=True)
    return df


def load_csv(path=None):
    """Betölt CSV fájlból."""
    if path is None:
        path = os.path.join(DATA_DIR, "mnq_historical.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _clean_df(df)
    return pd.DataFrame()


def save_csv(df, path=None):
    """Mentés CSV-be."""
    if path is None:
        path = os.path.join(DATA_DIR, "mnq_historical.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Adat mentve: {path} ({len(df)} sor)")


if __name__ == "__main__":
    df = download_mnq(days=30, interval="5m")
    if not df.empty:
        print(f"Letöltve: {len(df)} sor, {df.index[0]} - {df.index[-1]}")
        save_csv(df)
    else:
        print("Nem sikerült adatot letölteni.")
