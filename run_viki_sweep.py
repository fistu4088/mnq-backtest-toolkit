#!/usr/bin/env python3
"""Viki strategy sweep — 2026-10-14.

4 new candidate strategies on MNQ=F 1h RTH data (2024-10..2026-10),
walk-forward: fit 2024-10-16..2025-10-13, OOS validate 2025-10-14..2026-10-13.

Engine pattern from run_paper_v1.py: 1 MNQ, point_value 2.0, EOD flatten,
no overnight holds, one position at a time per strategy.

Candidates:
  1. gap_fade      — daily open gap-fade with VWAP-reversion exit
  2. ema_pullback  — EMA100 regime pullback to EMA20 (UP/DOWN regime only)
  3. donchian      — Donchian N-day breakout with EMA100 regime filter
  4. atr_trail     — EMA9/21 cross + ATR chandelier trailing stop

OOS gate: trades>=20, profit_factor>=1.3, net>0, maxDD<=50% of net.
"""
import json
import os
import sys
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data import _clean_df
from backtest.indicators import compute_all, ema
from backtest.metrics import compute_metrics
from backtest.strategies import Signal

POINT_VALUE = 2.0
SIZE = 1
MAX_RISK_USD = 1500.0         # research cap; 1h ATR stops are far wider than 5m paper-v1 ($120)
ENTRY_CUTOFF_MIN = 15 * 60 + 30  # no new entries on 15:30/16:00 bars
FLATTEN_MIN = 15 * 60 + 30       # flatten at last bar of day (16:00 close)
MAX_TRADES_DAY = 2

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "mnq_1h_2y_fresh.csv")
OUT_TRADES = os.path.join(ROOT, "data", "trades_viki_sweep.csv")
OUT_JSON = os.path.join(ROOT, "reports", "2026-10-14_viki_strategy_sweep.json")
OUT_MD = os.path.join(ROOT, "reports", "2026-10-14_viki_strategy_sweep.md")

FIT_START = pd.Timestamp("2024-10-16", tz="America/New_York")
FIT_END = pd.Timestamp("2025-10-13", tz="America/New_York")
OOS_START = pd.Timestamp("2025-10-14", tz="America/New_York")
OOS_END = pd.Timestamp("2026-10-14", tz="America/New_York")


def load_fresh():
    df = pd.read_csv(DATA, index_col=0)
    # CSV stamps are offset-aware with mixed EST/EDT offsets -> parse as UTC, convert to ET
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("America/New_York")
    df = _clean_df(df)  # RTH filter + column cleanup; tz already America/New_York
    return df


def minutes_of(ts) -> int:
    return ts.hour * 60 + ts.minute


# ---------------------------------------------------------------- strategies
def sig_gap_fade(df, gap_atr, sl_mult):
    """Fade the opening gap: gap > +gap_atr*ATR -> short, gap < -gap_atr*ATR -> long.
    Exit on VWAP cross (reversion), hard SL sl_mult*ATR, EOD flatten.
    Entry only on the first bar of the day."""
    n = len(df)
    sig = pd.Series(Signal.HOLD, index=df.index)
    sl_pts = pd.Series(0.0, index=df.index)
    atr = df["ATR_14"].astype(float)
    prev_close = df["Close"].shift(1)
    day_first = np.array([df.index[i].date() != df.index[i - 1].date() for i in range(n)])
    for i in range(1, n):
        if not day_first[i]:
            continue
        a = float(atr.iloc[i - 1])
        if not np.isfinite(a) or a <= 0:
            continue
        gap = float(df["Open"].iloc[i]) - float(prev_close.iloc[i - 1])
        if gap > gap_atr * a:
            sig.iloc[i] = Signal.SELL
            sl_pts.iloc[i] = sl_mult * a
        elif gap < -gap_atr * a:
            sig.iloc[i] = Signal.BUY
            sl_pts.iloc[i] = sl_mult * a
    return sig, sl_pts, "vwap"  # exit mode


def sig_ema_pullback(df, regime_lb, tol_atr, tp_rr, sl_mult=2.0):
    """Regime = EMA100 slope over regime_lb bars + close side of EMA100.
    UP: buy when bar touches EMA20 (low <= EMA20 + tol*ATR) and closes back above it.
    DOWN: mirror short."""
    n = len(df)
    sig = pd.Series(Signal.HOLD, index=df.index)
    sl_pts = pd.Series(0.0, index=df.index)
    e100 = df["EMA_100"].astype(float)
    e20 = ema(df["Close"].astype(float), 20)
    atr = df["ATR_14"].astype(float)
    slope = e100 - e100.shift(regime_lb)
    for i in range(max(regime_lb, 14) + 1, n):
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0 or not np.isfinite(slope.iloc[i]):
            continue
        c, lo, hi = float(df["Close"].iloc[i]), float(df["Low"].iloc[i]), float(df["High"].iloc[i])
        e20v, e100v = float(e20.iloc[i]), float(e100.iloc[i])
        if slope.iloc[i] > 0 and c > e100v and lo <= e20v + tol_atr * a and c > e20v:
            sig.iloc[i] = Signal.BUY
            sl_pts.iloc[i] = sl_mult * a
        elif slope.iloc[i] < 0 and c < e100v and hi >= e20v - tol_atr * a and c < e20v:
            sig.iloc[i] = Signal.SELL
            sl_pts.iloc[i] = sl_mult * a
    return sig, sl_pts, ("fixed", tp_rr)


def sig_donchian(df, days, sl_mult, tp_rr):
    """Donchian `days`-day breakout with EMA100 regime filter.
    Long: close > prior N-day high AND close > EMA100. Short mirrored."""
    n = len(df)
    sig = pd.Series(Signal.HOLD, index=df.index)
    sl_pts = pd.Series(0.0, index=df.index)
    bars = days * 7  # ~7 RTH 1h bars/day
    hh = df["High"].astype(float).rolling(bars, min_periods=bars).max().shift(1)
    ll = df["Low"].astype(float).rolling(bars, min_periods=bars).min().shift(1)
    e100 = df["EMA_100"].astype(float)
    atr = df["ATR_14"].astype(float)
    for i in range(bars + 1, n):
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0 or not np.isfinite(hh.iloc[i]):
            continue
        c = float(df["Close"].iloc[i])
        if c > float(hh.iloc[i]) and c > float(e100.iloc[i]):
            sig.iloc[i] = Signal.BUY
            sl_pts.iloc[i] = sl_mult * a
        elif c < float(ll.iloc[i]) and c < float(e100.iloc[i]):
            sig.iloc[i] = Signal.SELL
            sl_pts.iloc[i] = sl_mult * a
    return sig, sl_pts, ("fixed", tp_rr)


def sig_atr_trail(df, fast, slow, trail_atr):
    """EMA fast/slow cross entry, chandelier ATR trailing stop, no fixed TP."""
    n = len(df)
    sig = pd.Series(Signal.HOLD, index=df.index)
    sl_pts = pd.Series(0.0, index=df.index)
    ef = ema(df["Close"].astype(float), fast)
    es = ema(df["Close"].astype(float), slow)
    atr = df["ATR_14"].astype(float)
    for i in range(slow + 1, n):
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue
        if ef.iloc[i - 1] <= es.iloc[i - 1] and ef.iloc[i] > es.iloc[i]:
            sig.iloc[i] = Signal.BUY
            sl_pts.iloc[i] = trail_atr * a
        elif ef.iloc[i - 1] >= es.iloc[i - 1] and ef.iloc[i] < es.iloc[i]:
            sig.iloc[i] = Signal.SELL
            sl_pts.iloc[i] = trail_atr * a
    return sig, sl_pts, ("trail", trail_atr)


# ---------------------------------------------------------------- engine
def run_backtest(df, sig, sl_pts, exit_mode):
    """Generic engine: entry at signal-bar close, manage SL/TP/trail/vwap on
    later bars, EOD flatten, one position, max MAX_TRADES_DAY entries/day."""
    trades = []
    skips = Counter()
    open_trade = None
    last_date = None
    trades_today = 0

    def close_trade(ts, price, reason):
        nonlocal open_trade
        t = open_trade
        pts = (float(price) - t["entry"]) * t["direction"]
        pnl = round(pts * POINT_VALUE * SIZE, 2)
        risk = t["sl_pts"]
        trades.append({
            "entry_time": t["entry_time"], "exit_time": ts,
            "direction": "BUY" if t["direction"] == 1 else "SELL",
            "entry_price": t["entry"], "exit_price": float(price),
            "sl_pts": risk, "pnl": pnl,
            "r_multiple": round(pts / risk, 2) if risk > 0 else 0.0,
            "exit_reason": reason, "size": SIZE,
        })
        open_trade = None

    n = len(df)
    for i in range(n):
        ts = df.index[i]
        row = df.iloc[i]
        d = ts.date()
        mins = minutes_of(ts)
        is_last = (i == n - 1) or (df.index[i + 1].date() != d)

        if d != last_date:
            if open_trade is not None:
                close_trade(ts, row["Close"], "end_of_day")
            trades_today = 0
            last_date = d

        if open_trade is not None:
            high, low, close = float(row["High"]), float(row["Low"]), float(row["Close"])
            direction = open_trade["direction"]
            exit_price, reason = None, None
            if exit_mode == "vwap":
                v = float(row["VWAP"])
                if direction == 1 and low <= open_trade["sl"]:
                    exit_price, reason = open_trade["sl"], "stop_loss"
                elif direction == -1 and high >= open_trade["sl"]:
                    exit_price, reason = open_trade["sl"], "stop_loss"
                elif direction == 1 and close >= v:
                    exit_price, reason = close, "vwap_reversion"
                elif direction == -1 and close <= v:
                    exit_price, reason = close, "vwap_reversion"
            elif isinstance(exit_mode, tuple) and exit_mode[0] == "fixed":
                tp_pts = open_trade["sl_pts"] * exit_mode[1]
                hit_sl = (low <= open_trade["sl"]) if direction == 1 else (high >= open_trade["sl"])
                hit_tp = (high >= open_trade["tp"]) if direction == 1 else (low <= open_trade["tp"])
                if hit_sl:
                    exit_price, reason = open_trade["sl"], "stop_loss"
                elif hit_tp:
                    exit_price, reason = open_trade["tp"], "take_profit"
            elif isinstance(exit_mode, tuple) and exit_mode[0] == "trail":
                k = exit_mode[1]
                a = float(row["ATR_14"])
                if direction == 1:
                    if low <= open_trade["trail"]:
                        exit_price, reason = open_trade["trail"], "trailing_stop"
                    open_trade["trail"] = max(open_trade["trail"], high - k * a)
                else:
                    if high >= open_trade["trail"]:
                        exit_price, reason = open_trade["trail"], "trailing_stop"
                    open_trade["trail"] = min(open_trade["trail"], low + k * a)
            if exit_price is not None:
                close_trade(ts, exit_price, reason)

        if open_trade is not None and (mins >= FLATTEN_MIN or is_last):
            close_trade(ts, row["Close"], "session_flatten")

        s = sig.iloc[i]
        if s is Signal.HOLD:
            continue
        if open_trade is not None:
            skips["open_position"] += 1
            continue
        if mins >= ENTRY_CUTOFF_MIN:
            skips["after_cutoff"] += 1
            continue
        if trades_today >= MAX_TRADES_DAY:
            skips["max_trades_day"] += 1
            continue
        sp = float(sl_pts.iloc[i])
        if sp <= 0 or sp * POINT_VALUE > MAX_RISK_USD:
            skips["risk_cap"] += 1
            continue

        entry = float(row["Close"])
        direction = 1 if s is Signal.BUY else -1
        fixed_rr = exit_mode[1] if isinstance(exit_mode, tuple) and exit_mode[0] == "fixed" else 0.0
        trail_k = exit_mode[1] if isinstance(exit_mode, tuple) and exit_mode[0] == "trail" else 0.0
        atr_v = float(row["ATR_14"])
        t = {"direction": direction, "entry": entry, "sl_pts": sp,
             "entry_time": ts}
        if direction == 1:
            t["sl"] = entry - sp
            t["tp"] = entry + sp * fixed_rr
            t["trail"] = float(row["High"]) - trail_k * atr_v
        else:
            t["sl"] = entry + sp
            t["tp"] = entry - sp * fixed_rr
            t["trail"] = float(row["Low"]) + trail_k * atr_v
        open_trade = t
        trades_today += 1

    if open_trade is not None:
        close_trade(df.index[-1], df["Close"].iloc[-1], "end_of_data")

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return tdf, {"total_trades": 0, "win_rate": 0.0, "net_profit": 0.0,
                     "profit_factor": 0.0, "max_drawdown": 0.0}, skips
    return tdf, compute_metrics(tdf), skips


def slice_period(df, start, end):
    return df[(df.index >= start) & (df.index < end)]


def gate(m):
    net = m.get("net_profit", 0.0)
    pf = m.get("profit_factor", 0.0)
    dd = m.get("max_drawdown", 0.0)
    checks = {
        "trades>=20": m.get("total_trades", 0) >= 20,
        "net>0": net > 0,
        "pf>=1.3": pf >= 1.3,
        "maxDD<=50%net": net > 0 and dd <= 0.5 * net,
    }
    return all(checks.values()), checks


# ---------------------------------------------------------------- sweep
def main():
    df = load_fresh()
    df = compute_all(df)
    print(f"data: {len(df)} rows {df.index[0]} -> {df.index[-1]}")

    candidates = {
        "gap_fade": {
            "fn": sig_gap_fade,
            "grid": {"gap_atr": [0.5, 0.75, 1.0], "sl_mult": [1.5, 2.0, 2.5]},
        },
        "ema_pullback": {
            "fn": sig_ema_pullback,
            "grid": {"regime_lb": [20, 40], "tol_atr": [0.5, 1.0], "tp_rr": [1.5, 2.0]},
        },
        "donchian": {
            "fn": sig_donchian,
            "grid": {"days": [10, 15, 20], "sl_mult": [2.0, 2.5], "tp_rr": [2.0, 3.0]},
        },
        "atr_trail": {
            "fn": sig_atr_trail,
            "grid": {"fast": [9, 10], "slow": [21, 50], "trail_atr": [2.5, 3.0, 3.5]},
        },
    }

    fit_df = slice_period(df, FIT_START, FIT_END + pd.Timedelta(days=1))
    oos_df = slice_period(df, OOS_START, OOS_END)
    print(f"fit rows: {len(fit_df)}  oos rows: {len(oos_df)}")

    results = {}
    all_trades = []
    total_combos = 0

    for name, cfg in candidates.items():
        combos = []
        for combo in product(*cfg["grid"].values()):
            params = dict(zip(cfg["grid"].keys(), combo))
            total_combos += 1
            sig, sl_pts, exit_mode = cfg["fn"](df, **params)
            fit_t, fit_m, _ = run_backtest(fit_df, sig.reindex(fit_df.index), sl_pts.reindex(fit_df.index), exit_mode)
            oos_t, oos_m, oos_skips = run_backtest(oos_df, sig.reindex(oos_df.index), sl_pts.reindex(oos_df.index), exit_mode)
            combos.append((params, fit_m, oos_m))

        # selection on FIT only: prefer profitable combos with >=15 trades, rank by PF
        ok = [c for c in combos if c[1].get("total_trades", 0) >= 15 and c[1].get("net_profit", 0) > 0]
        if ok:
            best = max(ok, key=lambda c: (min(c[1].get("profit_factor", 0.0), 10.0), c[1].get("net_profit", 0.0)))
            sel_note = "profitable fit combos ranked by PF"
        else:
            best = max(combos, key=lambda c: c[1].get("net_profit", 0.0))
            sel_note = "no combo met fit gate (>=15 trades & net>0); best-net fallback"

        params, fit_m, oos_m = best
        passed, checks = gate(oos_m)
        oos_t, _, skips = run_backtest(oos_df, *(lambda s, p, e: (s.reindex(oos_df.index), p.reindex(oos_df.index), e))(*cfg["fn"](df, **params)))
        if not oos_t.empty:
            tmp = oos_t.copy()
            tmp["strategy"] = name
            all_trades.append(tmp)

        results[name] = {
            "params": params,
            "selection_note": sel_note,
            "combos": len(combos),
            "fit": {k: fit_m.get(k) for k in ("total_trades", "win_rate", "net_profit", "profit_factor", "max_drawdown")},
            "oos": {k: oos_m.get(k) for k in ("total_trades", "win_rate", "net_profit", "profit_factor", "max_drawdown")},
            "oos_gate_checks": checks,
            "verdict": "PASS" if passed else "FAIL",
            "oos_exit_reasons": oos_t["exit_reason"].value_counts().to_dict() if not oos_t.empty else {},
            "oos_skips": dict(skips),
        }
        print(f"{name}: params={params} fit_net={fit_m.get('net_profit'):.0f} oos_net={oos_m.get('net_profit'):.0f} verdict={results[name]['verdict']}")

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trades_all.empty:
        trades_all.to_csv(OUT_TRADES, index=False)

    payload = {
        "date": "2026-10-14",
        "data": {"file": DATA, "symbol": "MNQ=F", "interval": "1h", "rows": int(len(df)),
                 "window": f"{df.index[0]} -> {df.index[-1]}", "rth_only": True},
        "walk_forward": {"fit": "2024-10-16..2025-10-13", "oos": "2025-10-14..2026-10-13"},
        "engine": {"point_value": POINT_VALUE, "size": SIZE, "eod_flatten": True,
                   "entry_cutoff": "15:30 ET", "max_risk_usd": MAX_RISK_USD,
                   "max_trades_day": MAX_TRADES_DAY},
        "oos_gate": "trades>=20, profit_factor>=1.3, net>0, maxDD<=50% of net",
        "candidates_tested": len(candidates),
        "param_combos_tested": total_combos,
        "results": results,
        "pass_count": sum(1 for r in results.values() if r["verdict"] == "PASS"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    def fmt(m):
        pf = m.get("profit_factor", 0.0)
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        return (f"{m.get('total_trades', 0)} | {m.get('win_rate', 0) * 100:.1f}% | "
                f"${m.get('net_profit', 0):.2f} | {pf_s} | ${m.get('max_drawdown', 0):.2f}")

    lines = [
        "# Viki strategy sweep — 2026-10-14",
        "",
        f"Data: `MNQ=F` 1h RTH, {payload['data']['rows']} bars, {payload['data']['window']}.",
        "Walk-forward: fit **2024-10-16..2025-10-13**, OOS **2025-10-14..2026-10-13**.",
        "Engine: 1 MNQ, $2/pt, EOD flatten (no overnight), entry cutoff 15:30 ET, "
        f"risk cap ${MAX_RISK_USD:.0f}/trade, max {MAX_TRADES_DAY} entries/day.",
        f"OOS gate: `{payload['oos_gate']}`.",
        "",
        "| Strategy | Params | Fit trades | Fit WR | Fit net | Fit PF | Fit maxDD | OOS trades | OOS WR | OOS net | OOS PF | OOS maxDD | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['params']} | {fmt(r['fit'])} | {fmt(r['oos'])} | **{r['verdict']}** |")
    lines += ["", "## OOS gate detail", ""]
    for name, r in results.items():
        lines.append(f"- **{name}** ({r['verdict']}): {r['oos_gate_checks']} · exits={r['oos_exit_reasons']}")
    passes = [n for n, r in results.items() if r["verdict"] == "PASS"]
    lines += [
        "",
        f"## Verdict: {len(passes)}/{len(results)} candidates pass the OOS gate"
        + (f" — {', '.join(passes)}" if passes else " — none"),
        "",
        "Notes:",
        "- Params selected on the FIT window only (profitable combos with >=15 fit trades, ranked by PF); OOS never used for selection.",
        "- 1h Yahoo futures bars are stamped 10:00–16:00 ET; 'daily open' = first bar Open vs previous day's last Close.",
        "- Prior history: VWAP Momentum + Session Breakout failed OOS (-$770, 5m Aug–Oct 2026); regime-dependent edge only in UP regimes.",
        "",
        f"JSON: `{OUT_JSON}` · trades: `{OUT_TRADES}`",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({k: {"verdict": v["verdict"], "oos": v["oos"], "params": v["params"]}
                      for k, v in results.items()}, indent=2, default=str))
    print(f"report: {OUT_MD}")


if __name__ == "__main__":
    main()
