#!/usr/bin/env python3
"""Paper v1 combined book: A (VWAP Momentum) + B (Session Breakout).

Implements the 2026-09-30 strategy v1 + Lucia envelope on 1 MNQ:
- A priority over B, no stacking
- A: TP1 = full close (1.5R); B: full close at 2.5R
- No entries after 15:45 ET; flatten 15:55 ET
- Hard skip if SL_pts * $2 > $120
- Soft daily -$250 (no new entries); hard daily -$400 flatten+lock
- 3 consecutive losses halt; max 4 trades / RTH
- session_ema parked
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data import load_csv
from backtest.indicators import compute_all
from backtest.metrics import compute_metrics
from backtest.strategies import SessionBreakoutStrategy, Signal, VWAPMomentumStrategy

POINT_VALUE = 2.0
SIZE = 1
SOFT_DAILY = -250.0
HARD_DAILY = -400.0
MAX_TRADES = 4
MAX_CONSEC_LOSS = 3
MAX_RISK_USD = 120.0
ENTRY_CUTOFF_MIN = 15 * 60 + 45  # 15:45 ET (override via --cutoff)
FLATTEN_MIN = 15 * 60 + 55       # 15:55 ET
SKIP_AFTER_KEY = "after_1545"

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_TRADES = os.path.join(ROOT, "data", "trades_paper_v1.csv")
OUT_METRICS = os.path.join(ROOT, "data", "metrics_paper_v1.json")
OUT_REPORT = os.path.join(ROOT, "reports", "2026-09-30_rami_paper_v1.md")


def minutes_of(ts) -> int:
    return ts.hour * 60 + ts.minute


def usd_pt_ok(row) -> bool:
    pts = (float(row["exit_price"]) - float(row["entry_price"])) * (
        1.0 if row["direction"] == "BUY" else -1.0
    )
    if abs(pts) <= 1e-9:
        return abs(float(row["pnl"])) <= 1e-9
    return abs(float(row["pnl"]) / pts - POINT_VALUE) < 1e-6


def run():
    data_path = os.path.join(ROOT, "data", "mnq_historical.csv")
    df = load_csv(data_path)
    if df.empty:
        raise SystemExit(f"empty data: {data_path}")
    df = compute_all(df)

    a = VWAPMomentumStrategy()
    b = SessionBreakoutStrategy()
    sig_a = a.generate_signals(df)
    sig_b = b.generate_signals(df)

    trades = []
    skips = Counter()
    open_trade = None
    last_date = None
    daily_pnl = 0.0
    trades_today = 0
    consec_loss = 0
    day_locked = False

    def close_trade(ts, price, reason):
        nonlocal open_trade, daily_pnl, consec_loss, day_locked
        t = open_trade
        pts = (float(price) - t["entry"]) * t["direction"]
        pnl = round(pts * POINT_VALUE * SIZE, 2)
        risk = t["sl_pts"]
        r_mult = round(pts / risk, 2) if risk > 0 else 0.0
        rec = {
            "entry_time": t["entry_time"],
            "exit_time": ts,
            "strategy": t["strategy"],
            "direction": "BUY" if t["direction"] == 1 else "SELL",
            "entry_price": t["entry"],
            "exit_price": float(price),
            "sl": t["sl"],
            "tp": t["tp"],
            "sl_pts": t["sl_pts"],
            "pnl": pnl,
            "r_multiple": r_mult,
            "exit_reason": reason,
            "size": SIZE,
        }
        trades.append(rec)
        daily_pnl += pnl
        if pnl < 0:
            consec_loss += 1
        else:
            consec_loss = 0
        if daily_pnl <= HARD_DAILY:
            day_locked = True
        open_trade = None
        return rec

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
            daily_pnl = 0.0
            trades_today = 0
            consec_loss = 0
            day_locked = False
            last_date = d

        # Manage open
        if open_trade is not None:
            high, low, close = float(row["High"]), float(row["Low"]), float(row["Close"])
            sl, tp, direction = open_trade["sl"], open_trade["tp"], open_trade["direction"]
            hit_sl = (low <= sl) if direction == 1 else (high >= sl)
            hit_tp = (high >= tp) if direction == 1 else (low <= tp)
            if hit_sl:
                close_trade(ts, sl, "stop_loss")
            elif hit_tp:
                close_trade(ts, tp, "take_profit")

        # Flatten 15:55 (last RTH bar) — before considering new entries
        if open_trade is not None and (mins >= FLATTEN_MIN or is_last):
            close_trade(ts, row["Close"], "session_flatten")

        sa, sb = sig_a.iloc[i], sig_b.iloc[i]
        want = None
        if sa is not Signal.HOLD:
            want = ("vwap_momentum", sa.value)
        elif sb is not Signal.HOLD:
            want = ("session_breakout", sb.value)

        if want is None:
            continue
        if open_trade is not None:
            skips["open_position"] += 1
            continue
        if mins >= ENTRY_CUTOFF_MIN:
            skips[SKIP_AFTER_KEY] += 1
            continue
        if day_locked:
            skips["day_locked"] += 1
            continue
        if daily_pnl <= SOFT_DAILY:
            skips["soft_daily"] += 1
            continue
        if consec_loss >= MAX_CONSEC_LOSS:
            skips["consec_loss"] += 1
            continue
        if trades_today >= MAX_TRADES:
            skips["max_trades"] += 1
            continue

        atr = row.get("ATR_14", 10)
        if pd.isna(atr) or atr <= 0:
            skips["bad_atr"] += 1
            continue

        name, direction = want
        sl_mult = 2.5 if name == "vwap_momentum" else 2.0
        tp_rr = 1.5 if name == "vwap_momentum" else 2.5  # A: TP1 full; B: 2.5R full
        sl_pts = float(atr) * sl_mult
        risk_usd = sl_pts * POINT_VALUE * SIZE
        if risk_usd > MAX_RISK_USD:
            skips["sl_skip_120"] += 1
            continue

        entry = float(row["Close"])
        if direction == 1:
            sl, tp = entry - sl_pts, entry + sl_pts * tp_rr
        else:
            sl, tp = entry + sl_pts, entry - sl_pts * tp_rr

        open_trade = {
            "strategy": name,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_pts": sl_pts,
            "entry_time": ts,
        }
        trades_today += 1

    if open_trade is not None:
        close_trade(df.index[-1], df["Close"].iloc[-1], "end_of_data")

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise SystemExit("paper v1 produced 0 trades")

    overnight = int((trades_df["exit_reason"] == "end_of_day").sum())
    eod = int((trades_df["exit_reason"] == "end_of_data").sum())
    usd_ok = bool(trades_df.apply(usd_pt_ok, axis=1).all())

    metrics = compute_metrics(trades_df)
    by_strat = {}
    for name, g in trades_df.groupby("strategy"):
        by_strat[name] = {
            "n": int(len(g)),
            "win_rate": float((g["pnl"] > 0).mean()),
            "net_pnl": float(g["pnl"].sum()),
            "exits": g["exit_reason"].value_counts().to_dict(),
        }

    payload = {
        "window": f"{df.index[0]} — {df.index[-1]}",
        "entry_cutoff_min": ENTRY_CUTOFF_MIN,
        "rows": int(len(df)),
        "trading_days": int(df.index.normalize().nunique()),
        "k0": {
            "point_value": POINT_VALUE,
            "usd_pt_ok": usd_ok,
            "overnight_end_of_day": overnight,
            "end_of_data": eod,
        },
        "skips": dict(skips),
        "metrics": {
            k: (None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, str)) or v is None
        },
        "by_strategy": by_strat,
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
    }

    trades_df.to_csv(OUT_TRADES, index=False)
    with open(OUT_METRICS, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    m = payload["metrics"]
    lines = [
        "# Paper v1 combined book — 2026-09-30",
        "",
        f"Ablak: `{payload['window']}` · {payload['rows']} bár · {payload['trading_days']} RTH nap.",
        "Szabály: 1 MNQ, A>B, A TP1=full 1.5R, B full 2.5R, flatten 15:55, belépés tilos 15:45 után,",
        "hard skip SL>$120, soft −$250 / hard −$400, 3 vesztes halt, max 4 trade/nap. session_ema parkolva.",
        "",
        "## K0",
        f"- usd/pt == 2.00: **{'PASS' if usd_ok else 'FAIL'}**",
        f"- overnight (`end_of_day`): **{overnight}** (kell 0)",
        f"- `end_of_data`: {eod}",
        "",
        "## Eredmény",
        f"- trades: **{m.get('total_trades')}** · WR **{m.get('win_rate', 0)*100:.1f}%** · net **${m.get('net_profit', 0):.2f}**",
        f"- PF {m.get('profit_factor')} · Sharpe {m.get('sharpe_ratio')} · maxDD **${m.get('max_drawdown', 0):.2f}**",
        f"- avg R {m.get('avg_r_multiple')} · leave_one_day_neg {m.get('leave_one_day_neg')} / {m.get('trading_days')}",
        "",
        "## Stratégiánként",
    ]
    for name, s in by_strat.items():
        lines.append(
            f"- **{name}**: n={s['n']} WR={s['win_rate']*100:.1f}% net=${s['net_pnl']:.2f} exits={s['exits']}"
        )
    lines += [
        "",
        "## Kilépések / skippek",
        f"- exits: {payload['exit_reasons']}",
        f"- skipped signals: {dict(skips)}",
        "",
        f"CSV: `{OUT_TRADES}` · JSON: `{OUT_METRICS}`",
        "",
        "K1 ettől még piros — ez paper replay a meglévő 60 napos rally-ablakon, nem új validáció.",
    ]
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, default=str))
    print(f"\nreport: {OUT_REPORT}")
    print(f"trades: {OUT_TRADES}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cutoff", default="15:45", help="No new entries at/after this ET time (HH:MM).")
    args = p.parse_args()
    hh, mm = args.cutoff.split(":")
    ENTRY_CUTOFF_MIN = int(hh) * 60 + int(mm)
    SKIP_AFTER_KEY = f"after_{hh}{mm}"
    if args.cutoff != "15:45":
        tag = f"cutoff{hh}{mm}"
        OUT_TRADES = os.path.join(ROOT, "data", f"trades_paper_v1_{tag}.csv")
        OUT_METRICS = os.path.join(ROOT, "data", f"metrics_paper_v1_{tag}.json")
        OUT_REPORT = os.path.join(ROOT, "reports", f"2026-09-30_rami_paper_v1_{tag}.md")
    run()
