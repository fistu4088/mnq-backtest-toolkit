"""Teljesítménymutatók számítása."""

import numpy as np
import pandas as pd


def compute_metrics(trades_df):
    """Összes metrika kiszámítása a trade listából.

    Args:
        trades_df: DataFrame oszlopokkal: entry_time, exit_time, direction,
                   entry_price, exit_price, pnl, r_multiple
    Returns:
        dict: metrikák
    """
    if trades_df.empty:
        return {"error": "No trades"}

    metrics = {}
    pnl_series = trades_df["pnl"]
    winning_trades = trades_df[trades_df["pnl"] > 0]
    losing_trades = trades_df[trades_df["pnl"] <= 0]

    # Alap metrikák
    metrics["total_trades"] = len(trades_df)
    metrics["winning_trades"] = len(winning_trades)
    metrics["losing_trades"] = len(losing_trades)
    metrics["win_rate"] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0

    # PnL
    metrics["gross_profit"] = float(winning_trades["pnl"].sum()) if len(winning_trades) > 0 else 0.0
    metrics["gross_loss"] = float(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0.0
    metrics["net_profit"] = float(pnl_series.sum())

    # Profit Factor
    gross_loss_abs = abs(metrics["gross_loss"])
    metrics["profit_factor"] = float(metrics["gross_profit"] / gross_loss_abs) if gross_loss_abs > 0 else float("inf")

    # Average trade
    metrics["avg_win"] = float(winning_trades["pnl"].mean()) if len(winning_trades) > 0 else 0.0
    metrics["avg_loss"] = float(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 0.0
    metrics["avg_trade"] = float(pnl_series.mean())

    # R múltipló
    if "r_multiple" in trades_df.columns:
        metrics["avg_r_multiple"] = float(trades_df["r_multiple"].mean())

    # Max drawdown (equity curve alapján)
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    metrics["max_drawdown"] = float(abs(drawdown.min()))
    peak = max(cumulative.max(), 1)
    metrics["max_drawdown_pct"] = float(abs(drawdown.min() / peak)) if peak > 0 else 0.0

    # Sharpe Ratio (annualizált, risk-free = 0)
    if len(trades_df) > 1 and float(pnl_series.std()) > 0:
        # Tőzsdei napok száma évente: ~252
        # Mivel trade-enként számolunk, a Sharpe trade-alapú
        metrics["sharpe_ratio"] = float((pnl_series.mean() / pnl_series.std()) * np.sqrt(252))
    else:
        metrics["sharpe_ratio"] = 0.0

    # Sortino (downside deviation)
    downside = pnl_series[pnl_series < 0]
    if len(downside) > 0 and float(downside.std()) > 0:
        metrics["sortino_ratio"] = float((pnl_series.mean() / downside.std()) * np.sqrt(252))
    else:
        metrics["sortino_ratio"] = 0.0

    # Calmar Ratio
    if metrics["max_drawdown"] > 0:
        metrics["calmar_ratio"] = float(metrics["net_profit"] / metrics["max_drawdown"])
    else:
        metrics["calmar_ratio"] = float("inf")

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_streak = 0

    for pnl in pnl_series:
        if pnl > 0:
            current_streak = current_streak + 1 if current_streak > 0 else 1
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            current_streak = current_streak - 1 if current_streak < 0 else -1
            max_consec_losses = max(max_consec_losses, abs(current_streak))

    metrics["max_consecutive_wins"] = max_consec_wins
    metrics["max_consecutive_losses"] = max_consec_losses

    # Nap-koncentráció / nap-robusztusság (K1 előszűrő, Betti nyomán):
    # mennyire áll a nettó PnL néhány napon, és hány nap kihagyása
    # fordítja negatívba az összesített eredményt.
    if "entry_time" in trades_df.columns:
        try:
            days = pd.to_datetime(trades_df["entry_time"]).dt.date
            day_pnl = trades_df.groupby(days)["pnl"].sum()
            net = float(day_pnl.sum())

            metrics["trading_days"] = int(len(day_pnl))
            metrics["positive_days"] = int((day_pnl > 0).sum())
            metrics["negative_days"] = int((day_pnl <= 0).sum())

            if net > 0:
                sorted_days = day_pnl.sort_values(ascending=False)
                metrics["top1_day_share"] = float(sorted_days.iloc[0] / net)
                top3 = float(sorted_days.head(3).sum() / net)
                metrics["top3_day_share"] = top3

                # leave-one-day-out: hány nap kihagyása teszi negatívba a nettót
                neg_count = int((net - day_pnl < 0).sum())
                metrics["leave_one_day_neg"] = neg_count
                metrics["leave_one_day_neg_pct"] = neg_count / len(day_pnl) if len(day_pnl) > 0 else 0.0
        except Exception:
            pass

    return metrics


def print_metrics(metrics):
    """Metrikák kiírása olvasható formában."""
    if "error" in metrics:
        print(f"⚠️  {metrics['error']}")
        return

    print(f"{'='*55}")
    print(f"  📊  BACKTEST EREDMÉNYEK")
    print(f"{'='*55}")
    print(f"  Összes trade:      {metrics['total_trades']}")
    print(f"  Nyerő / Vesztes:   {metrics['winning_trades']} / {metrics['losing_trades']}")
    print(f"  Win Rate:          {metrics['win_rate']*100:.1f}%")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"")
    print(f"  Nettó profit:      ${metrics['net_profit']:.2f}")
    print(f"  Brutto nyereség:   ${metrics['gross_profit']:.2f}")
    print(f"  Brutto veszteség:  ${metrics['gross_loss']:.2f}")
    print(f"")
    print(f"  Átlag nyerő:       ${metrics['avg_win']:.2f}")
    print(f"  Átlag vesztes:     ${metrics['avg_loss']:.2f}")
    print(f"  Átlag trade:       ${metrics['avg_trade']:.2f}")
    print(f"  Átlag R-multiple:  {metrics.get('avg_r_multiple', 0):.2f}R")
    print(f"")
    print(f"  Max Drawdown:      ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']*100:.1f}%)")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}")
    print(f"")
    print(f"  Max nyerő sorozat: {metrics['max_consecutive_wins']}")
    print(f"  Max vesztő sorozat:{metrics['max_consecutive_losses']}")
    print(f"{'='*55}")
