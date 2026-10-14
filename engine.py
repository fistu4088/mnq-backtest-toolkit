"""Backtest engine — partial exits, trailing, breakeven, session-aware."""

import pandas as pd
import numpy as np
from .risk_manager import TopstepRiskManager
from .metrics import compute_metrics
from .logger import setup_logger
from .strategies import Signal

# MNQ (Micro E-mini Nasdaq-100): 1 index pont = $2.00 (4 tick/pont × $0.50).
# K0 fix: a korábbi 20.0 az NQ (full-size) szorzó volt — 10× túljelentett P&L.
MNQ_POINT_VALUE = 2.0
MNQ_TICK_VALUE = 0.50


class Trade:
    """Egyetlen trade összes eseménnyel (partial + maradék)."""

    def __init__(self, entry_time, direction, entry_price, sl, tp1, tp2,
                 partial_ratio=0.5, trailing_activate=0, trailing_distance=0,
                 breakeven_after_r=0, size=1, point_value=MNQ_POINT_VALUE):
        self.entry_time = entry_time
        self.direction = direction  # 1 = BUY, -1 = SELL
        self.entry_price = entry_price
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.partial_ratio = partial_ratio
        self.trailing_activate = trailing_activate
        self.trailing_distance = trailing_distance
        self.breakeven_after_r = breakeven_after_r
        self.size = size
        self.point_value = point_value
        self.remaining_size = size
        self.partial_filled = False

        # Exit state
        self.exit_parts = []  # List of (time, price, pnl, r, reason, size)
        self.exit_time = None

        # Trailing state
        self.trailing_active = False
        self.trailing_high = entry_price if direction == 1 else entry_price
        self.breakeven_active = False

        # R multiple for breakeven calculation
        self._risk_points = abs(entry_price - sl)

    def get_highest_price(self):
        return self.trailing_high if self.direction == 1 else self.trailing_low

    def update_trailing(self, high, low):
        """Frissíti a trailing stop szintet a gyertya high/low alapján."""
        if self.direction == 1:  # BUY
            if high > self.trailing_high:
                self.trailing_high = high
            if self.trailing_active:
                new_sl = self.trailing_high - self.trailing_distance
                if new_sl > self.sl:
                    self.sl = new_sl

        else:  # SELL
            if low < self.trailing_high:
                self.trailing_high = low
            if self.trailing_active:
                new_sl = self.trailing_high + self.trailing_distance
                if new_sl < self.sl:
                    self.sl = new_sl

    def check_breakeven(self, high, low):
        """Breakeven aktiválása, ha elég messze van az árfolyam."""
        if self.breakeven_active or self.breakeven_after_r <= 0:
            return

        if self.direction == 1:  # BUY
            move = high - self.entry_price
        else:  # SELL
            move = self.entry_price - low

        r_multiple = move / self._risk_points if self._risk_points > 0 else 0
        if r_multiple >= self.breakeven_after_r:
            self.sl = self.entry_price  # Breakeven
            self.breakeven_active = True

    def check_trailing_activate(self, high, low):
        """Trailing aktiválása, ha elég messze van az árfolyam."""
        if self.trailing_active or self.trailing_activate <= 0:
            return

        if self.direction == 1:  # BUY
            move = high - self.entry_price
        else:  # SELL
            move = self.entry_price - low

        if move >= self.trailing_activate:
            self.trailing_active = True
            # Kezdeti trailing SL beállítása
            if self.direction == 1:
                self.sl = max(self.sl, self.trailing_high - self.trailing_distance)
            else:
                self.sl = min(self.sl, self.trailing_high + self.trailing_distance)

    def close_partial(self, exit_time, exit_price, reason="partial_tp"):
        """Első partial exit (pl. 50% TP1-nél)."""
        close_size = max(1, round(self.size * self.partial_ratio))
        close_size = min(close_size, self.remaining_size)

        if close_size <= 0:
            return 0.0

        self.partial_filled = True

        entry_p = float(self.entry_price)
        exit_p = float(exit_price)
        points = (exit_p - entry_p) * float(self.direction)
        risk_points = self._risk_points
        pnl = round(points * self.point_value * float(close_size), 2)
        r_mult = round(points / risk_points, 2) if risk_points > 0 else 0.0

        self.exit_parts.append({
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": pnl,
            "r_multiple": r_mult,
            "exit_reason": reason,
            "size": close_size,
        })

        self.remaining_size -= close_size

        # Ha elfogyott a méret, teljesen zárt
        if self.remaining_size <= 0:
            self._finalize(exit_time)

        return pnl

    def close_remaining(self, exit_time, exit_price, reason="exit"):
        """Maradék pozíció zárása."""
        if self.remaining_size <= 0:
            return 0.0

        entry_p = float(self.entry_price)
        exit_p = float(exit_price)
        points = (exit_p - entry_p) * float(self.direction)
        risk_points = self._risk_points
        pnl = round(points * self.point_value * float(self.remaining_size), 2)
        r_mult = round(points / risk_points, 2) if risk_points > 0 else 0.0

        self.exit_parts.append({
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": pnl,
            "r_multiple": r_mult,
            "exit_reason": reason,
            "size": self.remaining_size,
        })

        self.remaining_size = 0
        self._finalize(exit_time)
        return pnl

    def close_full(self, exit_time, exit_price, reason="exit"):
        """Teljes pozíció zárása egy ütemben."""
        # Ha volt már partial, csak a maradékot zárjuk
        if self.partial_filled and self.remaining_size > 0:
            return self.close_remaining(exit_time, exit_price, reason)
        # Különben teljeset zárjuk
        return self.close_remaining(exit_time, exit_price, reason)

    def close(self, exit_time, exit_price, reason="manual"):
        """Kompatibilitási wrapper — teljes zárás."""
        return self.close_full(exit_time, exit_price, reason)

    def _finalize(self, exit_time):
        """Trade véglegesítése a legutolsó exit időponttal."""
        if not self.exit_parts:
            self.exit_time = exit_time
            self.pnl = 0.0
            self.r_multiple = 0.0
            self.exit_reason = "empty"
            return

        self.exit_time = exit_time
        self.pnl = round(sum(p["pnl"] for p in self.exit_parts), 2)
        # Súlyozott átlag R-multiple
        total_size = sum(p["size"] for p in self.exit_parts)
        if total_size > 0:
            weighted_r = sum(p["r_multiple"] * p["size"] for p in self.exit_parts)
            self.r_multiple = round(weighted_r / total_size, 2)
        else:
            self.r_multiple = 0.0
        self.exit_reason = self.exit_parts[-1]["exit_reason"]

    def to_dict(self):
        # Utolsó exit idő és ár
        last_exit = self.exit_parts[-1] if self.exit_parts else {
            "exit_time": self.exit_time,
            "exit_price": self.entry_price,
            "exit_reason": "open",
        }
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time or self.entry_time,
            "direction": "BUY" if self.direction == 1 else "SELL",
            "entry_price": self.entry_price,
            "exit_price": last_exit["exit_price"],
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "pnl": self.pnl,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason,
            "size": self.size,
            "partial_filled": self.partial_filled,
            "num_exits": len(self.exit_parts),
        }


class BacktestEngine:
    """Backtest engine — partial exits, trailing, breakeven, session-aware."""

    def __init__(self, df, strategy, risk_mgr=None, logger=None):
        self.df = df
        self.strategy = strategy
        self.risk_mgr = risk_mgr or TopstepRiskManager()
        self.logger = logger or setup_logger()
        self.trades = []
        self.current_trade = None

        # K0 fix: point_value a konfigból (mnq_specs.point_value = 2.0), nem hardcode.
        mnq_specs = getattr(self.risk_mgr, "rules", {}).get("mnq_specs", {}) or {}
        self.point_value = float(mnq_specs.get("point_value", MNQ_POINT_VALUE))

    def run(self):
        """Backtest futtatása."""
        signals = self.strategy.generate_signals(self.df)
        last_trade_date = None

        for i in range(len(self.df)):
            timestamp = self.df.index[i]
            row = self.df.iloc[i]
            signal = signals.iloc[i]
            current_date = timestamp.date()

            # Session beállítása a risk managerben
            session = self.strategy.get_session(timestamp) if hasattr(self.strategy, 'get_session') else "open"
            self.risk_mgr.set_session(session)

            # Napi reset
            if current_date != last_trade_date:
                if self.current_trade:
                    self._close_trade(timestamp, row["Close"], "end_of_day")
                self.risk_mgr.reset_daily()
                last_trade_date = current_date

            # Risk check
            can_trade, reason = self.risk_mgr.can_trade()
            if not can_trade:
                if self.current_trade:
                    self._close_trade(timestamp, row["Close"], f"risk_block: {reason}")
                continue

            # Nyitott trade kezelése
            if self.current_trade:
                self._manage_open_trade(timestamp, row)
                if self.current_trade is None:
                    continue

            # Új trade nyitása, ha van jel és nincs nyitott pozíció
            if signal is not Signal.HOLD and self.current_trade is None:
                direction = signal.value
                entry_price = row["Close"]
                atr = row.get("ATR_14", 10)
                if pd.isna(atr) or atr <= 0:
                    atr = 10

                # SL/TP: stratégia saját beállításai elsőbbséget élveznek
                # Session paraméterek csak trail/t_min_trade-hez
                if hasattr(self.strategy, 'params'):
                    strat_sl = self.strategy.params.get("sl_atr", None)
                    strat_tp = self.strategy.params.get("tp_rr", None)
                else:
                    strat_sl = None
                    strat_tp = None

                if hasattr(self.strategy, 'get_session_params'):
                    sp = self.strategy.get_session_params(timestamp, atr)
                    # Use strategy's own SL/TP when defined, fallback to session params
                    sl_mult = strat_sl if strat_sl is not None else sp["sl_atr"]
                    tp_rr = strat_tp if strat_tp is not None else sp["tp_rr"]
                    trail_atr = sp.get("trail_atr", 0)
                else:
                    sl_mult = strat_sl if strat_sl is not None else 1.5
                    tp_rr = strat_tp if strat_tp is not None else 2.5
                    trail_atr = 0

                sl_distance = atr * sl_mult

                if direction == 1:  # BUY
                    sl = entry_price - sl_distance
                    tp1 = entry_price + sl_distance * 1.5    # 1.5R partial
                    tp2 = entry_price + sl_distance * tp_rr  # 2.5R full
                else:  # SELL
                    sl = entry_price + sl_distance
                    tp1 = entry_price - sl_distance * 1.5
                    tp2 = entry_price - sl_distance * tp_rr

                # Kockázat ellenőrzése
                risk_points = abs(entry_price - sl)
                risk_amount = risk_points * self.point_value
                if risk_amount > self.risk_mgr.max_risk_per_trade * 2:
                    continue

                size = self.risk_mgr.get_position_size()
                breakeven_r = 1.0  # 1R után breakeven
                trailing_dist = atr * trail_atr if trail_atr > 0 else 0
                trailing_activate = sl_distance * 2 if trailing_dist > 0 else 0

                self.current_trade = Trade(
                    timestamp, direction, entry_price, sl, tp1, tp2,
                    partial_ratio=0.5,
                    trailing_activate=trailing_activate,
                    trailing_distance=trailing_dist,
                    breakeven_after_r=breakeven_r,
                    size=size,
                    point_value=self.point_value,
                )
                self.risk_mgr.trades_today += 1

            # EOD flatten — daytrade: napi utolsó gyertya előtt zárás (16:00 ET előtt,
            # nincs overnight kitettség). A data már RTH-ra szűrt (09:30–16:00), így a
            # napi utolsó gyertya az utolsó 5m bar (15:55 ET).
            is_last_bar_of_day = (i == len(self.df) - 1) or (self.df.index[i + 1].date() != current_date)
            if is_last_bar_of_day and self.current_trade:
                self._close_trade(timestamp, row["Close"], "eod_flatten")

        # Utolsó trade zárása
        if self.current_trade:
            self._close_trade(self.df.index[-1], self.df["Close"].iloc[-1], "end_of_data")

        trades_df = self._trades_to_df()
        metrics = compute_metrics(trades_df)
        return metrics, trades_df

    def _manage_open_trade(self, timestamp, row):
        """Nyitott trade kezelése: trailing, breakeven, SL/TP check, partial."""
        trade = self.current_trade
        high = row["High"]
        low = row["Low"]
        close = row["Close"]

        # 1. Trailing state frissítése
        trade.update_trailing(high, low)
        trade.check_trailing_activate(high, low)
        trade.check_breakeven(high, low)

        # 2. SL check (AKTUÁLIS SL — lehet hogy trailing módosította)
        if trade.direction == 1:  # BUY
            if low <= trade.sl:
                self._close_trade(timestamp, trade.sl, "stop_loss")
                return
        else:  # SELL
            if high >= trade.sl:
                self._close_trade(timestamp, trade.sl, "stop_loss")
                return

        # 3. Ha nincs partial még, TP1 check
        if not trade.partial_filled:
            if trade.direction == 1:
                if high >= trade.tp1:
                    # Partial exit: close 50% at TP1
                    pnl = trade.close_partial(timestamp, trade.tp1, "partial_tp1")
                    self.logger.info(
                        f"{timestamp.strftime('%m/%d %H:%M')} | "
                        f"PARTIAL {'BUY ' if trade.direction == 1 else 'SELL'} | "
                        f"Entry: ${trade.entry_price:.1f} → TP1: ${trade.tp1:.1f} | "
                        f"PnL: ${pnl:.1f}"
                    )
                    # Ha elfogyott, zárva
                    if trade.remaining_size <= 0:
                        self._finalize_trade(timestamp)
                        return
            else:
                if low <= trade.tp1:
                    pnl = trade.close_partial(timestamp, trade.tp1, "partial_tp1")
                    self.logger.info(
                        f"{timestamp.strftime('%m/%d %H:%M')} | "
                        f"PARTIAL {'SELL'} | "
                        f"Entry: ${trade.entry_price:.1f} → TP1: ${trade.tp1:.1f} | "
                        f"PnL: ${pnl:.1f}"
                    )
                    if trade.remaining_size <= 0:
                        self._finalize_trade(timestamp)
                        return

        # 4. Ha volt partial, TP2 check (maradék pozíció)
        if trade.partial_filled and trade.remaining_size > 0:
            if trade.direction == 1:
                if high >= trade.tp2:
                    self._close_trade(timestamp, trade.tp2, "take_profit_tp2")
                    return
                elif close >= trade.tp2:
                    # Zárás close-on, ha TP2-t érintette
                    pass
            else:
                if low <= trade.tp2:
                    self._close_trade(timestamp, trade.tp2, "take_profit_tp2")
                    return

    def _close_trade(self, timestamp, price, reason):
        """Trade zárása és naplózása."""
        if self.current_trade is None:
            return 0.0

        # Ha volt partial, csak maradékot zárunk
        pnl = self.current_trade.close_full(timestamp, price, reason)
        self.risk_mgr.update_balance(pnl)

        self.logger.info(
            f"{timestamp.strftime('%m/%d %H:%M')} | "
            f"{'BUY ' if self.current_trade.direction == 1 else 'SELL'} | "
            f"Entry: ${self.current_trade.entry_price:.1f} → "
            f"Exit: ${price:.1f} | "
            f"PnL: ${pnl:.1f} ({self.current_trade.exit_reason})"
        )

        self._finalize_trade(timestamp)
        return pnl

    def _finalize_trade(self, timestamp):
        """Trade véglegesítése és listába tétele."""
        if self.current_trade:
            self.current_trade._finalize(timestamp)
            self.trades.append(self.current_trade)
            self.current_trade = None

    def _trades_to_df(self):
        """Trade lista → DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
