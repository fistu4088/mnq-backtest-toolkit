# MNQ Backtest Toolkit

Honest, no-nonsense backtesting framework for Micro Nasdaq (MNQ) day-trading strategies, built on Python + pandas + yfinance (all free).

## What it does
- Pulls MNQ/NQ futures data (1h 2y, 5m 60d) from Yahoo Finance with caching
- Hand-written indicators (EMA, SMA, ATR, RSI, VWAP) — no fragile TA libs
- Event-driven backtest engine: intraday only, EOD flatten, ATR stops, daily loss caps
- 6 ready strategies (VWAP momentum, session breakout, gap fade, Donchian, EMA pullback, ATR trail)
- Walk-forward validation harness (fit period / out-of-sample period split)
- Metrics: win rate, profit factor, max drawdown, expectancy, per-regime breakdown
- Regime filter (EMA100 on 1h) with hysteresis — the single biggest edge-multiplier found

## What it is NOT
- NOT a signal service, NOT a guarantee of profits. The author ran 6 strategy families through out-of-sample validation and most failed. That honesty is the point: this toolkit tells you FAST whether your idea survives before you risk money.
- Educational/research software. Not financial advice.

## Requirements
Python 3.10+, pandas, numpy, yfinance. No paid dependencies.

## Contents
- backtest/ — data, indicators, strategies, engine, metrics, risk manager
- runners: run_paper_v1.py (paper book), run_viki_sweep.py (walk-forward sweep)
- config/strategies.yaml — strategy parameters
- README.md — quickstart

## Quickstart
```
pip install pandas numpy yfinance
python run_viki_sweep.py   # walk-forward validation of 4 strategies
python run_paper_v1.py     # daily paper book
```
