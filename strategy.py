"""
strategy.py — 量化交易策略

改进说明：
- 基于R86配置（val_score=2.1159），将固定周期出场改为ATR动态止损
- 做多出场：从固定28周期低点 → 2.0x ATR止损
- 做空出场：从固定36周期高点 → 2.0x ATR止盈（反向追踪）
- ATR能自适应波动率变化，在高波动期扩大止损空间，低波动期收紧

预期：减少因固定周期导致的假止损，提高Sharpe Ratio
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 共用指标 ──
    ema50 = close.ewm(span=50, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    atr14 = tr.rolling(14).mean()
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.5 * atr
    keltner_lower = ema50 - 2.5 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── EMA150 斜率熊市检测 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    # ── 做多系统（ATR动态止损） ──
    entry_high = high.rolling(58).max()
    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_entry_price = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_entry_price = close.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            # ATR动态止损：多头止损位 = 入场价 - 2.0 * ATR
            stop_loss = long_entry_price - 2.0 * atr.iloc[i]
            if close.iloc[i] < stop_loss:
                in_long = False
                long_entry_price = 0.0
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（ATR动态止盈） ──
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_price = 0.0

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.50
        else:
            # ATR动态止盈：空头止盈位 = 入场价 + 2.0 * ATR（反向追踪）
            take_profit = short_entry_price + 2.0 * atr.iloc[i]
            if close.iloc[i] > take_profit:
                in_short = False
                short_entry_price = 0.0
            else:
                short_signal.iloc[i] = -0.50

    return (long_signal + short_signal).clip(-1.0, 1.0)