"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进：R20 -> R21
- 改动：去除做空系统的 ADX>25 限制，改为固定ATR移动止损保护
- 理由：R17做空40%仓位val=2.39但训练集亏损，说明严格过滤导致过拟合
       去掉ADX限制增加做空次数，ATR止损保护防止大幅回撤
- 预期：增加做空样本量，提高泛化能力
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
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 做多系统（始终运行，25% 仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_entry_price = 0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_entry_price = close.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            # ATR移动止损：入场价 - 2*ATR 且低于前一根K线低点
            atr_stop = long_entry_price - 2 * atr.iloc[i]
            trail_stop = min(atr_stop, low.iloc[i-1])
            if close.iloc[i] < exit_low.iloc[i - 1] or close.iloc[i] < trail_stop:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（仅在 EMA斜率熊市 中激活，40% 仓位） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_price = 0

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05

        if not in_short:
            if (bear_confirmed
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            # ATR移动止损保护：入场价 + 2*ATR 且高于前一根K线高点
            atr_stop = short_entry_price + 2 * atr.iloc[i]
            trail_stop = max(atr_stop, high.iloc[i-1])
            if close.iloc[i] > exit_high.iloc[i - 1] or close.iloc[i] > trail_stop:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)