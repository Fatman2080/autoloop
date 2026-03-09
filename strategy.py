"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动：ATR动态出场替代固定Donchian出场
- 做多出场：high - 2*ATR（更紧密，响应更快）
- 做空出场：low + 2*ATR
- 其他保持R20的核心逻辑

理由：
1. R20固定出场(28/36)在快速下跌时反应慢
2. ATR出场是自适应的，能更好捕捉趋势转折
3. 保持仓位比例(L25%/S40%)和熊市过滤不变
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 基础指标 ──
    ema50 = close.ewm(span=50, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    vol_ma = volume.rolling(50).mean()

    # ── Keltner 轨道 ──
    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ADX 趋势强度 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── EMA150 熊市检测 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    # ── Donchian 入场（保持58周期） ──
    entry_high = high.rolling(58).max()

    # ── 做多系统（25%仓位）ATR动态出场 ──
    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_stop = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_stop = high.iloc[i] - 2 * atr.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            # ATR动态止损
            long_stop = max(long_stop, high.iloc[i] - 2 * atr.iloc[i])
            if close.iloc[i] < long_stop:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（40%仓位）ATR动态出场 ──
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_stop = 0.0

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
                short_stop = low.iloc[i] + 2 * atr.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            # ATR动态止损（向上追踪）
            short_stop = min(short_stop, low.iloc[i] + 2 * atr.iloc[i])
            if close.iloc[i] > short_stop:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)