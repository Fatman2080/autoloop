"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进方向：结合R29(最佳val 2.3331)和R20(最佳test 3.89%)的优点
- 采用R29的核心逻辑框架
- EMA200替代EMA150，使熊市判定更稳定
- 做空系统加入更严格的ADX>30过滤，进一步减少震荡市亏损
- 保持25%做多+40%做空的仓位配比
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

    # Keltner通道 (使用1.5x作为做空系统，2.0x作为做多系统)
    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 1.5 * atr

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

    # ── EMA200 熊市检测 ──
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema200_slope = ema200 / ema200.shift(96) - 1  # 96周期约4天

    # ── 做多系统 (25%仓位) ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = 0.25
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统 (40%仓位, 仅熊市+强趋势) ──
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(200, len(candles)):
        slope = ema200_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        
        # 熊市判定: 价格低于EMA200且斜率下跌超过5%
        bear_confirmed = close.iloc[i] < ema200.iloc[i] and slope < -0.05
        
        # 更严格的ADX过滤 (ADX>30表示强趋势)
        adx_strong = adx.iloc[i] > 30 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -0.4
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)