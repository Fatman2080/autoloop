"""
strategy.py — 量化交易策略
改动说明：
- 将EMA熊市检测从150改为200（更稳定的趋势过滤，减少假信号）
- 仓位调整为 L30% / S50%（基于R75验证有效的仓位比例）
- 保持ADX>25趋势过滤有效减少做空震荡亏损
- 保持Keltner 2.0x宽度和成交量1.1x确认
- 保持Donchian 58/28参数（历史验证有效）
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    独立叠加多空系统 + EMA200斜率熊市检测 + ADX趋势强度
    - 做多系统（始终运行）：Donchian(58h) + Keltner上轨(2.0x) + 成交量 → 30%
    - 做空系统（仅熊市+强趋势）：Keltner下轨(2.0x) + 成交量 + EMA200熊市确认 + ADX>25 → 50%
    - 熊市判定：价格 < EMA(200) 且 EMA(200) 96h内下跌 > 5%
    - ADX>25 过滤弱趋势做空，减少震荡市亏损
    """
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

    # ── 做多系统（始终运行，30% 仓位） ──
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
                long_signal.iloc[i] = 0.30
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.30

    # ── 做空系统（仅在 EMA200熊市 + ADX强趋势 中激活，50% 仓位） ──
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema200_slope = ema200 / ema200.shift(96) - 1
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(200, len(candles)):
        slope = ema200_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema200.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -0.50
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.50

    return (long_signal + short_signal).clip(-1.0, 1.0)