"""
strategy.py — 量化交易策略
改动说明：
1. 做空仓位从 40% 提升到 50%（R75 的 L30%/S50% 在 val 上达到 2.3323）
2. 出场周期从 36 缩小到 30，捕捉更快的做空反转
3. 移除 ADX 过滤，简化做空触发条件，增加交易机会
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    输入：K线数据 DataFrame
    输出：仓位信号 Series，值在 -1.0 ~ 1.0 之间
    
    策略：独立叠加多空系统 + EMA 斜率熊市检测
    - 做多系统（始终运行）：Donchian(58h) + Keltner上轨(2.0x) + 成交量 → 25%
    - 做空系统（仅熊市）：Keltner下轨(2.0x) + 成交量 + 熊市确认 → 50%
    - 熊市判定：价格 < EMA(150) 且 EMA(150) 96h内下跌 > 5%
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

    # ── 做多系统（始终运行，25% 仓位） ──
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

    # ── 做空系统（仅在 EMA斜率熊市中激活，50% 仓位） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(30).max()  # 从 36 缩小到 30

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

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
                short_signal.iloc[i] = -0.5  # 从 -0.4 提升到 -0.5
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.5

    return (long_signal + short_signal).clip(-1.0, 1.0)