"""
strategy.py — 量化交易策略
基于 R20 最佳策略 (val_score=1.870)，改进如下：
- 做空仓位从 40% 提升至 50%（参考 R29/R75 高分组合）
- 保持 EMA150 斜率熊市检测 + ADX>25 趋势过滤
- 保持独立叠加多空系统架构

改动说明：
历史结果显示 L25%/S40% 组合 (R19) 达到 val_score 1.732，
而 L30%/S50% 组合 (R75) 达到 val_score 2.3323。
本次尝试做空 50% 仓位，看能否突破 R20 的 1.870。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    输入：K线数据 DataFrame
    输出：仓位信号 Series (-1.0 ~ 1.0)
    
    策略：独立叠加多空系统 + EMA150斜率熊市检测 + ADX趋势强度
    - 做多系统：Donchian(58h) + Keltner上轨(2.0x) + 成交量 → 25%
    - 做空系统：Keltner下轨(2.0x) + 熊市确认(EMA150斜率<-5%) + ADX>25 → 50%
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

    # ── 做多系统（25% 仓位） ──
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

    # ── 做空系统（50% 仓位，熊市+强趋势时激活） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

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
                short_signal.iloc[i] = -0.5  # 从 -0.4 提升至 -0.5
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.5

    return (long_signal + short_signal).clip(-1.0, 1.0)