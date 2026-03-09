"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进点(R21)：
1. ATR动态仓位：atr_ratio = atr / atr_ma，仓位 = 基础仓位 * (0.7 + 0.6 * atr_ratio)
   - 波动大时仓位更重，波动小时仓位更轻
2. 趋势判定改进：EMA150 + ATR斜率确认，替代单纯的斜率< -5%
   - 检测ATR是否在上升(趋势可能反转)，避免在震荡市做空
3. 保留ADX>25做空过滤

预期效果：ATR动态仓位能更好适应波动周期，减少低波动时的假信号
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
    atr14 = tr.rolling(14).mean()
    
    # ATR动态仓位因子
    atr_ma = atr.rolling(20).mean()
    atr_ratio = atr / atr_ma
    # 仓位因子: 0.7 ~ 1.3 (波动大时更激进)
    pos_factor = (0.7 + 0.6 * atr_ratio).clip(0.7, 1.3)
    
    # ATR斜率(用于趋势确认)
    atr_slope = atr / atr.shift(20) - 1

    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 趋势判定 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    # ── 做多系统（始终运行） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()
    vol_ma = volume.rolling(50).mean()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    base_long_pos = 0.25

    for i in range(58, len(candles)):
        long_pos = base_long_pos * pos_factor.iloc[i]
        
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = long_pos
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = long_pos

    # ── 做空系统（熊市 + ADX强趋势 + ATR上升确认） ──
    exit_high = high.rolling(36).max()
    base_short_pos = 0.40

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        
        # 熊市确认: 价格 < EMA150 且 斜率 < -5%
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        
        # ATR斜率确认: 上升表示波动加大，趋势可能延续
        atr_s = atr_slope.iloc[i]
        if np.isnan(atr_s):
            atr_s = 0.0
        atr_expanding = atr_s > -0.1  # ATR不急剧收缩
        
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False
        
        short_pos = base_short_pos * pos_factor.iloc[i]

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and atr_expanding
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -short_pos
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -short_pos

    return (long_signal + short_signal).clip(-1.0, 1.0)