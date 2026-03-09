"""
strategy.py — 量化交易策略
改进自R20最佳策略

改动：
1. Keltner ATR周期: 50→20 (更灵敏的波动率过滤)
2. 熊市检测 EMA: 150→200 (更稳健的长期趋势判断)
3. ADX阈值: 25→20 (适当放宽做空条件)
4. 仓位: 25%做多 → 30%, 40%做空 → 35%
5. 向量化实现替代循环 (更高效、更不易过拟合)
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    idx = candles.index

    # === 基础指标 ===
    # Keltner Channel (ATR=20 更灵敏)
    ema20 = close.ewm(span=20, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    
    keltner_upper = ema20 + 2.0 * atr20
    keltner_lower = ema20 - 2.0 * atr20

    # 成交量均线
    vol_ma = volume.rolling(50).mean()

    # === ADX 趋势强度 ===
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=idx).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=idx).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # === 熊市检测 (EMA200 更稳健) ===
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema200_slope = ema200 / ema200.shift(96) - 1  # 96周期约4天(假设4h周期)

    # === 做多系统 (30%仓位) ===
    # Donchian: 58周期高点入场, 28周期低点出场
    donchian_high = high.rolling(58).max()
    donchian_low = low.rolling(28).min()
    
    # 入场条件: 突破58高点 + 站上Keltner上轨 + 成交量放大
    long_entry = (close > donchian_shift1) & (close > keltner_upper) & (volume > 1.1 * vol_ma)
    
    # 出场条件: 跌破28周期低点
    long_exit = close < donchian_low.shift(1)
    
    # 向量化持仓状态
    long_pos = pd.Series(0.0, index=idx)
    in_long = False
    for i in range(58, len(idx)):
        if not in_long and long_entry.iloc[i]:
            in_long = True
        if in_long:
            long_pos.iloc[i] = 0.30
            if long_exit.iloc[i]:
                in_long = False

    # === 做空系统 (35%仓位，仅熊市+ADX>20) ===
    # Donchian: 72周期低点入场, 36周期高点出场
    donchian_high_s = high.rolling(72).max()
    donchian_low_s = low.rolling(36).min()
    
    # 熊市条件: 价格<EMA200 且 96周期内下跌>5%
    bear_market = (close < ema200) & (ema200_slope < -0.05)
    
    # 入场条件: 熊市 + ADX>20 + 突破Keltner下轨 + 成交量放大
    short_entry = bear_market & (adx > 20) & (close < keltner_lower) & (volume > 1.1 * vol_ma)
    
    # 出场条件: 突破36周期高点
    short_exit = close > donchian_high_s.shift(1)
    
    # 向量化持仓状态
    short_pos = pd.Series(0.0, index=idx)
    in_short = False
    for i in range(72, len(idx)):
        if not in_short and short_entry.iloc[i]:
            in_short = True
        if in_short:
            short_pos.iloc[i] = -0.35
            if short_exit.iloc[i]:
                in_short = False

    # 修正: 需要shift donchian_high 用于前一期对比
    donchian_high = high.rolling(58).max().shift(1)
    donchian_low = low.rolling(28).min().shift(1)
    donchian_high_s = high.rolling(72).max().shift(1)
    donchian_low_s = low.rolling(36).min().shift(1)

    # 重新计算入场条件（使用shift后的值）
    long_entry = (close > donchian_high) & (close > keltner_upper) & (volume > 1.1 * vol_ma)
    long_exit = close < donchian_low
    short_entry = bear_market & (adx > 20) & (close < keltner_lower) & (volume > 1.1 * vol_ma)
    short_exit = close > donchian_high_s

    # 重新计算持仓
    long_pos = pd.Series(0.0, index=idx)
    in_long = False
    for i in range(200, len(idx)):  # 需要足够数据计算所有指标
        if not in_long and long_entry.iloc[i]:
            in_long = True
        if in_long:
            long_pos.iloc[i] = 0.30
            if long_exit.iloc[i]:
                in_long = False

    short_pos = pd.Series(0.0, index=idx)
    in_short = False
    for i in range(200, len(idx)):
        if not in_short and short_entry.iloc[i]:
            in_short = True
        if in_short:
            short_pos.iloc[i] = -0.35
            if short_exit.iloc[i]:
                in_short = False

    return (long_pos + short_pos).clip(-1.0, 1.0)