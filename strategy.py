"""
strategy.py — 双均线交叉 + 成交量确认 + ATR动态仓位
改进：在成交量确认基础上，加入ATR（平均真实波幅）来动态调整仓位。
理由：当前最佳策略使用成交量确认（val_score=2.2373），但仓位固定±0.8。
      ATR可衡量市场波动性，在高波动时降低仓位以减少回撤，在低波动时增加仓位以提高收益。
      这样可在不增加信号复杂度的情况下，优化仓位管理，提升Sharpe和收益率。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认 + ATR动态仓位策略
    
    1. 计算EMA20和EMA60，金叉做多，死叉做空
    2. 成交量确认：仅当成交量超过20日均量时执行信号
    3. ATR动态仓位：根据ATR占价格的百分比动态调整仓位（高波动低仓位，低波动高仓位）
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    # 计算双均线
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    
    # 计算成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 计算ATR（14日）
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    
    # ATR占价格百分比（波动率）
    atr_pct = atr14 / close
    
    # 基础信号
    signal = pd.Series(0.0, index=candles.index)
    signal[ema20 > ema60] = 0.8
    signal[ema20 < ema60] = -0.8
    
    # 成交量确认
    volume_confirmed = volume > volume_ma20
    
    # 计算动态仓位因子：波动率越低仓位越高，波动率越高仓位越低
    # ATR百分比均值约为0.02~0.05，使用0.03作为基准
    atr_factor = np.clip(0.03 / atr_pct, 0.3, 1.5)
    
    # 应用成交量确认 + 动态仓位
    final_signal = pd.Series(0.0, index=candles.index)
    for i in range(1, len(candles)):
        if volume_confirmed.iloc[i]:
            final_signal.iloc[i] = signal.iloc[i] * atr_factor.iloc[i]
        else:
            final_signal.iloc[i] = final_signal.iloc[i-1]
    
    final_signal.iloc[0] = signal.iloc[0]
    
    return final_signal.clip(-1.0, 1.0)