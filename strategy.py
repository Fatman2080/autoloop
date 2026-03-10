"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤

改进方向：添加均线差距幅度过滤
理由：历史最佳策略（val_score=2.3479）使用成交量确认+动态均线周期，但未过滤均线差距过小的假信号。
      当EMA快线与慢线差距很小时，金叉/死叉往往是震荡市中的假信号。
      通过设置最小均线差距阈值（如2%），只在趋势明确时开仓，可减少无效交易。
      保持固定仓位0.8，专注于信号质量改进。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤策略
    
    1. 计算ATR百分比衡量市场波动率
    2. 根据波动率动态选择均线周期：高波动用EMA10/30，低波动用EMA20/60
    3. 成交量确认：仅当成交量超过20日均量时执行信号
    4. 均线差距过滤：仅当EMA快线与慢线差距超过2%时认可信号
    5. 保持固定仓位0.8
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    # 计算ATR（14日）和ATR百分比
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    atr_pct = atr14 / close
    
    # 根据ATR百分比动态选择均线周期
    ema_short_fast = close.ewm(span=10, adjust=False).mean()
    ema_short_slow = close.ewm(span=30, adjust=False).mean()
    ema_long_fast = close.ewm(span=20, adjust=False).mean()
    ema_long_slow = close.ewm(span=60, adjust=False).mean()
    
    # 创建动态均线序列
    ema_fast = pd.Series(index=close.index, dtype=float)
    ema_slow = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if i >= 60:
            if atr_pct.iloc[i] > 0.03:
                ema_fast.iloc[i] = ema_short_fast.iloc[i]
                ema_slow.iloc[i] = ema_short_slow.iloc[i]
            else:
                ema_fast.iloc[i] = ema_long_fast.iloc[i]
                ema_slow.iloc[i] = ema_long_slow.iloc[i]
        else:
            ema_fast.iloc[i] = close.iloc[i]
            ema_slow.iloc[i] = close.iloc[i]
    
    # 均线差距百分比（用于过滤）
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    
    # 成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号（金叉/死叉）
    signal = pd.Series(0.0, index=candles.index)
    signal[ema_fast > ema_slow] = 0.8
    signal[ema_fast < ema_slow] = -0.8
    
    # 均线差距过滤：差距需超过2%
    gap_threshold = 0.02
    strong_trend = ema_diff_pct.abs() > gap_threshold
    
    # 成交量确认
    volume_confirmed = volume > volume_ma20
    
    # 应用双重过滤
    final_signal = pd.Series(0.0, index=candles.index)
    for i in range(1, len(candles)):
        if volume_confirmed.iloc[i] and strong_trend.iloc[i]:
            final_signal.iloc[i] = signal.iloc[i]
        else:
            final_signal.iloc[i] = final_signal.iloc[i-1]
    
    final_signal.iloc[0] = signal.iloc[0]
    
    return final_signal.clip(-1.0, 1.0)