"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期调整

改进方向：基于价格波动率动态调整均线周期
理由：历史最佳策略（val_score=2.3053）使用固定EMA20/60，但在不同市场波动下表现不稳定。
      当市场波动率较高时，使用较短周期EMA（如10/30）能更快响应；波动率较低时，使用较长周期EMA（如20/60）更稳定。
      通过ATR波动率指标动态切换均线周期，适应不同市场环境，提升策略适应性。

改动：在ATR动态仓位基础上，根据ATR百分比调整均线周期
      当ATR_pct > 阈值（如0.03）时，使用短周期EMA10/EMA30；否则使用原EMA20/EMA60
      同时移除ATR仓位调整因子，保持信号纯净度
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认 + 动态均线周期策略
    
    1. 计算ATR百分比衡量市场波动率
    2. 根据波动率动态选择均线周期：高波动用EMA10/30，低波动用EMA20/60
    3. 成交量确认：仅当成交量超过20日均量时执行信号
    4. 保持固定仓位0.8，专注于信号质量改进
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
    # 高波动(ATR_pct>0.03)时使用短周期EMA10/30，低波动时使用EMA20/60
    ema_short_fast = close.ewm(span=10, adjust=False).mean()
    ema_short_slow = close.ewm(span=30, adjust=False).mean()
    ema_long_fast = close.ewm(span=20, adjust=False).mean()
    ema_long_fast = ema_long_fast.fillna(close)  # 避免NaN
    ema_long_slow = close.ewm(span=60, adjust=False).mean()
    
    # 创建动态均线序列
    ema_fast = pd.Series(index=close.index, dtype=float)
    ema_slow = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if i >= 60:  # 确保有足够数据
            if atr_pct.iloc[i] > 0.03:
                ema_fast.iloc[i] = ema_short_fast.iloc[i]
                ema_slow.iloc[i] = ema_short_slow.iloc[i]
            else:
                ema_fast.iloc[i] = ema_long_fast.iloc[i]
                ema_slow.iloc[i] = ema_long_slow.iloc[i]
        else:
            ema_fast.iloc[i] = close.iloc[i]
            ema_slow.iloc[i] = close.iloc[i]
    
    # 成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号（金叉/死叉）
    signal = pd.Series(0.0, index=candles.index)
    signal[ema_fast > ema_slow] = 0.8
    signal[ema_fast < ema_slow] = -0.8
    
    # 成交量确认
    volume_confirmed = volume > volume_ma20
    
    # 应用成交量确认
    final_signal = pd.Series(0.0, index=candles.index)
    for i in range(1, len(candles)):
        if volume_confirmed.iloc[i]:
            final_signal.iloc[i] = signal.iloc[i]
        else:
            final_signal.iloc[i] = final_signal.iloc[i-1]
    
    final_signal.iloc[0] = signal.iloc[0]
    
    return final_signal.clip(-1.0, 1.0)