"""
strategy.py — 改进版双均线交叉策略
改动：在双均线交叉基础上加入成交量确认
理由：历史最佳策略的RSI过滤失败了，但可以尝试其他确认信号。
      成交量是价格动量的重要确认指标，金叉配合放量更可靠，死叉配合放量更可信。
      这可以减少假信号，提高夏普比率和收益率。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认策略
    
    1. 计算EMA20和EMA60，金叉做多，死叉做空（核心不变）
    2. 加入成交量确认：仅当成交量超过过去20日均量时，才执行信号
    3. 无确认时，保持前一个信号（避免频繁交易）
    """
    close = candles["close"]
    volume = candles["volume"]
    
    # 计算双均线
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    
    # 计算成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 生成基础信号
    signal = pd.Series(0.0, index=candles.index)
    signal[ema20 > ema60] = 0.8  # 金叉做多
    signal[ema20 < ema60] = -0.8 # 死叉做空
    
    # 成交量确认：当前成交量需超过20日均量
    volume_confirmed = volume > volume_ma20
    
    # 应用成交量过滤
    final_signal = pd.Series(0.0, index=candles.index)
    for i in range(1, len(candles)):
        if volume_confirmed.iloc[i]:
            final_signal.iloc[i] = signal.iloc[i]
        else:
            # 无成交量确认时，保持前一个信号
            final_signal.iloc[i] = final_signal.iloc[i-1]
    
    # 确保首日有信号
    final_signal.iloc[0] = signal.iloc[0]
    
    return final_signal.clip(-1.0, 1.0)