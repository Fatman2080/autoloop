"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    基线策略：双均线交叉（EMA20/EMA60）

    - EMA20 > EMA60 → 做多 0.5
    - EMA20 < EMA60 → 做空 0.3
    - 简单基线，供 LLM 迭代改进
    """
    close = candles["close"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()

    signal = pd.Series(0.0, index=candles.index)
    signal[ema20 > ema60] = 0.5
    signal[ema20 < ema60] = -0.3

    return signal.clip(-1.0, 1.0)
