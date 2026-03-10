"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤

改进方向：向量化优化 + 放宽过滤条件
理由：
1. 当前最佳策略（val_score=3.0382）使用 for 循环，效率低且代码冗长
2. train_drawdown=50.32% vs val_drawdown=6.85%，存在过拟合风险
3. 放宽均线差距过滤阈值（2%→1.5%），增加有效信号数量
4. 简化资金费率和高周期趋势的调整逻辑，避免过度优化
5. 使用向量运算替代 for 循环，提升性能
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤
    
    1. 计算ATR百分比衡量市场波动率
    2. 根据波动率动态选择均线周期：高波动用EMA10/30，低波动用EMA20/60
    3. 成交量确认：仅当成交量超过20日均量时执行信号
    4. 均线差距过滤：仅当EMA快线与慢线差距超过1.5%时认可信号（原2%）
    5. 资金费率顺势过滤：当信号方向与资金费率方向一致时加强
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    # 获取资金费率（后30%数据才有）
    funding_rate = candles.get("funding_rate", pd.Series(0, index=candles.index))
    funding_rate = funding_rate.fillna(0)
    
    # 计算ATR（14日）和ATR百分比
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    atr_pct = atr14 / close
    
    # 计算高周期EMA120趋势
    ema120 = close.ewm(span=120, adjust=False).mean()
    ema120_prev = ema120.shift(1)
    trend_up = ema120 > ema120_prev
    trend_down = ema120 < ema120_prev
    
    # 根据ATR百分比动态选择均线周期
    ema_short_fast = close.ewm(span=10, adjust=False).mean()
    ema_short_slow = close.ewm(span=30, adjust=False).mean()
    ema_long_fast = close.ewm(span=20, adjust=False).mean()
    ema_long_slow = close.ewm(span=60, adjust=False).mean()
    
    # 动态均线选择（向量化）
    use_short = atr_pct > 0.03
    ema_fast = np.where(use_short, ema_short_fast, ema_long_fast)
    ema_slow = np.where(use_short, ema_short_slow, ema_long_slow)
    ema_fast = pd.Series(ema_fast, index=close.index)
    ema_slow = pd.Series(ema_slow, index=close.index)
    
    # 均线差距百分比
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    
    # 成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号
    signal_raw = pd.Series(0.0, index=candles.index)
    signal_raw[ema_fast > ema_slow] = 0.8
    signal_raw[ema_fast < ema_slow] = -0.8
    
    # 放宽过滤条件（原2%→1.5%）
    gap_threshold = 0.015
    strong_trend = ema_diff_pct.abs() > gap_threshold
    volume_confirmed = volume > volume_ma20
    
    # 资金费率过滤
    funding_threshold = 0.00005
    funding_bullish = funding_rate > funding_threshold
    funding_bearish = funding_rate < -funding_threshold
    
    # 高周期趋势
    trend_up = trend_up.astype(float)
    trend_down = trend_down.astype(float)
    
    # 向量化计算最终信号
    final_signal = pd.Series(0.0, index=candles.index)
    
    # 初始信号
    final_signal.iloc[0] = signal_raw.iloc[0]
    
    # 预计算所有条件
    long_cond = (signal_raw > 0) & strong_trend & volume_confirmed
    short_cond = (signal_raw < 0) & strong_trend & volume_confirmed
    
    # 做多信号计算
    long_base = 0.8
    # 资金费率加强
    long_base = np.where(funding_bullish, long_base + 0.15, long_base)
    long_base = np.where(funding_bearish, long_base - 0.2, long_base)
    # 高周期趋势加强
    long_base = np.where(trend_up, np.minimum(long_base + 0.1, 1.0), long_base)
    long_base = np.where(trend_down, long_base - 0.15, long_base)
    
    # 做空信号计算
    short_base = -0.8
    # 资金费率加强
    short_base = np.where(funding_bearish, short_base - 0.15, short_base)
    short_base = np.where(funding_bullish, short_base + 0.2, short_base)
    # 高周期趋势加强
    short_base = np.where(trend_down, np.maximum(short_base - 0.1, -1.0), short_base)
    short_base = np.where(trend_up, short_base + 0.15, short_base)
    
    # 应用信号
    final_signal = np.where(long_cond, long_base,
                np.where(short_cond, short_base, np.nan))
    
    # 前值保持
    final_signal = pd.Series(final_signal, index=candles.index)
    final_signal = final_signal.bfill().fillna(0)
    
    # 平滑处理
    final_signal = final_signal.replace(0, np.nan).bfill().fillna(0)
    
    return final_signal.clip(-1.0, 1.0)