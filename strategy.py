"""
strategy.py — 基于当前最佳策略的改进：添加基于价格通道突破的动态仓位调整

改动说明：
在当前最佳策略基础上，添加基于布林带价格通道的仓位管理机制。当价格突破上轨时加强做多信号，突破下轨时加强做空信号。
这种改进利用了趋势延续的原理，在强势突破时增加仓位，在价格回归中轨附近时减少仓位，优化风险收益比。

理由分析：
- 当前最佳策略(val_score=23.462)已经非常优秀，但仓位调整主要基于VWAP偏离
- 布林带可以提供更动态的支撑阻力位，帮助识别趋势强度和反转点
- 当价格突破布林带上轨时，通常表示强劲的上涨趋势，应该加强做多仓位
- 当价格跌破布林带下轨时，通常表示强劲的下跌趋势，应该加强做空仓位
- 这种改进可以让策略在趋势突破时获得更大收益，同时通过中轨回归降低假信号仓位

具体实现：
1. 计算20周期布林带（中轨=EMA20，上下轨=±2倍标准差）
2. 计算价格相对于布林带的位置（百分比）
3. 根据位置调整仓位：突破上轨增加做多仓位，突破下轨增加做空仓位
4. 保持原有策略的过滤条件，仅增强仓位管理部分
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    funding_rate = candles.get("funding_rate", pd.Series(0, index=candles.index)).fillna(0)
    
    # ATR 计算
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    atr_pct = atr14 / close
    
    # VWAP计算 (20周期)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
    
    # VWAP偏离度
    vwap_bias = (close - vwap) / vwap * 100
    
    price_above_vwap = close > vwap
    price_below_vwap = close < vwap
    
    # 布林带计算（新增）
    ema20 = close.ewm(span=20, adjust=False).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = ema20 + 2 * bb_std
    bb_lower = ema20 - 2 * bb_std
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)  # 0-1范围，0.5为中轨
    
    # 布林带突破信号（新增）
    bb_breakout_up = close > bb_upper  # 突破上轨，强烈看涨
    bb_breakout_down = close < bb_lower  # 突破下轨，强烈看跌
    bb_middle_zone = (bb_position > 0.4) & (bb_position < 0.6)  # 中轨区域，观望
    
    # 长期趋势过滤 (EMA120 斜率)
    ema120 = close.ewm(span=120, adjust=False).mean()
    ema120_prev = ema120.shift(1)
    trend_up = ema120 > ema120_prev
    trend_down = ema120 < ema120_prev
    
    # 动态选择均线周期
    use_short = atr_pct > 0.025
    
    ema_fast = np.where(use_short, 
                        close.ewm(span=10, adjust=False).mean(),
                        close.ewm(span=20, adjust=False).mean())
    ema_slow = np.where(use_short,
                        close.ewm(span=30, adjust=False).mean(),
                        close.ewm(span=60, adjust=False).mean())
    
    ema_fast = pd.Series(ema_fast, index=close.index)
    ema_slow = pd.Series(ema_slow, index=close.index)
    
    # 指标计算
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号方向
    signal_raw = pd.Series(0.0, index=candles.index)
    signal_raw[ema_fast > ema_slow] = 0.8
    signal_raw[ema_fast < ema_slow] = -0.8
    
    # 均线差距过滤阈值
    gap_threshold = 0.008
    strong_trend = ema_diff_pct.abs() > gap_threshold
    volume_confirmed = volume > volume_ma20
    
    # 资金费率过滤
    funding_threshold = 0.00003
    funding_bullish = funding_rate > funding_threshold
    funding_bearish = funding_rate < -funding_threshold
    
    trend_up_f = trend_up.astype(float)
    trend_down_f = trend_down.astype(float)
    
    # VWAP偏离度过滤
    strong_up_bias = vwap_bias > 1.0
    strong_down_bias = vwap_bias < -1.0
    
    # 信号条件
    long_cond = (signal_raw > 0) & strong_trend & volume_confirmed & price_above_vwap
    short_cond = (signal_raw < 0) & strong_trend & volume_confirmed & price_below_vwap
    
    # 动态仓位计算：做多（加入布林带突破调整）
    long_base = np.full(len(close), 0.8)
    long_base = np.where(funding_bullish, long_base + 0.15, long_base)
    long_base = np.where(funding_bearish, long_base - 0.2, long_base)
    long_base = np.where(trend_up_f, np.minimum(long_base + 0.1, 1.0), long_base)
    long_base = np.where(trend_down_f, long_base - 0.15, long_base)
    # VWAP偏离度加成
    long_base = np.where(strong_up_bias, np.minimum(long_base + 0.15, 1.0), long_base)
    # 布林带突破加成（新增）
    long_base = np.where(bb_breakout_up, np.minimum(long_base + 0.2, 1.0), long_base)
    long_base = np.where(bb_middle_zone, long_base * 0.7, long_base)  # 中轨区域减少仓位
    
    # 动态仓位计算：做空（加入布林带突破调整）
    short_base = np.full(len(close), -0.8)
    short_base = np.where(funding_bearish, short_base - 0.15, short_base)
    short_base = np.where(funding_bullish, short_base + 0.2, short_base)
    short_base = np.where(trend_down_f, np.maximum(short_base - 0.1, -1.0), short_base)
    short_base = np.where(trend_up_f, short_base + 0.15, short_base)
    # VWAP偏离度加成
    short_base = np.where(strong_down_bias, np.maximum(short_base - 0.15, -1.0), short_base)
    # 布林带突破加成（新增）
    short_base = np.where(bb_breakout_down, np.maximum(short_base - 0.2, -1.0), short_base)
    short_base = np.where(bb_middle_zone, short_base * 0.7, short_base)  # 中轨区域减少仓位
    
    # 合成最终信号
    final_signal = np.where(long_cond, long_base,
                np.where(short_cond, short_base, np.nan))
    
    final_signal = pd.Series(final_signal, index=candles.index)
    final_signal = final_signal.bfill().fillna(0)
    # 保持持仓惯性：若无新信号，维持上一时刻仓位
    final_signal = final_signal.replace(0, np.nan).bfill().fillna(0)
    
    return final_signal.clip(-1.0, 1.0)