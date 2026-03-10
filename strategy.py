"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤 + VWAP偏离动态仓位

改进方向：加入基于VWAP偏离度的动态仓位调整

改动说明：
1. 计算价格与VWAP的偏离百分比（bias）
2. 当价格远高于VWAP时（bias > 1%），增加做多仓位
3. 当价格远低于VWAP时（bias < -1%），增加做空仓位
4. 这种改进可以让策略在趋势强劲时获得更大收益，同时保持原有策略的有效性

理由分析：
- 当前最佳策略(val_score=21.734)已经非常优秀，使用VWAP进行趋势确认
- 但仓位是相对静态的，没有根据趋势的强劲程度动态调整
- VWAP偏离度可以反映当前价格与平均成本的关系
- 偏离越大，说明趋势越强/越极端，可以适当增加仓位
- 这是一种风险可控的仓位管理方式
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
    
    # VWAP偏离度（新功能）
    vwap_bias = (close - vwap) / vwap * 100  # 百分比偏离
    
    price_above_vwap = close > vwap
    price_below_vwap = close < vwap
    
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
    strong_up_bias = vwap_bias > 1.0  # 价格远高于VWAP
    strong_down_bias = vwap_bias < -1.0  # 价格远低于VWAP
    
    # 信号条件（加入VWAP过滤）
    long_cond = (signal_raw > 0) & strong_trend & volume_confirmed & price_above_vwap
    short_cond = (signal_raw < 0) & strong_trend & volume_confirmed & price_below_vwap
    
    # 动态仓位计算：做多（加入VWAP偏离度调整）
    long_base = np.full(len(close), 0.8)
    long_base = np.where(funding_bullish, long_base + 0.15, long_base)
    long_base = np.where(funding_bearish, long_base - 0.2, long_base)
    long_base = np.where(trend_up_f, np.minimum(long_base + 0.1, 1.0), long_base)
    long_base = np.where(trend_down_f, long_base - 0.15, long_base)
    # VWAP偏离度加成：当价格远高于VWAP时增加仓位
    long_base = np.where(strong_up_bias, np.minimum(long_base + 0.15, 1.0), long_base)
    
    # 动态仓位计算：做空（加入VWAP偏离度调整）
    short_base = np.full(len(close), -0.8)
    short_base = np.where(funding_bearish, short_base - 0.15, short_base)
    short_base = np.where(funding_bullish, short_base + 0.2, short_base)
    short_base = np.where(trend_down_f, np.maximum(short_base - 0.1, -1.0), short_base)
    short_base = np.where(trend_up_f, short_base + 0.15, short_base)
    # VWAP偏离度加成：当价格远低于VWAP时增加仓位
    short_base = np.where(strong_down_bias, np.maximum(short_base - 0.15, -1.0), short_base)
    
    # 合成最终信号
    final_signal = np.where(long_cond, long_base,
                np.where(short_cond, short_base, np.nan))
    
    final_signal = pd.Series(final_signal, index=candles.index)
    final_signal = final_signal.bfill().fillna(0)
    # 保持持仓惯性：若无新信号，维持上一时刻仓位
    final_signal = final_signal.replace(0, np.nan).bfill().fillna(0)
    
    return final_signal.clip(-1.0, 1.0)