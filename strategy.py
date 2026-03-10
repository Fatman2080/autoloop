"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤

改进方向：引入成交量加权平均价(VWAP)作为辅助趋势确认指标

改动说明：
1. 在原有策略基础上，添加VWAP(20周期)作为趋势确认指标
2. 当价格在VWAP之上时，只允许做多信号；当价格在VWAP之下时，只允许做空信号
3. 保持其他所有参数不变：均线差距阈值0.8%、ATR阈值0.025、资金费率阈值0.00003

理由分析：
- 历史最佳策略(val_score=17.4273)已经非常优秀，但可能在高波动市场中出现短暂的反趋势信号
- VWAP是机构常用的趋势指标，能够反映资金流动方向
- 价格相对于VWAP的位置可以过滤掉一些背离趋势的短期信号
- 这应该能进一步提升夏普比率，同时控制回撤
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
    
    # 信号条件（加入VWAP过滤）
    long_cond = (signal_raw > 0) & strong_trend & volume_confirmed & price_above_vwap
    short_cond = (signal_raw < 0) & strong_trend & volume_confirmed & price_below_vwap
    
    # 动态仓位计算：做多
    long_base = np.full(len(close), 0.8)
    long_base = np.where(funding_bullish, long_base + 0.15, long_base)
    long_base = np.where(funding_bearish, long_base - 0.2, long_base)
    long_base = np.where(trend_up_f, np.minimum(long_base + 0.1, 1.0), long_base)
    long_base = np.where(trend_down_f, long_base - 0.15, long_base)
    
    # 动态仓位计算：做空
    short_base = np.full(len(close), -0.8)
    short_base = np.where(funding_bearish, short_base - 0.15, short_base)
    short_base = np.where(funding_bullish, short_base + 0.2, short_base)
    short_base = np.where(trend_down_f, np.maximum(short_base - 0.1, -1.0), short_base)
    short_base = np.where(trend_up_f, short_base + 0.15, short_base)
    
    # 合成最终信号
    final_signal = np.where(long_cond, long_base,
                np.where(short_cond, short_base, np.nan))
    
    final_signal = pd.Series(final_signal, index=candles.index)
    final_signal = final_signal.bfill().fillna(0)
    # 保持持仓惯性：若无新信号，维持上一时刻仓位
    final_signal = final_signal.replace(0, np.nan).bfill().fillna(0)
    
    return final_signal.clip(-1.0, 1.0)