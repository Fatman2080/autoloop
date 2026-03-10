"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤 + 高周期趋势确认

改进方向：添加高周期EMA趋势确认
理由：当前最佳策略（val_score=3.0325）使用资金费率顺势过滤效果显著，但仅依赖单一周期信号。
      加入更高时间周期（如日线EMA120）的趋势判断，当信号方向与高周期趋势一致时加强信号，
      不一致时减弱信号。这能过滤逆周期的假信号，提升信号质量。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤 + 高周期趋势确认
    
    1. 计算ATR百分比衡量市场波动率
    2. 根据波动率动态选择均线周期：高波动用EMA10/30，低波动用EMA20/60
    3. 成交量确认：仅当成交量超过20日均量时执行信号
    4. 均线差距过滤：仅当EMA快线与慢线差距超过2%时认可信号
    5. 资金费率顺势过滤：当信号方向与资金费率方向一致时加强
    6. 高周期趋势确认：EMA120趋势向上时加强做多，向下时加强做空
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
    
    # 均线差距百分比
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    
    # 成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号
    signal_raw = pd.Series(0.0, index=candles.index)
    signal_raw[ema_fast > ema_slow] = 0.8
    signal_raw[ema_fast < ema_slow] = -0.8
    
    # 多重过滤条件
    gap_threshold = 0.02
    strong_trend = ema_diff_pct.abs() > gap_threshold
    volume_confirmed = volume > volume_ma20
    
    # 资金费率过滤
    funding_threshold = 0.00005
    funding_bullish = funding_rate > funding_threshold
    funding_bearish = funding_rate < -funding_threshold
    funding_neutral = ~funding_bullish & ~funding_bearish
    
    # 应用过滤并生成最终信号
    final_signal = pd.Series(0.0, index=candles.index)
    for i in range(1, len(candles)):
        if volume_confirmed.iloc[i] and strong_trend.iloc[i]:
            raw_sig = signal_raw.iloc[i]
            trend = trend_up.iloc[i]
            down_trend = trend_down.iloc[i]
            
            if raw_sig > 0:  # 做多信号
                base = 0.8
                # 资金费率调整
                if funding_bullish.iloc[i]:
                    base += 0.2
                elif funding_bearish.iloc[i]:
                    base -= 0.3
                # 高周期趋势调整
                if trend:
                    base = min(base + 0.1, 1.0)
                elif down_trend:
                    base -= 0.2
                final_signal.iloc[i] = base
            elif raw_sig < 0:  # 做空信号
                base = -0.8
                # 资金费率调整
                if funding_bearish.iloc[i]:
                    base -= 0.2
                elif funding_bullish.iloc[i]:
                    base += 0.3
                # 高周期趋势调整
                if down_trend:
                    base = max(base - 0.1, -1.0)
                elif trend:
                    base += 0.2
                final_signal.iloc[i] = base
            else:
                final_signal.iloc[i] = final_signal.iloc[i-1]
        else:
            final_signal.iloc[i] = final_signal.iloc[i-1]
    
    final_signal.iloc[0] = signal_raw.iloc[0]
    
    return final_signal.clip(-1.0, 1.0)