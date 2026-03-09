"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进方向：EMA交叉 + 短期ATR止损代替Donchian
- 做多：EMA(20)上穿EMA(50)触发，做空：EMA(20)下穿EMA(50)触发
- 出场：ATR(14)*2 动态追踪止损代替固定周期最低/最高
- 保持EMA150熊市检测 + ADX>25做空过滤
- 仓位：做多25%，做空40%（同R20）

理由：Donchian通道在震荡市中容易频繁触发止损，
EMA交叉更平滑，能减少无效交易。ATR动态止损更灵活。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 共用技术指标 ──
    # EMA 交叉系统
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    # ATR 计算（用于动态止损）
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    
    # 成交量均线
    vol_ma = volume.rolling(50).mean()
    
    # 熊市检测：EMA150斜率
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    # ADX 趋势强度
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14_for_adx = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14_for_adx * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14_for_adx * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()
    
    # Keltner 轨道（用于成交量过滤）
    ema50_for_kelt = close.ewm(span=50, adjust=False).mean()
    keltner_upper = ema50_for_kelt + 2.0 * atr14
    keltner_lower = ema50_for_kelt - 2.0 * atr14

    # ── 做多系统：EMA金叉 + 成交量确认 ──
    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_stop = 0.0
    
    for i in range(51, len(candles)):
        # 金叉条件：ema20上穿ema50
        golden_cross = ema20.iloc[i] > ema50.iloc[i] and ema20.iloc[i-1] <= ema50.iloc[i-1]
        
        if not in_long:
            if (golden_cross 
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_stop = close.iloc[i] - 2.0 * atr14.iloc[i]  # 初始止损
                long_signal.iloc[i] = 0.25
        else:
            # ATR动态止损上移
            current_stop = close.iloc[i] - 2.0 * atr14.iloc[i]
            long_stop = max(long_stop, current_stop)
            
            if close.iloc[i] < long_stop:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统：EMA死叉 + 熊市 + ADX>25 + 成交量确认 ──
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_stop = 0.0
    
    for i in range(151, len(candles)):
        # 死叉条件：ema20下穿ema50
        death_cross = ema20.iloc[i] < ema50.iloc[i] and ema20.iloc[i-1] >= ema50.iloc[i-1]
        
        # 熊市确认
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        
        # ADX趋势强度
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False
        
        if not in_short:
            if (bear_confirmed 
                    and adx_strong
                    and death_cross
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_stop = close.iloc[i] + 2.0 * atr14.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            # ATR动态止损下移
            current_stop = close.iloc[i] + 2.0 * atr14.iloc[i]
            short_stop = min(short_stop, current_stop)
            
            if close.iloc[i] > short_stop:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)