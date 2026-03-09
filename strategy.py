"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动：ATR追踪止损替代固定Donchian出场
- 做多出场：max(入场价 - 2*ATR14, 跟踪低点) 替代固定Donchian(28)低
- 做空出场：min(入场价 + 2*ATR14, 跟踪高点) 替代固定Donchian(36)高
- ATR止损更灵活，让盈利仓位继续奔跑，亏损时及时止损

其他参数保持R20不变：
- 做多25%，做空40%
- EMA150斜率<-5%熊市检测 + ADX>25趋势过滤
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 共用指标 ──
    ema50 = close.ewm(span=50, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    atr14 = tr.rolling(14).mean()
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 熊市检测 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    # ── 做多系统（25% 仓位，ATR追踪止损） ──
    entry_high = high.rolling(58).max()
    
    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_entry_price = 0.0
    long_stop_price = 0.0
    long_trailing_low = float('inf')

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_entry_price = close.iloc[i]
                long_trailing_low = low.iloc[i]
                long_stop_price = long_entry_price - 2.0 * atr14.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            # 更新跟踪低点
            long_trailing_low = min(long_trailing_low, low.iloc[i])
            # 动态止损：max(入场-2ATR, 跟踪低点)
            long_stop_price = max(long_entry_price - 2.0 * atr14.iloc[i], long_trailing_low)
            
            if close.iloc[i] < long_stop_price:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（40% 仓位，仅熊市+强趋势，ATR追踪止损） ──
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_price = 0.0
    short_trailing_high = 0.0

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_entry_price = close.iloc[i]
                short_trailing_high = high.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            # 更新跟踪高点
            short_trailing_high = max(short_trailing_high, high.iloc[i])
            # 动态止损：min(入场+2ATR, 跟踪高点)
            short_stop_price = min(short_entry_price + 2.0 * atr14.iloc[i], short_trailing_high)
            
            if close.iloc[i] > short_stop_price:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)