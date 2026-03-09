"""
strategy.py — 量化交易策略
改进方向：R20稳健仓位(L25%/S40%) + ATR追踪止损替代固定周期出场
- 做多出场：使用 ATR 追踪，保留盈利趋势
- 做空出场：使用 ATR 追踪，减少过早平仓
- 保持 EMA150 斜率熊市检测 + ADX>25 趋势过滤
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
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── ATR 追踪止损参数 ──
    atr20 = tr.rolling(20).mean()
    atr_mult = 2.5  # ATR倍数

    # ── 做多系统（25% 仓位） ──
    entry_high = high.rolling(58).max()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_stop = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                # 入场价 - ATR 作为初始止损
                long_stop = low.iloc[i] - atr_mult * atr20.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            # ATR 追踪止损更新
            current_stop = low.iloc[i] - atr_mult * atr20.iloc[i]
            # 只上移止损，不下移
            long_stop = max(long_stop, current_stop)
            
            if close.iloc[i] < long_stop:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（40% 仓位，仅熊市+强趋势激活） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_stop = 0.0

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
                # 入场价 + ATR 作为初始止损
                short_stop = high.iloc[i] + atr_mult * atr20.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            # ATR 追踪止损更新
            current_stop = high.iloc[i] + atr_mult * atr20.iloc[i]
            # 只下移止损，不上移
            short_stop = min(short_stop, current_stop)
            
            if close.iloc[i] > short_stop:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)