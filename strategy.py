"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动说明（R44 → R45）：
- 改进点：收紧 EMA150 斜率阈值 (-0.05 → -0.03)，同时增加 200EMA 双重确认
- 理由：R20 做空系统在验证集表现优秀(train_score=0.103, val_score=1.870)，
        但训练集表现偏弱。收紧斜率阈值可减少熊市误判，使做空信号更精准，
        200EMA 作为二级过滤可提升趋势判断准确性
- 预期：在保持高验证集 Sharpe 的同时，提升训练集表现
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

    # ── 改进：使用 200EMA 作为二级趋势过滤 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    ema200_slope = ema200 / ema200.shift(96) - 1

    # ── 做多系统（始终运行，25% 仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = 0.25
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 改进：收紧做空条件，EMA150斜率阈值 -0.05→-0.03，加入200EMA双重确认 ──
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(200, len(candles)):
        slope150 = ema150_slope.iloc[i]
        slope200 = ema200_slope.iloc[i]
        if np.isnan(slope150):
            slope150 = 0.0
        if np.isnan(slope200):
            slope200 = 0.0
        
        # 收紧斜率阈值，并加入200EMA趋势确认
        bear_confirmed = (close.iloc[i] < ema150.iloc[i] 
                          and slope150 < -0.03 
                          and ema200.iloc[i] < ema200.iloc[i-20])  # 200EMA也在下跌
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -0.4
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)