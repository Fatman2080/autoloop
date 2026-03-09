"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动说明（R45改进）:
- 保持R20核心架构：Donchian(58/28)+Keltner(2.0x)+多空独立系统+EMA150熊市检测
- 做空系统增加双重ADX过滤：ADX>30 且 +DI < -DI（趋势确认更严格）
- 做空出场收紧：36→30周期，减少趋势反转时的亏损
- 引入"趋势强度加分"：当ADX>35时，做多仓位从25%提升至30%，做空从40%提升至45%
- 理由：历史数据显示ADX>25过滤做空有效，但当前阈值可能过于宽松，ADX>30可进一步减少假突破
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 基础指标 ──
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

    # ── 趋势强度标记（用于仓位调整） ──
    strong_trend = adx > 35  # 强趋势标志

    # ── 做多系统（始终运行） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        # 动态仓位：强趋势时30%，正常25%
        long_size = 0.30 if strong_trend.iloc[i] else 0.25
        
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = long_size
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = long_size

    # ── 做空系统（熊市 + 双重ADX过滤） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(30).max()  # 收紧：36→30

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        
        # 双重ADX过滤：ADX>30 且 +DI < -DI（确认下跌趋势）
        adx_val = adx.iloc[i]
        adx_strong = (adx_val > 30 and plus_di.iloc[i] < minus_di.iloc[i]) if not np.isnan(adx_val) else False
        
        # 动态仓位：强趋势时45%，正常40%
        short_size = -0.45 if strong_trend.iloc[i] else -0.40

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = short_size
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = short_size

    return (long_signal + short_signal).clip(-1.0, 1.0)