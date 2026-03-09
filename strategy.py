"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进说明 (vs Round20):
- 熊市检测: EMA150(96h,-5%) → EMA100(48h,-3%)，降低阈值增加做空信号
- 做空出场: 固定36周期高点 → ATR追踪(2.5x)，更灵敏
- ADX阈值: 25 → 20，减少过滤过度

核心逻辑保持:
- 做多25%: Donchian58 + Keltner上轨2.0x + 成交量1.1x
- 做空40%: Keltner下轨2.0x + 熊市确认 + ADX>20 + ATR追踪出场
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

    # ── 做多系统（始终运行，25% 仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_stop_price = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_stop_price = exit_low.iloc[i - 1]
                long_signal.iloc[i] = 0.25
        else:
            # 固定出场
            if close.iloc[i] < long_stop_price:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25
                long_stop_price = max(long_stop_price, exit_low.iloc[i - 1])

    # ── 做空系统（熊市 + ATR追踪出场，40% 仓位） ──
    # 改进1: 使用EMA100(48h斜率)替代EMA150(96h斜率)，阈值-3%替代-5%
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema100_slope = ema100 / ema100.shift(48) - 1

    # 改进2: ATR追踪出场 (2.5x ATR)
    atr20 = tr.rolling(20).mean()
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_stop_price = 0.0

    for i in range(100, len(candles)):
        slope = ema100_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        # 熊市确认: 价格<EMA100 且 48h斜率<-3%
        bear_confirmed = close.iloc[i] < ema100.iloc[i] and slope < -0.03
        # 改进3: ADX阈值从25降到20
        adx_strong = adx.iloc[i] > 20 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                # ATR追踪止损初始化: 入场价 - 2.5 * ATR
                short_stop_price = close.iloc[i] + 2.5 * atr20.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            # ATR追踪: 持续上移止损
            new_stop = close.iloc[i] + 2.5 * atr20.iloc[i]
            short_stop_price = min(short_stop_price, new_stop)
            
            if close.iloc[i] > short_stop_price:
                in_short = False
            else:
                short_signal.iloc[i] = -0.40

    return (long_signal + short_signal).clip(-1.0, 1.0)