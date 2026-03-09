"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动说明：
- EMA周期从150→120：提高对趋势变化的敏感度，更早识别牛熊转换
- 加入ATR动态止盈：做多止盈=入场价+2*ATR，做空止盈=入场价-2*ATR
- 保持ADX>25趋势过滤，做空仓位保持20%严格限制

原理：ATR动态止盈可以让盈利持仓博取更多收益，同时在趋势逆转时及时退出。
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

    # ── 熊市检测（EMA120 提高敏感度） ──
    ema120 = close.ewm(span=120, adjust=False).mean()
    ema120_slope = ema120 / ema120.shift(96) - 1

    # ── 做多系统（始终运行，25% 仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(30).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_entry_price = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_entry_price = close.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            # 动态止盈：价格跌破入场价-2*ATR 或 跌破出场通道
            stop_loss = long_entry_price - 2 * atr14.iloc[i]
            if close.iloc[i] < exit_low.iloc[i - 1] or close.iloc[i] < stop_loss:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（仅熊市+强趋势，20% 仓位） ──
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_price = 0.0

    for i in range(120, len(candles)):
        slope = ema120_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema120.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.2
        else:
            # 动态止盈：价格涨破入场价+2*ATR 或 涨破出场通道
            stop_loss = short_entry_price + 2 * atr14.iloc[i]
            if close.iloc[i] > exit_high.iloc[i - 1] or close.iloc[i] > stop_loss:
                in_short = False
            else:
                short_signal.iloc[i] = -0.2

    return (long_signal + short_signal).clip(-1.0, 1.0)