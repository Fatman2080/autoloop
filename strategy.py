"""
strategy.py — 量化交易策略

改进说明（基于R86 val_score=2.1159）：
- 将固定出场周期(36)改为ATR动态止损：做空时在入场价 - 2.0*ATR处设置止损
- ATR止损让退出点自适应市场波动，在剧烈波动时保护利润，在震荡时避免被洗出
- 保持做空仓位70%，Keltner 2.5x，EMA150熊市检测，ADX>25，成交量1.1x确认

预期：ATR动态止损比固定周期更灵活，能更好保护盈利
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
    atr14 = tr.rolling(14).mean()  # 用于动态止损
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.5 * atr
    keltner_lower = ema50 - 2.5 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

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
                in_short = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（EMA斜率熊市 + ADX强趋势 + ATR动态止损，70% 仓位） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    entry_price = 0.0
    stop_loss = 0.0

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
                entry_price = close.iloc[i]
                stop_loss = entry_price + 2.0 * atr14.iloc[i]  # 2倍ATR止损
                short_signal.iloc[i] = -0.70
        else:
            # ATR动态止损：价格突破止损位或突破固定周期则退出
            exit_high = high.rolling(36).max().iloc[i - 1]
            if close.iloc[i] > stop_loss or close.iloc[i] > exit_high:
                in_short = False
                entry_price = 0.0
                stop_loss = 0.0
            else:
                # 更新止损位（向上跟踪，但只允许上升不允许下降）
                new_stop = close.iloc[i] + 2.0 * atr14.iloc[i]
                if new_stop < stop_loss:
                    stop_loss = new_stop
                short_signal.iloc[i] = -0.70

    return (long_signal + short_signal).clip(-1.0, 1.0)