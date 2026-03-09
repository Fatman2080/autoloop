"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动：ATR动态止盈/止损替代固定出场
- 做多止盈：exit_low 改为 ATR 追踪，止盈位 = 持仓成本 + 2.5*ATR
- 做空止盈：exit_high 改为 ATR 追踪，止盈位 = 持仓成本 - 2.5*ATR
- 止损：入场价 - 2*ATR（做多），入场价 + 2*ATR（做空）

理由：固定出场在趋势反转时容易被打掉，ATR动态出场能更好捕捉趋势。
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

    # ── EMA 斜率熊市检测 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    # ── 做多系统（始终运行，25% 仓位）──
    entry_high = high.rolling(58).max()
    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    entry_price = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                entry_price = close.iloc[i]
                long_signal.iloc[i] = 0.25
        else:
            current_atr = atr.iloc[i]
            stop_loss = entry_price - 2.0 * current_atr
            take_profit = entry_price + 2.5 * current_atr
            
            if close.iloc[i] < stop_loss:
                in_long = False
            elif close.iloc[i] > take_profit:
                # 止盈后平一半，留仓位继续博
                long_signal.iloc[i] = 0.125
                in_long = False
        # 保持持仓
        if in_long:
            long_signal.iloc[i] = 0.25

    # ── 做空系统（仅在熊市+强趋势中激活，40% 仓位）──
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    entry_price_short = 0.0

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
                entry_price_short = close.iloc[i]
                short_signal.iloc[i] = -0.4
        else:
            current_atr = atr.iloc[i]
            stop_loss = entry_price_short + 2.0 * current_atr
            take_profit = entry_price_short - 2.5 * current_atr
            
            if close.iloc[i] > stop_loss:
                in_short = False
            elif close.iloc[i] < take_profit:
                short_signal.iloc[i] = -0.2
                in_short = False
        # 保持持仓
        if in_short:
            short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)