"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进点 (vs R20):
- 做空系统简化：去掉EMA150熊市限制，只保留ADX>20趋势强度过滤
- ATR追踪止损：代替固定的exit_high/low，动态跟踪价格波动
- Keltner参数微调：做空用1.5x（更敏感），做多用2.0x保持不变

理由:
- 历史显示ATR固定止损失败(R11)，但ATR追踪止损更灵活
- R29 val_score=2.33说明做空系统有潜力
- 减少限制条件可能增加有效交易次数
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
    keltner_lower = ema50 - 1.5 * atr  # 调窄做空轨道

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

    # ── 做空系统（简化版：仅ADX>20强趋势，无熊市限制） ──
    exit_high = high.rolling(36).max()
    
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    atr_stop = 0.0
    
    for i in range(150, len(candles)):
        adx_val = adx.iloc[i]
        if np.isnan(adx_val):
            adx_val = 0.0
        adx_strong = adx_val > 20  # 降低阈值增加机会
        
        if not in_short:
            if (adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                atr_stop = 2.5 * atr.iloc[i]  # ATR追踪止损起点
                short_signal.iloc[i] = -0.35  # 仓位降至35%
        else:
            # ATR追踪止损
            stop_price = close.iloc[i] + atr_stop
            if close.iloc[i] > exit_high.iloc[i - 1] or close.iloc[i] > stop_price:
                in_short = False
            else:
                # 动态更新止损位（仅收窄不放宽）
                new_stop = 2.0 * atr.iloc[i]
                if new_stop < atr_stop:
                    atr_stop = new_stop
                short_signal.iloc[i] = -0.35

    return (long_signal + short_signal).clip(-1.0, 1.0)