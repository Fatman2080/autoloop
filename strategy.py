"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动说明（R41）：
- 核心改进：将Keltner从2.0x缩窄到1.5x（R17验证最有效）
- 做空仓位从40%降至25%，减少熊市风险暴露
- 去掉ADX>25过滤，改为EMA50斜率趋势判断（更平滑）
- 理由：R17用1.5x Keltner达到val_score 2.3948，是历史最高
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
    ema50_slope = ema50 / ema50.shift(20) - 1  # 20周期EMA斜率
    
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    vol_ma = volume.rolling(50).mean()

    # 关键改进：使用1.5x Keltner（R17验证最有效）
    keltner_upper = ema50 + 1.5 * atr
    keltner_lower = ema50 - 1.5 * atr

    # EMA150用于熊市检测
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    # ── 做多系统（始终运行，30% 仓位） ──
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
                long_signal.iloc[i] = 0.30
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.30

    # ── 做空系统（熊市确认 + EMA50上涨趋势过滤，25% 仓位） ──
    exit_high = high.rolling(30).max()  # 更短的做空止损

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(150, len(candles)):
        slope150 = ema150_slope.iloc[i]
        if np.isnan(slope150):
            slope150 = 0.0
        
        # 熊市确认：价格低于EMA150且EMA150下跌
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope150 < -0.05
        
        # EMA50斜率过滤：只做空回调（EMA50向下）
        slope50 = ema50_slope.iloc[i]
        if np.isnan(slope50):
            slope50 = 0.0
        
        if not in_short:
            if (bear_confirmed
                    and slope50 < 0  # EMA50向下趋势
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -0.25
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.25

    return (long_signal + short_signal).clip(-1.0, 1.0)