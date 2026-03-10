import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    改进方向：在现有最佳策略基础上，加入动量确认机制，避免在震荡市中频繁交易。
    具体改动：添加ROC(10)指标，当价格动量与信号方向一致时才开仓，提高信号质量。
    理由分析：历史最佳策略(val_score=23.892)已很优秀，但在震荡行情中可能出现假信号。
    加入动量过滤可以在趋势启动阶段增强信号，在动量减弱时减少仓位，优化风险调整收益。
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    funding_rate = candles.get("funding_rate", pd.Series(0, index=candles.index)).fillna(0)
    
    # ATR计算
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    atr_pct = atr14 / close
    
    # VWAP计算
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
    
    # VWAP偏离度
    vwap_bias = (close - vwap) / vwap * 100
    price_above_vwap = close > vwap
    price_below_vwap = close < vwap
    
    # 布林带计算
    ema20 = close.ewm(span=20, adjust=False).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = ema20 + 2 * bb_std
    bb_lower = ema20 - 2 * bb_std
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
    bb_breakout_up = close > bb_upper
    bb_breakout_down = close < bb_lower
    bb_middle_zone = (bb_position > 0.4) & (bb_position < 0.6)
    
    # 长期趋势过滤
    ema120 = close.ewm(span=120, adjust=False).mean()
    ema120_prev = ema120.shift(1)
    trend_up = ema120 > ema120_prev
    trend_down = ema120 < ema120_prev
    
    # 动量指标（新增）：ROC(10)
    roc = (close - close.shift(10)) / close.shift(10) * 100
    
    # 动态选择均线周期
    use_short = atr_pct > 0.025
    ema_fast = np.where(use_short, close.ewm(span=10, adjust=False).mean(),
                        close.ewm(span=20, adjust=False).mean())
    ema_slow = np.where(use_short, close.ewm(span=30, adjust=False).mean(),
                        close.ewm(span=60, adjust=False).mean())
    ema_fast = pd.Series(ema_fast, index=close.index)
    ema_slow = pd.Series(ema_slow, index=close.index)
    
    # 指标计算
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号方向
    signal_raw = pd.Series(0.0, index=candles.index)
    signal_raw[ema_fast > ema_slow] = 0.8
    signal_raw[ema_fast < ema_slow] = -0.8
    
    # 过滤条件
    gap_threshold = 0.008
    strong_trend = ema_diff_pct.abs() > gap_threshold
    volume_confirmed = volume > volume_ma20
    funding_threshold = 0.00003
    funding_bullish = funding_rate > funding_threshold
    funding_bearish = funding_rate < -funding_threshold
    strong_up_bias = vwap_bias > 1.0
    strong_down_bias = vwap_bias < -1.0
    trend_up_f = trend_up.astype(float)
    trend_down_f = trend_down.astype(float)
    
    # 动量过滤（新增）：方向一致确认
    momentum_confirmed_long = roc > 0.5  # 上涨动量确认
    momentum_confirmed_short = roc < -0.5  # 下跌动量确认
    
    # 信号条件（加入动量过滤）
    long_cond = ((signal_raw > 0) & strong_trend & volume_confirmed & 
                 price_above_vwap & momentum_confirmed_long)
    short_cond = ((signal_raw < 0) & strong_trend & volume_confirmed & 
                  price_below_vwap & momentum_confirmed_short)
    
    # 动态仓位计算：做多
    long_base = np.full(len(close), 0.8)
    long_base = np.where(funding_bullish, long_base + 0.15, long_base)
    long_base = np.where(funding_bearish, long_base - 0.2, long_base)
    long_base = np.where(trend_up_f, np.minimum(long_base + 0.1, 1.0), long_base)
    long_base = np.where(trend_down_f, long_base - 0.15, long_base)
    long_base = np.where(strong_up_bias, np.minimum(long_base + 0.15, 1.0), long_base)
    long_base = np.where(bb_breakout_up, np.minimum(long_base + 0.2, 1.0), long_base)
    long_base = np.where(bb_middle_zone, long_base * 0.7, long_base)
    
    # 动态仓位计算：做空
    short_base = np.full(len(close), -0.8)
    short_base = np.where(funding_bearish, short_base - 0.15, short_base)
    short_base = np.where(funding_bullish, short_base + 0.2, short_base)
    short_base = np.where(trend_down_f, np.maximum(short_base - 0.1, -1.0), short_base)
    short_base = np.where(trend_up_f, short_base + 0.15, short_base)
    short_base = np.where(strong_down_bias, np.maximum(short_base - 0.15, -1.0), short_base)
    short_base = np.where(bb_breakout_down, np.maximum(short_base - 0.2, -1.0), short_base)
    short_base = np.where(bb_middle_zone, short_base * 0.7, short_base)
    
    # 合成最终信号
    final_signal = np.where(long_cond, long_base,
                np.where(short_cond, short_base, np.nan))
    
    final_signal = pd.Series(final_signal, index=candles.index)
    final_signal = final_signal.bfill().fillna(0)
    final_signal = final_signal.replace(0, np.nan).bfill().fillna(0)
    
    return final_signal.clip(-1.0, 1.0)