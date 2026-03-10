"""
strategy.py — 双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤

改进方向：添加资金费率顺势过滤
理由：历史最佳策略（val_score=2.9215）已包含动态均线周期和均线差距过滤，但未利用资金费率数据。
      资金费率反映了永续合约市场的多空平衡，正资金费率（多头支付空头）通常表明市场情绪偏多，
      负资金费率则表明市场情绪偏空。当信号方向与资金费率方向一致时（如做多信号+正资金费率），
      应加强信号；反之应减弱信号。这能利用衍生品市场的情绪数据提升信号质量。
      仅修改资金费率过滤部分，保持其他条件不变，便于单独评估效果。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    双均线交叉 + 成交量确认 + 动态均线周期 + 均线差距过滤 + 资金费率顺势过滤
    
    1. 计算ATR百分比衡量市场波动率
    2. 根据波动率动态选择均线周期：高波动用EMA10/30，低波动用EMA20/60
    3. 成交量确认：仅当成交量超过20日均量时执行信号
    4. 均线差距过滤：仅当EMA快线与慢线差距超过2%时认可信号
    5. 资金费率顺势过滤：当信号方向与资金费率方向一致时（做多+正资金费率，做空+负资金费率），
       信号强度从0.8提升至1.0；不一致时降至0.5；中性资金费率保持0.8
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    # 获取资金费率（后30%数据才有，前面为NaN）
    funding_rate = candles.get("funding_rate", pd.Series(0, index=candles.index))
    funding_rate = funding_rate.fillna(0)
    
    # 计算ATR（14日）和ATR百分比
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    atr_pct = atr14 / close
    
    # 根据ATR百分比动态选择均线周期
    ema_short_fast = close.ewm(span=10, adjust=False).mean()
    ema_short_slow = close.ewm(span=30, adjust=False).mean()
    ema_long_fast = close.ewm(span=20, adjust=False).mean()
    ema_long_slow = close.ewm(span=60, adjust=False).mean()
    
    # 创建动态均线序列
    ema_fast = pd.Series(index=close.index, dtype=float)
    ema_slow = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if i >= 60:
            if atr_pct.iloc[i] > 0.03:
                ema_fast.iloc[i] = ema_short_fast.iloc[i]
                ema_slow.iloc[i] = ema_short_slow.iloc[i]
            else:
                ema_fast.iloc[i] = ema_long_fast.iloc[i]
                ema_slow.iloc[i] = ema_long_slow.iloc[i]
        else:
            ema_fast.iloc[i] = close.iloc[i]
            ema_slow.iloc[i] = close.iloc[i]
    
    # 均线差距百分比（用于过滤）
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    
    # 成交量20日均线
    volume_ma20 = volume.rolling(window=20).mean()
    
    # 基础信号（金叉/死叉），保持0.8基础强度
    signal_raw = pd.Series(0.0, index=candles.index)
    signal_raw[ema_fast > ema_slow] = 0.8
    signal_raw[ema_fast < ema_slow] = -0.8
    
    # 均线差距过滤：差距需超过2%
    gap_threshold = 0.02
    strong_trend = ema_diff_pct.abs() > gap_threshold
    
    # 成交量确认
    volume_confirmed = volume > volume_ma20
    
    # 资金费率顺势过滤
    funding_threshold = 0.00005  # 0.5bps阈值，过滤微小波动
    funding_bullish = funding_rate > funding_threshold
    funding_bearish = funding_rate < -funding_threshold
    funding_neutral = ~funding_bullish & ~funding_bearish
    
    # 应用多重过滤
    final_signal = pd.Series(0.0, index=candles.index)
    for i in range(1, len(candles)):
        if volume_confirmed.iloc[i] and strong_trend.iloc[i]:
            raw_signal = signal_raw.iloc[i]
            
            # 根据资金费率调整信号强度
            if raw_signal > 0:  # 做多信号
                if funding_bullish.iloc[i]:
                    final_signal.iloc[i] = 1.0  # 增强做多
                elif funding_bearish.iloc[i]:
                    final_signal.iloc[i] = 0.5  # 减弱做多
                else:
                    final_signal.iloc[i] = 0.8  # 中性
            elif raw_signal < 0:  # 做空信号
                if funding_bearish.iloc[i]:
                    final_signal.iloc[i] = -1.0  # 增强做空
                elif funding_bullish.iloc[i]:
                    final_signal.iloc[i] = -0.5  # 减弱做空
                else:
                    final_signal.iloc[i] = -0.8  # 中性
            else:
                final_signal.iloc[i] = final_signal.iloc[i-1]
        else:
            final_signal.iloc[i] = final_signal.iloc[i-1]
    
    final_signal.iloc[0] = signal_raw.iloc[0]
    
    return final_signal.clip(-1.0, 1.0)