"""
改进版策略 v5.0
核心理念：保持原版的高胜率入场逻辑，优化盈亏比

问题诊断：
- 原版策略胜率58.33%（很好），但盈亏比0.85（问题所在）
- 平均盈利 < 平均亏损，长期会侵蚀收益

解决方案：
1. 保持严格的入场条件（多条件确认）
2. 优化止损：更紧的初始止损，减少单笔亏损
3. 优化止盈：分批止盈，让利润奔跑
4. 智能移动止损：保护利润
"""
import pandas as pd
import numpy as np
from typing import Tuple


class ImprovedStrategy:
    """改进版策略 v5.0 - 保持高胜率，优化盈亏比"""
    
    def __init__(
        self,
        # EMA参数（与原版一致）
        ema_fast: int = 12,
        ema_slow: int = 26,
        ema_trend: int = 55,
        # RSI参数
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        # ATR参数 - 关键优化
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.8,       # 更紧的止损（原2.5）
        atr_take_profit_multiplier: float = 4.5, # 更大的止盈目标（原3.0）
        # ADX趋势过滤
        adx_period: int = 14,
        adx_threshold: int = 20,
        # 风险管理
        risk_per_trade: float = 0.02,
        # 智能移动止损 - 分级
        use_trailing_stop: bool = True,
        # 信号控制
        min_bars_between_signals: int = 15,
        require_all_conditions: bool = True,
        # 过滤器
        use_volume_filter: bool = True,
        volume_ma_period: int = 20,
        min_volume_ratio: float = 0.9,
        # 分批止盈
        use_partial_take_profit: bool = True,
        partial_tp_ratio: float = 0.5,          # 第一次止盈平掉50%
        partial_tp_multiplier: float = 2.5,     # 第一止盈点：2.5倍ATR
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.risk_per_trade = risk_per_trade
        self.use_trailing_stop = use_trailing_stop
        self.min_bars_between_signals = min_bars_between_signals
        self.require_all_conditions = require_all_conditions
        self.use_volume_filter = use_volume_filter
        self.volume_ma_period = volume_ma_period
        self.min_volume_ratio = min_volume_ratio
        self.use_partial_take_profit = use_partial_take_profit
        self.partial_tp_ratio = partial_tp_ratio
        self.partial_tp_multiplier = partial_tp_multiplier
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # EMA
        df['EMA_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['EMA_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/self.atr_period, adjust=False).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr_smooth = df['ATR'].replace(0, np.nan)
        plus_di = 100 * plus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr_smooth
        minus_di = 100 * minus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr_smooth
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=1/self.adx_period, adjust=False).mean().fillna(0)
        df['plus_DI'] = plus_di.fillna(0)
        df['minus_DI'] = minus_di.fillna(0)
        
        # 成交量
        if self.use_volume_filter and 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(self.volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        else:
            df['volume_ratio'] = 1.0
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    def _calculate_signal_strength(self, row: pd.Series) -> float:
        """计算信号强度 (-1 到 1)"""
        score = 0.0
        atr = row['ATR'] if pd.notna(row['ATR']) and row['ATR'] > 0 else 1
        
        # EMA排列强度 (30%)
        ema_diff = (row['EMA_fast'] - row['EMA_slow']) / atr
        score += 0.30 * np.clip(ema_diff / 2, -1, 1)
        
        # ADX趋势强度 * DI方向 (30%)
        adx_strength = np.clip((row['ADX'] - 20) / 30, 0, 1)
        di_direction = 1 if row['plus_DI'] > row['minus_DI'] else -1
        score += 0.30 * adx_strength * di_direction
        
        # RSI偏离 (20%)
        rsi_deviation = (row['RSI'] - 50) / 50
        score += 0.20 * np.clip(rsi_deviation, -1, 1)
        
        # MACD (20%)
        macd_norm = row['MACD_hist'] / atr
        score += 0.20 * np.clip(macd_norm, -1, 1)
        
        return np.clip(score, -1, 1)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_indicators(df)
        
        n = len(df)
        signals = np.zeros(n)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)
        position_sizes = np.zeros(n)
        signal_strengths = np.zeros(n)
        
        current_position = 0
        entry_price = 0.0
        current_stop = np.nan
        current_tp = np.nan
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')
        bars_since_signal = 999
        partial_tp_hit = False  # 是否已触发部分止盈
        
        warmup = max(self.ema_trend, self.adx_period, 55)
        
        for i in range(warmup, n):
            row = df.iloc[i]
            
            # 计算信号强度
            strength = self._calculate_signal_strength(row)
            signal_strengths[i] = strength
            
            bars_since_signal += 1
            
            # ====== 条件判断（与原版一致，保持高胜率）======
            
            # 1. 强趋势 (ADX > 阈值)
            strong_trend = row['ADX'] > self.adx_threshold
            
            # 2. EMA多头/空头排列
            ema_bullish = (row['EMA_fast'] > row['EMA_slow']) and (row['EMA_slow'] > row['EMA_trend'])
            ema_bearish = (row['EMA_fast'] < row['EMA_slow']) and (row['EMA_slow'] < row['EMA_trend'])
            
            # 3. 价格在趋势正确一侧
            price_above_trend = row['close'] > row['EMA_trend']
            price_below_trend = row['close'] < row['EMA_trend']
            
            # 4. RSI不在极端区域
            rsi_ok_long = row['RSI'] < self.rsi_overbought and row['RSI'] > 35
            rsi_ok_short = row['RSI'] > self.rsi_oversold and row['RSI'] < 65
            
            # 5. MACD方向确认
            macd_bullish = row['MACD_hist'] > 0 and row['MACD'] > row['MACD_signal']
            macd_bearish = row['MACD_hist'] < 0 and row['MACD'] < row['MACD_signal']
            
            # 6. DI方向确认
            di_bullish = row['plus_DI'] > row['minus_DI']
            di_bearish = row['minus_DI'] > row['plus_DI']
            
            # 7. 成交量确认
            volume_ok = row['volume_ratio'] >= self.min_volume_ratio
            
            # ====== 做多信号 ======
            if current_position <= 0 and bars_since_signal >= self.min_bars_between_signals:
                conditions_met = [
                    strong_trend,
                    ema_bullish,
                    price_above_trend,
                    rsi_ok_long,
                    macd_bullish,
                    di_bullish,
                    volume_ok,
                    strength > 0.2
                ]
                
                if self.require_all_conditions:
                    long_condition = all(conditions_met)
                else:
                    long_condition = sum(conditions_met) >= 6
                
                if long_condition:
                    signals[i] = 1
                    entry_price = row['close']
                    
                    # 优化后的止损止盈
                    current_stop = entry_price - self.atr_stop_multiplier * row['ATR']
                    current_tp = entry_price + self.atr_take_profit_multiplier * row['ATR']
                    
                    risk = entry_price - current_stop
                    if risk > 0:
                        position_sizes[i] = min(self.risk_per_trade / (risk / entry_price), 1.0)
                    
                    highest_since_entry = row['high']
                    bars_since_signal = 0
                    current_position = 1
                    partial_tp_hit = False
            
            # ====== 做空信号 ======
            elif current_position >= 0 and bars_since_signal >= self.min_bars_between_signals:
                conditions_met = [
                    strong_trend,
                    ema_bearish,
                    price_below_trend,
                    rsi_ok_short,
                    macd_bearish,
                    di_bearish,
                    volume_ok,
                    strength < -0.2
                ]
                
                if self.require_all_conditions:
                    short_condition = all(conditions_met)
                else:
                    short_condition = sum(conditions_met) >= 6
                
                if short_condition:
                    signals[i] = -1
                    entry_price = row['close']
                    
                    current_stop = entry_price + self.atr_stop_multiplier * row['ATR']
                    current_tp = entry_price - self.atr_take_profit_multiplier * row['ATR']
                    
                    risk = current_stop - entry_price
                    if risk > 0:
                        position_sizes[i] = min(self.risk_per_trade / (risk / entry_price), 1.0)
                    
                    lowest_since_entry = row['low']
                    bars_since_signal = 0
                    current_position = -1
                    partial_tp_hit = False
            
            # ====== 智能移动止损（分级保护利润）======
            if self.use_trailing_stop and current_position != 0 and not np.isnan(current_stop):
                atr = row['ATR']
                
                if current_position == 1:  # 多头
                    highest_since_entry = max(highest_since_entry, row['high'])
                    profit_atr = (highest_since_entry - entry_price) / atr
                    
                    # 分级移动止损
                    if profit_atr >= 4.0:  # 大幅盈利
                        new_stop = highest_since_entry - 0.8 * atr
                    elif profit_atr >= 3.0:
                        new_stop = highest_since_entry - 1.0 * atr
                    elif profit_atr >= 2.0:
                        new_stop = highest_since_entry - 1.3 * atr
                    elif profit_atr >= 1.5:  # 保本止损
                        new_stop = entry_price + 0.2 * atr
                    else:
                        new_stop = current_stop
                    
                    if new_stop > current_stop:
                        current_stop = new_stop
                
                elif current_position == -1:  # 空头
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    profit_atr = (entry_price - lowest_since_entry) / atr
                    
                    if profit_atr >= 4.0:
                        new_stop = lowest_since_entry + 0.8 * atr
                    elif profit_atr >= 3.0:
                        new_stop = lowest_since_entry + 1.0 * atr
                    elif profit_atr >= 2.0:
                        new_stop = lowest_since_entry + 1.3 * atr
                    elif profit_atr >= 1.5:
                        new_stop = entry_price - 0.2 * atr
                    else:
                        new_stop = current_stop
                    
                    if new_stop < current_stop:
                        current_stop = new_stop
            
            stop_losses[i] = current_stop
            take_profits[i] = current_tp
        
        df['signal'] = signals.astype(int)
        df['stop_loss'] = stop_losses
        df['take_profit'] = take_profits
        df['position_size'] = position_sizes
        df['signal_strength'] = signal_strengths
        
        return df


class HighWinRateStrategy(ImprovedStrategy):
    """高胜率策略 - 保持原版止损，用移动止损保护利润"""
    
    def __init__(self):
        super().__init__(
            # 保持原版参数
            ema_fast=12,
            ema_slow=26,
            ema_trend=55,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            # 关键：保持原版止损，但增大止盈
            atr_stop_multiplier=2.5,  # 原版止损
            atr_take_profit_multiplier=5.0,  # 更大止盈（盈亏比2:1）
            adx_threshold=20,
            min_bars_between_signals=15,
            require_all_conditions=True,
            use_trailing_stop=True,
        )


class HighProfitRatioStrategy(ImprovedStrategy):
    """高盈亏比策略 - 让利润奔跑"""
    
    def __init__(self):
        super().__init__(
            ema_fast=12,
            ema_slow=26,
            ema_trend=55,
            # 保持原版止损，大幅增加止盈
            atr_stop_multiplier=2.5,
            atr_take_profit_multiplier=7.5,  # 盈亏比3:1
            adx_threshold=20,
            min_bars_between_signals=15,
            require_all_conditions=True,
            use_trailing_stop=True,
        )


class OptimalStrategy(ImprovedStrategy):
    """最优策略 - 保持原版止损，优化止盈和移动止损"""
    
    def __init__(self):
        super().__init__(
            ema_fast=12,
            ema_slow=26,
            ema_trend=55,
            # 保持原版止损，稍微增加止盈
            atr_stop_multiplier=2.5,  # 原版止损
            atr_take_profit_multiplier=4.0,  # 稍大止盈（盈亏比1.6:1）
            adx_threshold=20,
            min_bars_between_signals=15,
            require_all_conditions=True,
            use_trailing_stop=True,  # 关键：用移动止损保护利润
        )


class TrailingStopStrategy:
    """
    移动止损策略 - 核心优化版
    
    核心思路：
    1. 保持原版的入场逻辑（高胜率）
    2. 使用较小的固定止盈（容易触发）
    3. 激进的移动止损（保护利润）
    """
    
    def __init__(
        self,
        ema_fast: int = 12,
        ema_slow: int = 26,
        ema_trend: int = 55,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        atr_period: int = 14,
        # 关键参数优化
        atr_stop_multiplier: float = 2.5,    # 原版止损
        atr_take_profit_multiplier: float = 3.5,  # 稍大止盈
        adx_period: int = 14,
        adx_threshold: int = 20,
        risk_per_trade: float = 0.02,
        # 移动止损参数
        breakeven_trigger: float = 1.5,       # 盈利1.5x ATR移动到保本
        trailing_trigger: float = 2.0,        # 盈利2x ATR开始跟踪
        trailing_distance: float = 1.2,       # 跟踪距离1.2x ATR
        min_bars_between_signals: int = 15,
        use_volume_filter: bool = True,
        min_volume_ratio: float = 0.9,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.risk_per_trade = risk_per_trade
        self.breakeven_trigger = breakeven_trigger
        self.trailing_trigger = trailing_trigger
        self.trailing_distance = trailing_distance
        self.min_bars_between_signals = min_bars_between_signals
        self.use_volume_filter = use_volume_filter
        self.min_volume_ratio = min_volume_ratio
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # EMA
        df['EMA_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['EMA_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/self.atr_period, adjust=False).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr_smooth = df['ATR'].replace(0, np.nan)
        plus_di = 100 * plus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr_smooth
        minus_di = 100 * minus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr_smooth
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=1/self.adx_period, adjust=False).mean().fillna(0)
        df['plus_DI'] = plus_di.fillna(0)
        df['minus_DI'] = minus_di.fillna(0)
        
        # 成交量
        if self.use_volume_filter and 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        else:
            df['volume_ratio'] = 1.0
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_indicators(df)
        
        n = len(df)
        signals = np.zeros(n)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)
        position_sizes = np.zeros(n)
        
        current_position = 0
        entry_price = 0.0
        initial_stop = np.nan
        current_stop = np.nan
        current_tp = np.nan
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')
        bars_since_signal = 999
        
        warmup = max(self.ema_trend, self.adx_period, 55)
        
        for i in range(warmup, n):
            row = df.iloc[i]
            bars_since_signal += 1
            
            # ====== 入场条件（与原版一致）======
            strong_trend = row['ADX'] > self.adx_threshold
            ema_bullish = (row['EMA_fast'] > row['EMA_slow']) and (row['EMA_slow'] > row['EMA_trend'])
            ema_bearish = (row['EMA_fast'] < row['EMA_slow']) and (row['EMA_slow'] < row['EMA_trend'])
            price_above_trend = row['close'] > row['EMA_trend']
            price_below_trend = row['close'] < row['EMA_trend']
            rsi_ok_long = row['RSI'] < self.rsi_overbought and row['RSI'] > 35
            rsi_ok_short = row['RSI'] > self.rsi_oversold and row['RSI'] < 65
            macd_bullish = row['MACD_hist'] > 0 and row['MACD'] > row['MACD_signal']
            macd_bearish = row['MACD_hist'] < 0 and row['MACD'] < row['MACD_signal']
            di_bullish = row['plus_DI'] > row['minus_DI']
            di_bearish = row['minus_DI'] > row['plus_DI']
            volume_ok = row['volume_ratio'] >= self.min_volume_ratio
            
            # ====== 做多信号 ======
            if current_position <= 0 and bars_since_signal >= self.min_bars_between_signals:
                conditions = [strong_trend, ema_bullish, price_above_trend, rsi_ok_long, 
                             macd_bullish, di_bullish, volume_ok]
                if all(conditions):
                    signals[i] = 1
                    entry_price = row['close']
                    initial_stop = entry_price - self.atr_stop_multiplier * row['ATR']
                    current_stop = initial_stop
                    current_tp = entry_price + self.atr_take_profit_multiplier * row['ATR']
                    
                    risk = entry_price - current_stop
                    if risk > 0:
                        position_sizes[i] = min(self.risk_per_trade / (risk / entry_price), 1.0)
                    
                    highest_since_entry = row['high']
                    bars_since_signal = 0
                    current_position = 1
            
            # ====== 做空信号 ======
            elif current_position >= 0 and bars_since_signal >= self.min_bars_between_signals:
                conditions = [strong_trend, ema_bearish, price_below_trend, rsi_ok_short,
                             macd_bearish, di_bearish, volume_ok]
                if all(conditions):
                    signals[i] = -1
                    entry_price = row['close']
                    initial_stop = entry_price + self.atr_stop_multiplier * row['ATR']
                    current_stop = initial_stop
                    current_tp = entry_price - self.atr_take_profit_multiplier * row['ATR']
                    
                    risk = current_stop - entry_price
                    if risk > 0:
                        position_sizes[i] = min(self.risk_per_trade / (risk / entry_price), 1.0)
                    
                    lowest_since_entry = row['low']
                    bars_since_signal = 0
                    current_position = -1
            
            # ====== 移动止损逻辑 ======
            if current_position != 0 and not np.isnan(current_stop):
                atr = row['ATR']
                
                if current_position == 1:  # 多头
                    highest_since_entry = max(highest_since_entry, row['high'])
                    profit_atr = (highest_since_entry - entry_price) / atr
                    
                    # 保本止损
                    if profit_atr >= self.breakeven_trigger:
                        breakeven_stop = entry_price + 0.1 * atr  # 小幅盈利
                        if breakeven_stop > current_stop:
                            current_stop = breakeven_stop
                    
                    # 跟踪止损
                    if profit_atr >= self.trailing_trigger:
                        trailing_stop = highest_since_entry - self.trailing_distance * atr
                        if trailing_stop > current_stop:
                            current_stop = trailing_stop
                
                elif current_position == -1:  # 空头
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    profit_atr = (entry_price - lowest_since_entry) / atr
                    
                    if profit_atr >= self.breakeven_trigger:
                        breakeven_stop = entry_price - 0.1 * atr
                        if breakeven_stop < current_stop:
                            current_stop = breakeven_stop
                    
                    if profit_atr >= self.trailing_trigger:
                        trailing_stop = lowest_since_entry + self.trailing_distance * atr
                        if trailing_stop < current_stop:
                            current_stop = trailing_stop
            
            stop_losses[i] = current_stop
            take_profits[i] = current_tp
        
        df['signal'] = signals.astype(int)
        df['stop_loss'] = stop_losses
        df['take_profit'] = take_profits
        df['position_size'] = position_sizes
        
        return df

