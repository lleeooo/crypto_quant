"""
增强版 EMA + RSI + ATR 多因子策略 v3.0
基于盈利交易分析优化 + 黑天鹅事件防护

核心改进：
1. 基于盈利交易分析的最优因子过滤
2. 黑天鹅事件检测和防护
3. 动态风险调整
4. 更严格的入场条件
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.black_swan_protection import BlackSwanProtection


class EnhancedEmaRsiStrategy:
    """增强版策略 v3.0"""
    
    def __init__(
        self,
        # EMA参数
        ema_fast: int = 12,
        ema_slow: int = 26,
        ema_trend: int = 55,
        # RSI参数（基于盈利交易分析优化）
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        rsi_optimal_range: tuple = None,  # 盈利交易的最优RSI区间
        # ATR参数
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,
        atr_take_profit_multiplier: float = 3.0,
        # ADX趋势过滤（基于盈利交易分析优化）
        adx_period: int = 14,
        adx_threshold: int = 20,
        adx_optimal_range: tuple = None,  # 盈利交易的最优ADX区间
        # 风险管理
        risk_per_trade: float = 0.02,
        use_trailing_stop: bool = True,
        trailing_stop_activation: float = 1.5,
        trailing_stop_distance: float = 1.0,
        # 过滤器
        use_volume_filter: bool = True,
        volume_ma_period: int = 20,
        min_volume_ratio: float = 0.9,
        volume_optimal_range: tuple = None,  # 盈利交易的最优成交量区间
        # 信号控制
        min_bars_between_signals: int = 15,
        require_all_conditions: bool = True,
        # 黑天鹅防护
        enable_black_swan_protection: bool = True,
        # 基于盈利交易分析的过滤条件
        optimal_filters: dict = None
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.rsi_optimal_range = rsi_optimal_range
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_optimal_range = adx_optimal_range
        self.risk_per_trade = risk_per_trade
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        self.use_volume_filter = use_volume_filter
        self.volume_ma_period = volume_ma_period
        self.min_volume_ratio = min_volume_ratio
        self.volume_optimal_range = volume_optimal_range
        self.min_bars_between_signals = min_bars_between_signals
        self.require_all_conditions = require_all_conditions
        self.enable_black_swan_protection = enable_black_swan_protection
        self.optimal_filters = optimal_filters or {}
        
        # 初始化黑天鹅防护
        if self.enable_black_swan_protection:
            self.black_swan = BlackSwanProtection()
        else:
            self.black_swan = None
    
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
        df['ATR_PCT'] = df['ATR'] / df['close'] * 100
        
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
        
        # 波动率
        returns = df['close'].pct_change()
        df['VOLATILITY_10'] = returns.rolling(10).std() * 100
        df['VOLATILITY_20'] = returns.rolling(20).std() * 100
        
        # 趋势强度
        df['TREND_STRENGTH'] = self._calc_trend_strength(df, 20)
        
        return df
    
    def _calc_trend_strength(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算趋势强度"""
        trend_strength = pd.Series(0.0, index=df.index)
        
        for i in range(period, len(df)):
            prices = df['close'].iloc[i-period:i].values
            x = np.arange(len(prices))
            if len(prices) < 2:
                continue
            try:
                slope, intercept = np.polyfit(x, prices, 1)
                fitted = slope * x + intercept
                ss_res = np.sum((prices - fitted) ** 2)
                ss_tot = np.sum((prices - np.mean(prices)) ** 2)
                if ss_tot == 0:
                    continue
                r2 = 1 - (ss_res / ss_tot)
                trend_strength.iloc[i] = r2 * np.sign(slope)
            except:
                continue
        
        return trend_strength
    
    def _apply_optimal_filters(self, row: pd.Series) -> bool:
        """应用基于盈利交易分析的最优过滤条件"""
        if not self.optimal_filters:
            return True
        
        # RSI过滤
        if 'RSI' in self.optimal_filters and 'RSI' in row.index:
            rsi_filter = self.optimal_filters['RSI']
            rsi = row['RSI']
            if rsi < rsi_filter['min'] or rsi > rsi_filter['max']:
                return False
        
        # ADX过滤
        if 'ADX' in self.optimal_filters and 'ADX' in row.index:
            adx_filter = self.optimal_filters['ADX']
            adx = row['ADX']
            if adx < adx_filter['min'] or adx > adx_filter['max']:
                return False
        
        # 成交量过滤
        if 'VOLUME_RATIO' in self.optimal_filters and 'volume_ratio' in row.index:
            vol_filter = self.optimal_filters['VOLUME_RATIO']
            vol_ratio = row['volume_ratio']
            if vol_ratio < vol_filter['min'] or vol_ratio > vol_filter['max']:
                return False
        
        # ATR百分比过滤
        if 'ATR_PCT' in self.optimal_filters and 'ATR_PCT' in row.index:
            atr_filter = self.optimal_filters['ATR_PCT']
            atr_pct = row['ATR_PCT']
            if atr_pct < atr_filter['min'] or atr_pct > atr_filter['max']:
                return False
        
        # 波动率过滤
        if 'VOLATILITY_10' in self.optimal_filters and 'VOLATILITY_10' in row.index:
            vol_filter = self.optimal_filters['VOLATILITY_10']
            volatility = row['VOLATILITY_10']
            if volatility < vol_filter['min'] or volatility > vol_filter['max']:
                return False
        
        # 趋势强度过滤
        if 'TREND_STRENGTH' in self.optimal_filters and 'TREND_STRENGTH' in row.index:
            trend_filter = self.optimal_filters['TREND_STRENGTH']
            trend = row['TREND_STRENGTH']
            if trend < trend_filter['min'] or trend > trend_filter['max']:
                return False
        
        return True
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号（增强版）"""
        df = self.calculate_indicators(df)
        
        n = len(df)
        signals = np.zeros(n)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)
        position_sizes = np.zeros(n)
        signal_strengths = np.zeros(n)
        black_swan_flags = np.zeros(n)  # 黑天鹅事件标记
        
        current_position = 0
        entry_price = 0.0
        current_stop = np.nan
        current_tp = np.nan
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')
        bars_since_signal = 999
        
        warmup = max(self.ema_trend, self.adx_period, 55)
        
        for i in range(warmup, n):
            row = df.iloc[i]
            
            # 黑天鹅事件检测
            black_swan_detected = False
            risk_level = 'normal'
            if self.black_swan:
                bs_result = self.black_swan.detect_black_swan(df, i)
                black_swan_detected = bs_result['detected']
                risk_level = bs_result['risk_level']
                if black_swan_detected:
                    black_swan_flags[i] = 1
            
            # 计算信号强度
            strength = self._calculate_signal_strength(row)
            signal_strengths[i] = strength
            
            bars_since_signal += 1
            
            # ====== 条件判断 ======
            
            # 1. 强趋势 (ADX > 阈值)
            strong_trend = row['ADX'] > self.adx_threshold
            
            # 2. EMA多头/空头排列
            ema_bullish = (row['EMA_fast'] > row['EMA_slow']) and (row['EMA_slow'] > row['EMA_trend'])
            ema_bearish = (row['EMA_fast'] < row['EMA_slow']) and (row['EMA_slow'] < row['EMA_trend'])
            
            # 3. 价格在趋势正确一侧
            price_above_trend = row['close'] > row['EMA_trend']
            price_below_trend = row['close'] < row['EMA_trend']
            
            # 4. RSI不在极端区域（基于盈利交易分析优化）
            if self.rsi_optimal_range:
                rsi_min, rsi_max = self.rsi_optimal_range
                rsi_ok_long = row['RSI'] < rsi_max and row['RSI'] > rsi_min
                rsi_ok_short = row['RSI'] > (100 - rsi_max) and row['RSI'] < (100 - rsi_min)
            else:
                rsi_ok_long = row['RSI'] < self.rsi_overbought and row['RSI'] > 35
                rsi_ok_short = row['RSI'] > self.rsi_oversold and row['RSI'] < 65
            
            # 5. MACD方向确认
            macd_bullish = row['MACD_hist'] > 0 and row['MACD'] > row['MACD_signal']
            macd_bearish = row['MACD_hist'] < 0 and row['MACD'] < row['MACD_signal']
            
            # 6. DI方向确认
            di_bullish = row['plus_DI'] > row['minus_DI']
            di_bearish = row['minus_DI'] > row['plus_DI']
            
            # 7. 成交量确认（基于盈利交易分析优化）
            if self.volume_optimal_range:
                vol_min, vol_max = self.volume_optimal_range
                volume_ok = vol_min <= row['volume_ratio'] <= vol_max
            else:
                volume_ok = row['volume_ratio'] >= self.min_volume_ratio
            
            # 8. 应用最优过滤条件
            optimal_filter_ok = self._apply_optimal_filters(row)
            
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
                    strength > 0.2,
                    optimal_filter_ok,
                    not black_swan_detected  # 黑天鹅事件时不开仓
                ]
                
                if self.require_all_conditions:
                    long_condition = all(conditions_met)
                else:
                    long_condition = sum(conditions_met) >= 7  # 至少7个条件满足
                
                if long_condition:
                    signals[i] = 1
                    entry_price = row['close']
                    
                    # 根据风险等级调整止损
                    if self.black_swan and risk_level != 'normal':
                        emergency_sl = self.black_swan.calculate_emergency_stop_loss(
                            entry_price, 1, 
                            entry_price - self.atr_stop_multiplier * row['ATR'],
                            row['ATR'], risk_level
                        )
                        current_stop = emergency_sl
                    else:
                        current_stop = entry_price - self.atr_stop_multiplier * row['ATR']
                    
                    current_tp = entry_price + self.atr_take_profit_multiplier * row['ATR']
                    
                    risk = entry_price - current_stop
                    if risk > 0:
                        base_position_size = min(self.risk_per_trade / (risk / entry_price), 1.0)
                        # 根据风险等级调整仓位
                        if self.black_swan:
                            position_sizes[i] = self.black_swan.get_risk_adjusted_position_size(
                                base_position_size, risk_level
                            )
                        else:
                            position_sizes[i] = base_position_size
                    
                    highest_since_entry = row['high']
                    bars_since_signal = 0
                    current_position = 1
            
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
                    strength < -0.2,
                    optimal_filter_ok,
                    not black_swan_detected  # 黑天鹅事件时不开仓
                ]
                
                if self.require_all_conditions:
                    short_condition = all(conditions_met)
                else:
                    short_condition = sum(conditions_met) >= 7
                
                if short_condition:
                    signals[i] = -1
                    entry_price = row['close']
                    
                    # 根据风险等级调整止损
                    if self.black_swan and risk_level != 'normal':
                        emergency_sl = self.black_swan.calculate_emergency_stop_loss(
                            entry_price, -1,
                            entry_price + self.atr_stop_multiplier * row['ATR'],
                            row['ATR'], risk_level
                        )
                        current_stop = emergency_sl
                    else:
                        current_stop = entry_price + self.atr_stop_multiplier * row['ATR']
                    
                    current_tp = entry_price - self.atr_take_profit_multiplier * row['ATR']
                    
                    risk = current_stop - entry_price
                    if risk > 0:
                        base_position_size = min(self.risk_per_trade / (risk / entry_price), 1.0)
                        # 根据风险等级调整仓位
                        if self.black_swan:
                            position_sizes[i] = self.black_swan.get_risk_adjusted_position_size(
                                base_position_size, risk_level
                            )
                        else:
                            position_sizes[i] = base_position_size
                    
                    lowest_since_entry = row['low']
                    bars_since_signal = 0
                    current_position = -1
            
            # ====== 移动止损 ======
            if self.use_trailing_stop and current_position != 0 and not np.isnan(current_stop):
                if current_position == 1:
                    highest_since_entry = max(highest_since_entry, row['high'])
                    profit_atr = (highest_since_entry - entry_price) / row['ATR']
                    
                    if profit_atr >= self.trailing_stop_activation:
                        new_stop = highest_since_entry - self.trailing_stop_distance * row['ATR']
                        if new_stop > current_stop:
                            current_stop = new_stop
                
                elif current_position == -1:
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    profit_atr = (entry_price - lowest_since_entry) / row['ATR']
                    
                    if profit_atr >= self.trailing_stop_activation:
                        new_stop = lowest_since_entry + self.trailing_stop_distance * row['ATR']
                        if new_stop < current_stop:
                            current_stop = new_stop
            
            stop_losses[i] = current_stop
            take_profits[i] = current_tp
        
        df['signal'] = signals.astype(int)
        df['stop_loss'] = stop_losses
        df['take_profit'] = take_profits
        df['position_size'] = position_sizes
        df['signal_strength'] = signal_strengths
        df['black_swan_flag'] = black_swan_flags
        
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

