"""
优化版交易策略 v4.0
核心优化：
1. 评分式入场系统 - 不再要求所有条件同时满足
2. 动态盈亏比 - 根据市场波动率调整
3. 智能移动止损 - 更保守的激活，更宽的跟踪
4. 市场环境识别 - 趋势/震荡自适应
5. 二次信号确认 - 减少假突破
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


class OptimizedStrategy:
    """优化版策略 v4.0"""
    
    def __init__(
        self,
        # EMA参数
        ema_fast: int = 9,      # 加快响应
        ema_slow: int = 21,     # 更贴近市场
        ema_trend: int = 50,    # 趋势线
        # RSI参数
        rsi_period: int = 14,
        rsi_long_zone: Tuple[int, int] = (40, 65),   # 做多RSI区间
        rsi_short_zone: Tuple[int, int] = (35, 60),  # 做空RSI区间
        # ATR参数
        atr_period: int = 14,
        base_stop_multiplier: float = 1.5,    # 基础止损（更紧）
        base_profit_multiplier: float = 3.0,  # 基础止盈（盈亏比2:1）
        # ADX趋势过滤
        adx_period: int = 14,
        adx_trend_threshold: int = 25,     # 趋势确认阈值
        adx_strong_trend: int = 35,        # 强趋势阈值
        # 风险管理
        risk_per_trade: float = 0.015,     # 降低单次风险
        max_position_size: float = 0.8,    # 最大仓位
        # 移动止损（更保守）
        use_trailing_stop: bool = True,
        trailing_activation_profit: float = 2.0,  # 盈利2倍ATR才激活
        trailing_distance: float = 1.5,           # 跟踪距离1.5倍ATR
        # 信号控制
        min_bars_between_signals: int = 8,        # 减少信号间隔
        min_entry_score: float = 0.55,            # 入场评分阈值
        # 市场环境
        volatility_lookback: int = 20,
        trend_lookback: int = 30,
    ):
        # 保存所有参数
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.rsi_long_zone = rsi_long_zone
        self.rsi_short_zone = rsi_short_zone
        self.atr_period = atr_period
        self.base_stop_multiplier = base_stop_multiplier
        self.base_profit_multiplier = base_profit_multiplier
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_trend = adx_strong_trend
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_profit = trailing_activation_profit
        self.trailing_distance = trailing_distance
        self.min_bars_between_signals = min_bars_between_signals
        self.min_entry_score = min_entry_score
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # ===== EMA =====
        df['EMA_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['EMA_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()
        
        # EMA斜率（趋势方向）
        df['EMA_fast_slope'] = df['EMA_fast'].diff(3) / df['EMA_fast'].shift(3) * 100
        df['EMA_slow_slope'] = df['EMA_slow'].diff(5) / df['EMA_slow'].shift(5) * 100
        
        # ===== RSI =====
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # RSI斜率（动量变化）
        df['RSI_slope'] = df['RSI'].diff(3)
        
        # ===== ATR =====
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/self.atr_period, adjust=False).mean()
        df['ATR_pct'] = df['ATR'] / df['close'] * 100
        
        # ===== ADX =====
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
        df['DI_diff'] = df['plus_DI'] - df['minus_DI']
        
        # ===== MACD =====
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_slope'] = df['MACD_hist'].diff(2)  # MACD柱状图斜率
        
        # ===== 成交量 =====
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
            # 成交量趋势
            df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        else:
            df['volume_ratio'] = 1.0
            df['volume_trend'] = 1.0
        
        # ===== 市场环境 =====
        # 波动率状态
        vol_20 = df['close'].pct_change().rolling(20).std()
        vol_60 = df['close'].pct_change().rolling(60).std()
        df['volatility_ratio'] = vol_20 / vol_60.replace(0, np.nan)
        df['volatility_regime'] = np.where(df['volatility_ratio'] > 1.2, 'high',
                                   np.where(df['volatility_ratio'] < 0.8, 'low', 'normal'))
        
        # 趋势强度（线性回归R²）
        df['trend_strength'] = self._calc_trend_strength(df, self.trend_lookback)
        
        # 市场环境分类
        df['market_regime'] = self._classify_market(df)
        
        # ===== 价格位置 =====
        # 布林带位置
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_ma + 2 * bb_std
        df['bb_lower'] = bb_ma - 2 * bb_std
        df['bb_position'] = (df['close'] - bb_ma) / (2 * bb_std)
        
        # 通道位置
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        df['channel_position'] = (df['close'] - low_20) / (high_20 - low_20).replace(0, 1)
        
        return df
    
    def _calc_trend_strength(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算趋势强度（带方向）"""
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
    
    def _classify_market(self, df: pd.DataFrame) -> pd.Series:
        """分类市场环境"""
        regime = pd.Series('ranging', index=df.index)
        
        for i in range(len(df)):
            adx = df['ADX'].iloc[i]
            trend = df['trend_strength'].iloc[i] if 'trend_strength' in df.columns else 0
            
            if adx > self.adx_strong_trend and abs(trend) > 0.6:
                regime.iloc[i] = 'strong_trend'
            elif adx > self.adx_trend_threshold and abs(trend) > 0.4:
                regime.iloc[i] = 'trend'
            else:
                regime.iloc[i] = 'ranging'
        
        return regime
    
    def calculate_entry_score(self, row: pd.Series, direction: int) -> float:
        """
        计算入场评分 (0-1)
        direction: 1=做多, -1=做空
        评分系统替代硬性条件
        """
        score = 0.0
        max_score = 0.0
        
        # ===== 趋势因子 (权重: 35%) =====
        
        # 1. ADX趋势强度 (10%)
        max_score += 0.10
        adx = row['ADX']
        if adx > self.adx_strong_trend:
            score += 0.10
        elif adx > self.adx_trend_threshold:
            score += 0.07
        elif adx > 18:
            score += 0.04
        
        # 2. EMA排列 (10%)
        max_score += 0.10
        if direction == 1:
            if row['EMA_fast'] > row['EMA_slow'] > row['EMA_trend']:
                score += 0.10  # 完美多头排列
            elif row['EMA_fast'] > row['EMA_slow']:
                score += 0.06  # 短期多头
            elif row['close'] > row['EMA_fast']:
                score += 0.03  # 价格在快线上方
        else:
            if row['EMA_fast'] < row['EMA_slow'] < row['EMA_trend']:
                score += 0.10
            elif row['EMA_fast'] < row['EMA_slow']:
                score += 0.06
            elif row['close'] < row['EMA_fast']:
                score += 0.03
        
        # 3. EMA斜率方向 (8%)
        max_score += 0.08
        fast_slope = row['EMA_fast_slope']
        if direction == 1 and fast_slope > 0:
            score += min(0.08, abs(fast_slope) * 0.04)
        elif direction == -1 and fast_slope < 0:
            score += min(0.08, abs(fast_slope) * 0.04)
        
        # 4. 趋势强度 (7%)
        max_score += 0.07
        trend = row.get('trend_strength', 0)
        if direction == 1 and trend > 0.3:
            score += min(0.07, trend * 0.10)
        elif direction == -1 and trend < -0.3:
            score += min(0.07, abs(trend) * 0.10)
        
        # ===== 动量因子 (权重: 30%) =====
        
        # 5. RSI位置 (12%)
        max_score += 0.12
        rsi = row['RSI']
        if direction == 1:
            if self.rsi_long_zone[0] <= rsi <= self.rsi_long_zone[1]:
                score += 0.12  # 最优区间
            elif 35 <= rsi <= 70:
                score += 0.06  # 可接受区间
        else:
            if self.rsi_short_zone[0] <= rsi <= self.rsi_short_zone[1]:
                score += 0.12
            elif 30 <= rsi <= 65:
                score += 0.06
        
        # 6. RSI斜率（动量方向）(8%)
        max_score += 0.08
        rsi_slope = row.get('RSI_slope', 0)
        if direction == 1 and rsi_slope > 0:
            score += min(0.08, rsi_slope * 0.02)
        elif direction == -1 and rsi_slope < 0:
            score += min(0.08, abs(rsi_slope) * 0.02)
        
        # 7. MACD方向 (10%)
        max_score += 0.10
        macd_hist = row['MACD_hist']
        macd_slope = row.get('MACD_hist_slope', 0)
        if direction == 1:
            if macd_hist > 0 and macd_slope > 0:
                score += 0.10  # MACD正且上升
            elif macd_hist > 0 or macd_slope > 0:
                score += 0.05  # 部分满足
        else:
            if macd_hist < 0 and macd_slope < 0:
                score += 0.10
            elif macd_hist < 0 or macd_slope < 0:
                score += 0.05
        
        # ===== 方向确认因子 (权重: 20%) =====
        
        # 8. DI方向 (10%)
        max_score += 0.10
        di_diff = row['DI_diff']
        if direction == 1 and di_diff > 5:
            score += min(0.10, di_diff * 0.01)
        elif direction == -1 and di_diff < -5:
            score += min(0.10, abs(di_diff) * 0.01)
        
        # 9. 价格相对EMA趋势线 (10%)
        max_score += 0.10
        if direction == 1 and row['close'] > row['EMA_trend']:
            distance_pct = (row['close'] - row['EMA_trend']) / row['EMA_trend'] * 100
            if 0 < distance_pct < 3:  # 刚突破，不要追高
                score += 0.10
            elif distance_pct < 5:
                score += 0.06
        elif direction == -1 and row['close'] < row['EMA_trend']:
            distance_pct = (row['EMA_trend'] - row['close']) / row['EMA_trend'] * 100
            if 0 < distance_pct < 3:
                score += 0.10
            elif distance_pct < 5:
                score += 0.06
        
        # ===== 成交量因子 (权重: 10%) =====
        
        # 10. 成交量确认 (10%)
        max_score += 0.10
        vol_ratio = row.get('volume_ratio', 1.0)
        vol_trend = row.get('volume_trend', 1.0)
        if 0.8 <= vol_ratio <= 2.0:  # 适中成交量
            score += 0.05
            if vol_trend > 1.0:  # 成交量放大趋势
                score += 0.05
        
        # ===== 位置因子 (权重: 5%) =====
        
        # 11. 布林带位置 (5%)
        max_score += 0.05
        bb_pos = row.get('bb_position', 0)
        if direction == 1 and -0.5 < bb_pos < 0.5:  # 不追涨
            score += 0.05
        elif direction == -1 and -0.5 < bb_pos < 0.5:  # 不追跌
            score += 0.05
        
        # 归一化得分
        final_score = score / max_score if max_score > 0 else 0
        return final_score
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_indicators(df)
        
        n = len(df)
        signals = np.zeros(n)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)
        position_sizes = np.zeros(n)
        entry_scores = np.zeros(n)
        
        current_position = 0
        entry_price = 0.0
        current_stop = np.nan
        current_tp = np.nan
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')
        bars_since_signal = 999
        
        warmup = max(self.ema_trend, self.adx_period, 60)
        
        for i in range(warmup, n):
            row = df.iloc[i]
            bars_since_signal += 1
            
            # 计算多空评分
            long_score = self.calculate_entry_score(row, 1)
            short_score = self.calculate_entry_score(row, -1)
            
            # 记录评分用于分析
            entry_scores[i] = long_score - short_score
            
            # ===== 动态参数调整 =====
            market_regime = row.get('market_regime', 'ranging')
            volatility_regime = row.get('volatility_regime', 'normal')
            
            # 根据市场环境调整止损止盈
            if market_regime == 'strong_trend':
                stop_mult = self.base_stop_multiplier * 1.2  # 趋势中放宽止损
                profit_mult = self.base_profit_multiplier * 1.5  # 趋势中扩大止盈
                score_threshold = self.min_entry_score - 0.05  # 降低入场门槛
            elif market_regime == 'ranging':
                stop_mult = self.base_stop_multiplier * 0.8  # 震荡中收紧止损
                profit_mult = self.base_profit_multiplier * 0.8  # 震荡中缩小止盈
                score_threshold = self.min_entry_score + 0.05  # 提高入场门槛
            else:
                stop_mult = self.base_stop_multiplier
                profit_mult = self.base_profit_multiplier
                score_threshold = self.min_entry_score
            
            # 波动率调整
            if volatility_regime == 'high':
                stop_mult *= 1.3
                profit_mult *= 1.2
            elif volatility_regime == 'low':
                stop_mult *= 0.9
                profit_mult *= 0.9
            
            # ===== 做多信号 =====
            if current_position <= 0 and bars_since_signal >= self.min_bars_between_signals:
                if long_score >= score_threshold and long_score > short_score + 0.05:
                    # 趋势确认：检查评分趋势
                    score_improving = True
                    if i >= 2:
                        prev_score = self.calculate_entry_score(df.iloc[i-1], 1)
                        prev2_score = self.calculate_entry_score(df.iloc[i-2], 1)
                        # 评分应该在上升或维持高位
                        score_improving = (long_score >= prev_score * 0.95) or (prev_score > prev2_score)
                    
                    if score_improving:
                        signals[i] = 1
                        entry_price = row['close']
                        current_stop = entry_price - stop_mult * row['ATR']
                        current_tp = entry_price + profit_mult * row['ATR']
                        
                        risk = entry_price - current_stop
                        if risk > 0:
                            position_sizes[i] = min(
                                self.risk_per_trade / (risk / entry_price),
                                self.max_position_size
                            )
                        
                        highest_since_entry = row['high']
                        bars_since_signal = 0
                        current_position = 1
            
            # ===== 做空信号 =====
            elif current_position >= 0 and bars_since_signal >= self.min_bars_between_signals:
                if short_score >= score_threshold and short_score > long_score + 0.05:
                    # 趋势确认
                    score_improving = True
                    if i >= 2:
                        prev_score = self.calculate_entry_score(df.iloc[i-1], -1)
                        prev2_score = self.calculate_entry_score(df.iloc[i-2], -1)
                        score_improving = (short_score >= prev_score * 0.95) or (prev_score > prev2_score)
                    
                    if score_improving:
                        signals[i] = -1
                        entry_price = row['close']
                        current_stop = entry_price + stop_mult * row['ATR']
                        current_tp = entry_price - profit_mult * row['ATR']
                        
                        risk = current_stop - entry_price
                        if risk > 0:
                            position_sizes[i] = min(
                                self.risk_per_trade / (risk / entry_price),
                                self.max_position_size
                            )
                        
                        lowest_since_entry = row['low']
                        bars_since_signal = 0
                        current_position = -1
            
            # ===== 智能移动止损 =====
            if self.use_trailing_stop and current_position != 0 and not np.isnan(current_stop):
                atr = row['ATR']
                
                if current_position == 1:
                    highest_since_entry = max(highest_since_entry, row['high'])
                    profit_atr = (highest_since_entry - entry_price) / atr
                    
                    # 分级移动止损
                    if profit_atr >= self.trailing_activation_profit * 2:
                        # 大幅盈利：更紧的止损
                        new_stop = highest_since_entry - self.trailing_distance * 0.8 * atr
                    elif profit_atr >= self.trailing_activation_profit:
                        # 正常盈利：标准止损
                        new_stop = highest_since_entry - self.trailing_distance * atr
                    else:
                        new_stop = current_stop
                    
                    if new_stop > current_stop:
                        current_stop = new_stop
                
                elif current_position == -1:
                    lowest_since_entry = min(lowest_since_entry, row['low'])
                    profit_atr = (entry_price - lowest_since_entry) / atr
                    
                    if profit_atr >= self.trailing_activation_profit * 2:
                        new_stop = lowest_since_entry + self.trailing_distance * 0.8 * atr
                    elif profit_atr >= self.trailing_activation_profit:
                        new_stop = lowest_since_entry + self.trailing_distance * atr
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
        df['entry_score'] = entry_scores
        
        return df


class ConservativeStrategy(OptimizedStrategy):
    """保守版策略 - 更高胜率，更少交易"""
    
    def __init__(self):
        super().__init__(
            ema_fast=12,
            ema_slow=26,
            ema_trend=55,
            rsi_long_zone=(42, 62),      # 最优RSI区间
            rsi_short_zone=(38, 58),
            base_stop_multiplier=1.5,    # 适中止损
            base_profit_multiplier=3.0,  # 盈亏比 2:1
            adx_trend_threshold=25,      # 趋势确认
            adx_strong_trend=35,
            min_bars_between_signals=10,
            min_entry_score=0.52,        # 平衡的入场门槛
            risk_per_trade=0.015,
            trailing_activation_profit=2.0,
            trailing_distance=1.5,
        )


class AggressiveStrategy(OptimizedStrategy):
    """激进版策略 - 更多交易机会"""
    
    def __init__(self):
        super().__init__(
            ema_fast=8,
            ema_slow=18,
            ema_trend=45,
            rsi_long_zone=(38, 68),       # 稍宽松的RSI区间
            rsi_short_zone=(32, 62),
            base_stop_multiplier=1.8,     # 适中止损
            base_profit_multiplier=3.5,   # 盈亏比接近2:1
            adx_trend_threshold=20,       # 更低ADX门槛
            adx_strong_trend=30,
            min_bars_between_signals=6,
            min_entry_score=0.45,         # 更低入场门槛
            risk_per_trade=0.02,
            trailing_activation_profit=2.2,
            trailing_distance=1.8,
        )


class BalancedStrategy(OptimizedStrategy):
    """平衡版策略 - 胜率和盈亏比的最佳平衡"""
    
    def __init__(self):
        super().__init__(
            ema_fast=10,
            ema_slow=22,
            ema_trend=50,
            rsi_long_zone=(40, 65),
            rsi_short_zone=(35, 60),
            base_stop_multiplier=1.6,
            base_profit_multiplier=3.2,   # 盈亏比 2:1
            adx_trend_threshold=22,
            adx_strong_trend=32,
            min_bars_between_signals=7,
            min_entry_score=0.48,         # 适中入场门槛
            risk_per_trade=0.018,
            trailing_activation_profit=2.0,
            trailing_distance=1.6,
        )

