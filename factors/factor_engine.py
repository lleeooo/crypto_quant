"""
因子引擎
负责因子标准化、合成、评分和信号生成
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FactorWeight:
    """因子权重配置"""
    name: str
    weight: float
    direction: int = 1  # 1: 正向因子(值越大越看多), -1: 反向因子


class FactorEngine:
    """因子引擎"""
    
    # 默认因子权重配置
    DEFAULT_WEIGHTS = {
        # 趋势因子 (40%)
        'TREND_STRENGTH': 0.12,
        'EMA_CROSS': 0.10,
        'ADX_FACTOR': 0.10,
        'CHANNEL_POS': 0.08,
        
        # 动量因子 (25%)
        'MOM_20': 0.08,
        'ROC_12': 0.05,
        'MOM_QUALITY': 0.07,
        'ACCELERATION': 0.05,
        
        # 成交量因子 (20%)
        'SMART_MONEY': 0.08,
        'OBV_SLOPE': 0.05,
        'MFI': 0.04,
        'VWAP_DEV': 0.03,
        
        # 反转因子 (15%) - 注意这些是均值回归信号
        'RSI_FACTOR': 0.05,
        'BB_FACTOR': 0.05,
        'MEAN_REV': 0.05,
    }
    
    def __init__(
        self,
        lookback_period: int = 100,
        zscore_cap: float = 3.0,
        signal_threshold: float = 0.3,
        weights: Dict[str, float] = None
    ):
        """
        初始化因子引擎
        
        Args:
            lookback_period: 标准化的回看周期
            zscore_cap: Z-Score的截断值
            signal_threshold: 信号阈值，综合因子超过此值才生成信号
            weights: 自定义因子权重
        """
        self.lookback_period = lookback_period
        self.zscore_cap = zscore_cap
        self.signal_threshold = signal_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS
    
    def normalize_factors(self, factors: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        因子标准化
        
        Args:
            factors: 原始因子DataFrame
            method: 标准化方法 ('zscore', 'rank', 'minmax')
        
        Returns:
            标准化后的因子
        """
        normalized = pd.DataFrame(index=factors.index)
        
        for col in factors.columns:
            if method == 'zscore':
                # 滚动Z-Score标准化
                mean = factors[col].rolling(self.lookback_period, min_periods=20).mean()
                std = factors[col].rolling(self.lookback_period, min_periods=20).std()
                z = (factors[col] - mean) / std.replace(0, np.nan)
                # 截断极端值
                normalized[col] = z.clip(-self.zscore_cap, self.zscore_cap) / self.zscore_cap
                
            elif method == 'rank':
                # 滚动排名（0-1）转换为（-1, 1）
                rank = factors[col].rolling(self.lookback_period, min_periods=20).apply(
                    lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
                )
                normalized[col] = 2 * rank - 1
                
            elif method == 'minmax':
                # 滚动MinMax标准化
                roll_min = factors[col].rolling(self.lookback_period, min_periods=20).min()
                roll_max = factors[col].rolling(self.lookback_period, min_periods=20).max()
                normalized[col] = 2 * (factors[col] - roll_min) / (roll_max - roll_min) - 1
        
        return normalized.fillna(0)
    
    def composite_factor(self, normalized_factors: pd.DataFrame) -> pd.Series:
        """
        合成综合因子
        加权平均所有因子
        
        Returns:
            综合因子得分 (-1 到 1)
        """
        composite = pd.Series(0.0, index=normalized_factors.index)
        total_weight = 0.0
        
        for factor_name, weight in self.weights.items():
            if factor_name in normalized_factors.columns:
                composite += normalized_factors[factor_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            composite = composite / total_weight
        
        return composite.clip(-1, 1)
    
    def generate_signals(
        self, 
        composite_factor: pd.Series,
        min_holding_period: int = 10
    ) -> pd.Series:
        """
        基于综合因子生成交易信号
        
        Args:
            composite_factor: 综合因子得分
            min_holding_period: 最小持仓周期
        
        Returns:
            信号序列 (1: 多, -1: 空, 0: 无信号)
        """
        signals = pd.Series(0, index=composite_factor.index)
        
        current_position = 0
        bars_since_signal = 999
        
        for i in range(len(composite_factor)):
            bars_since_signal += 1
            score = composite_factor.iloc[i]
            
            if bars_since_signal < min_holding_period:
                continue
            
            # 做多信号
            if score > self.signal_threshold and current_position <= 0:
                signals.iloc[i] = 1
                current_position = 1
                bars_since_signal = 0
            
            # 做空信号
            elif score < -self.signal_threshold and current_position >= 0:
                signals.iloc[i] = -1
                current_position = -1
                bars_since_signal = 0
        
        return signals
    
    def factor_analysis(self, factors: pd.DataFrame, returns: pd.Series) -> Dict:
        """
        因子分析：计算每个因子的IC和收益相关性
        
        Args:
            factors: 因子DataFrame
            returns: 未来N期收益率
        
        Returns:
            因子分析报告
        """
        analysis = {}
        
        for col in factors.columns:
            factor = factors[col].dropna()
            ret = returns.reindex(factor.index).dropna()
            
            # 对齐
            common_idx = factor.index.intersection(ret.index)
            if len(common_idx) < 50:
                continue
            
            f = factor.loc[common_idx]
            r = ret.loc[common_idx]
            
            # IC (Information Coefficient) - 因子与未来收益的相关性
            ic = f.corr(r)
            
            # IC均值和标准差
            rolling_ic = f.rolling(20).corr(r)
            ic_mean = rolling_ic.mean()
            ic_std = rolling_ic.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0  # IC信息比率
            
            # 因子收益率（按因子分组）
            quantiles = pd.qcut(f, 5, labels=False, duplicates='drop')
            group_returns = r.groupby(quantiles).mean()
            
            analysis[col] = {
                'IC': ic,
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'ICIR': icir,
                'top_quantile_return': group_returns.iloc[-1] if len(group_returns) > 0 else 0,
                'bottom_quantile_return': group_returns.iloc[0] if len(group_returns) > 0 else 0,
                'spread': group_returns.iloc[-1] - group_returns.iloc[0] if len(group_returns) > 1 else 0
            }
        
        return analysis
    
    def print_factor_report(self, analysis: Dict):
        """打印因子分析报告"""
        print("\n" + "=" * 70)
        print("因子分析报告")
        print("=" * 70)
        print(f"{'因子名称':<20} {'IC':>8} {'ICIR':>8} {'多空收益差':>12}")
        print("-" * 70)
        
        # 按ICIR排序
        sorted_factors = sorted(analysis.items(), key=lambda x: abs(x[1].get('ICIR', 0)), reverse=True)
        
        for name, metrics in sorted_factors:
            ic = metrics.get('IC', 0)
            icir = metrics.get('ICIR', 0)
            spread = metrics.get('spread', 0)
            print(f"{name:<20} {ic:>8.4f} {icir:>8.4f} {spread*100:>11.2f}%")
        
        print("=" * 70)
    
    def get_factor_exposure(self, normalized_factors: pd.DataFrame) -> pd.DataFrame:
        """
        获取当前因子暴露
        返回最近一期各因子的值
        """
        latest = normalized_factors.iloc[-1].to_frame('exposure')
        latest['weight'] = latest.index.map(lambda x: self.weights.get(x, 0))
        latest['contribution'] = latest['exposure'] * latest['weight']
        return latest.sort_values('contribution', ascending=False)


class FactorStrategy:
    """
    基于因子的交易策略
    整合因子计算和信号生成
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        signal_threshold: float = 0.25,
        min_holding_period: int = 12,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 4.0,
        risk_per_trade: float = 0.02,
        weights: Dict[str, float] = None
    ):
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
        self.min_holding_period = min_holding_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.risk_per_trade = risk_per_trade
        self.weights = weights
        
        self.engine = FactorEngine(
            lookback_period=lookback_period,
            signal_threshold=signal_threshold,
            weights=weights
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: OHLCV数据
        
        Returns:
            带有信号的DataFrame
        """
        from factors.factor_lib import FactorCalculator
        
        df = df.copy()
        
        # 1. 计算所有因子
        calc = FactorCalculator(df)
        factors = calc.calculate_all()
        
        # 2. 标准化因子
        normalized = self.engine.normalize_factors(factors)
        
        # 3. 合成综合因子
        composite = self.engine.composite_factor(normalized)
        df['factor_score'] = composite
        
        # 4. 生成信号
        signals = self.engine.generate_signals(composite, self.min_holding_period)
        df['signal'] = signals
        
        # 5. 计算止损止盈
        atr = self._calc_atr(df, 14)
        
        stop_losses = []
        take_profits = []
        position_sizes = []
        
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:
                sl = df['close'].iloc[i] - self.stop_loss_atr * atr.iloc[i]
                tp = df['close'].iloc[i] + self.take_profit_atr * atr.iloc[i]
                risk = df['close'].iloc[i] - sl
                ps = min(self.risk_per_trade / (risk / df['close'].iloc[i]), 1.0) if risk > 0 else 0
            elif df['signal'].iloc[i] == -1:
                sl = df['close'].iloc[i] + self.stop_loss_atr * atr.iloc[i]
                tp = df['close'].iloc[i] - self.take_profit_atr * atr.iloc[i]
                risk = sl - df['close'].iloc[i]
                ps = min(self.risk_per_trade / (risk / df['close'].iloc[i]), 1.0) if risk > 0 else 0
            else:
                sl = np.nan
                tp = np.nan
                ps = 0
            
            stop_losses.append(sl)
            take_profits.append(tp)
            position_sizes.append(ps)
        
        df['stop_loss'] = stop_losses
        df['take_profit'] = take_profits
        df['position_size'] = position_sizes
        
        # 添加因子数据供分析
        for col in normalized.columns:
            df[f'F_{col}'] = normalized[col]
        
        return df
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

