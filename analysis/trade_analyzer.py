"""
盈利交易分析工具
分析盈利交易的特征，提取有效因子和策略模式
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from backtester.backtest import Trade, BacktestResult


@dataclass
class TradeFeature:
    """交易特征"""
    # 入场时特征
    entry_rsi: float
    entry_adx: float
    entry_atr_pct: float
    entry_volume_ratio: float
    entry_ema_fast_slow_diff: float
    entry_macd_hist: float
    entry_signal_strength: float
    
    # 市场环境
    volatility_10: float
    volatility_20: float
    trend_strength: float
    
    # 交易结果
    pnl: float
    pnl_pct: float
    holding_period: int
    exit_reason: str


class TradeAnalyzer:
    """交易分析器"""
    
    def __init__(self):
        self.winning_features = []
        self.losing_features = []
    
    def analyze_trades(
        self, 
        trades: List[Trade], 
        df: pd.DataFrame
    ) -> Dict:
        """
        分析交易特征
        
        Args:
            trades: 交易列表
            df: 包含指标数据的DataFrame
        
        Returns:
            分析结果字典
        """
        winning_features = []
        losing_features = []
        
        for trade in trades:
            # 找到入场和出场时间对应的索引
            entry_idx = df.index.get_loc(trade.entry_time) if trade.entry_time in df.index else None
            exit_idx = df.index.get_loc(trade.exit_time) if trade.exit_time in df.index else None
            
            if entry_idx is None:
                continue
            
            entry_row = df.iloc[entry_idx]
            
            # 提取入场时特征
            feature = {
                'entry_rsi': entry_row.get('RSI', 50),
                'entry_adx': entry_row.get('ADX', 0),
                'entry_atr_pct': (entry_row.get('ATR', 0) / entry_row['close'] * 100) if entry_row['close'] > 0 else 0,
                'entry_volume_ratio': entry_row.get('volume_ratio', 1.0),
                'entry_ema_fast_slow_diff': (entry_row.get('EMA_fast', 0) - entry_row.get('EMA_slow', 0)) / entry_row['close'] * 100 if entry_row['close'] > 0 else 0,
                'entry_macd_hist': entry_row.get('MACD_hist', 0),
                'entry_signal_strength': entry_row.get('signal_strength', 0),
                'volatility_10': self._calc_volatility(df, entry_idx, 10),
                'volatility_20': self._calc_volatility(df, entry_idx, 20),
                'trend_strength': self._calc_trend_strength(df, entry_idx, 20),
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'holding_period': (exit_idx - entry_idx) if exit_idx is not None and entry_idx is not None else 0,
                'exit_reason': trade.exit_reason,
                'direction': trade.direction
            }
            
            if trade.pnl > 0:
                winning_features.append(feature)
            else:
                losing_features.append(feature)
        
        self.winning_features = winning_features
        self.losing_features = losing_features
        
        # 统计分析
        analysis = self._statistical_analysis(winning_features, losing_features)
        
        return analysis
    
    def _calc_volatility(self, df: pd.DataFrame, idx: int, period: int) -> float:
        """计算波动率"""
        if idx < period:
            return 0
        returns = df['close'].iloc[idx-period:idx].pct_change().dropna()
        return returns.std() * 100 if len(returns) > 0 else 0
    
    def _calc_trend_strength(self, df: pd.DataFrame, idx: int, period: int) -> float:
        """计算趋势强度（线性回归R²）"""
        if idx < period:
            return 0
        prices = df['close'].iloc[idx-period:idx].values
        x = np.arange(len(prices))
        if len(prices) < 2:
            return 0
        try:
            slope, intercept = np.polyfit(x, prices, 1)
            fitted = slope * x + intercept
            ss_res = np.sum((prices - fitted) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            if ss_tot == 0:
                return 0
            r2 = 1 - (ss_res / ss_tot)
            return r2 * np.sign(slope)  # 带方向的趋势强度
        except:
            return 0
    
    def _statistical_analysis(
        self, 
        winning_features: List[Dict], 
        losing_features: List[Dict]
    ) -> Dict:
        """统计分析"""
        if not winning_features:
            return {}
        
        win_df = pd.DataFrame(winning_features)
        lose_df = pd.DataFrame(losing_features) if losing_features else pd.DataFrame()
        
        analysis = {}
        
        # 关键特征列表
        key_features = [
            'entry_rsi', 'entry_adx', 'entry_atr_pct', 'entry_volume_ratio',
            'entry_ema_fast_slow_diff', 'entry_macd_hist', 'entry_signal_strength',
            'volatility_10', 'volatility_20', 'trend_strength'
        ]
        
        for feature in key_features:
            if feature not in win_df.columns:
                continue
            
            win_values = win_df[feature].dropna()
            lose_values = lose_df[feature].dropna() if not lose_df.empty else pd.Series()
            
            if len(win_values) == 0:
                continue
            
            win_mean = win_values.mean()
            win_std = win_values.std()
            win_median = win_values.median()
            win_q25 = win_values.quantile(0.25)
            win_q75 = win_values.quantile(0.75)
            
            lose_mean = lose_values.mean() if len(lose_values) > 0 else 0
            lose_std = lose_values.std() if len(lose_values) > 0 else 0
            
            # 计算差异显著性
            diff = win_mean - lose_mean if len(lose_values) > 0 else win_mean
            diff_pct = (diff / abs(lose_mean)) * 100 if len(lose_values) > 0 and lose_mean != 0 else 0
            
            analysis[feature] = {
                'win_mean': win_mean,
                'win_std': win_std,
                'win_median': win_median,
                'win_q25': win_q25,
                'win_q75': win_q75,
                'lose_mean': lose_mean,
                'lose_std': lose_std,
                'diff': diff,
                'diff_pct': diff_pct,
                'optimal_range': (win_q25, win_q75)  # 最优区间
            }
        
        # 持仓周期分析
        if 'holding_period' in win_df.columns:
            win_holding = win_df['holding_period'].dropna()
            analysis['holding_period'] = {
                'win_mean': win_holding.mean(),
                'win_median': win_holding.median(),
                'optimal_range': (win_holding.quantile(0.25), win_holding.quantile(0.75))
            }
        
        # 退出原因分析
        if 'exit_reason' in win_df.columns:
            exit_reasons = win_df['exit_reason'].value_counts()
            analysis['exit_reasons'] = exit_reasons.to_dict()
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict):
        """打印分析报告"""
        print("\n" + "=" * 80)
        print("[盈利交易特征分析报告]")
        print("=" * 80)
        
        print(f"\n[统计概览]")
        print(f"   盈利交易数: {len(self.winning_features)}")
        print(f"   亏损交易数: {len(self.losing_features)}")
        
        print(f"\n[关键因子差异分析]")
        print(f"{'因子名称':<25} {'盈利均值':>12} {'亏损均值':>12} {'差异':>12} {'差异%':>10}")
        print("-" * 80)
        
        key_features = [
            'entry_rsi', 'entry_adx', 'entry_atr_pct', 'entry_volume_ratio',
            'entry_ema_fast_slow_diff', 'entry_macd_hist', 'entry_signal_strength',
            'volatility_10', 'trend_strength'
        ]
        
        for feature in key_features:
            if feature not in analysis:
                continue
            
            data = analysis[feature]
            win_mean = data['win_mean']
            lose_mean = data['lose_mean']
            diff = data['diff']
            diff_pct = data['diff_pct']
            
            feature_name = feature.replace('entry_', '').replace('_', ' ').upper()
            print(f"{feature_name:<25} {win_mean:>12.4f} {lose_mean:>12.4f} {diff:>12.4f} {diff_pct:>9.2f}%")
        
        print(f"\n[最优因子区间] (基于盈利交易的25%-75%分位数)")
        print("-" * 80)
        for feature in key_features:
            if feature not in analysis:
                continue
            data = analysis[feature]
            if 'optimal_range' in data:
                q25, q75 = data['optimal_range']
                feature_name = feature.replace('entry_', '').replace('_', ' ').upper()
                print(f"{feature_name:<25} [{q25:>8.4f}, {q75:>8.4f}]")
        
        if 'holding_period' in analysis:
            hp = analysis['holding_period']
            print(f"\n[持仓周期分析]")
            print(f"   平均持仓: {hp['win_mean']:.1f} 根K线")
            print(f"   中位数持仓: {hp['win_median']:.1f} 根K线")
            print(f"   最优区间: [{hp['optimal_range'][0]:.1f}, {hp['optimal_range'][1]:.1f}] 根K线")
        
        if 'exit_reasons' in analysis:
            print(f"\n[退出原因统计]")
            for reason, count in analysis['exit_reasons'].items():
                pct = count / len(self.winning_features) * 100
                print(f"   {reason}: {count} ({pct:.1f}%)")
        
        print("\n" + "=" * 80)
    
    def extract_optimal_filters(self, analysis: Dict) -> Dict:
        """
        提取最优过滤条件
        
        Returns:
            过滤条件字典
        """
        filters = {}
        
        key_features = {
            'entry_rsi': 'RSI',
            'entry_adx': 'ADX',
            'entry_atr_pct': 'ATR_PCT',
            'entry_volume_ratio': 'VOLUME_RATIO',
            'entry_ema_fast_slow_diff': 'EMA_DIFF',
            'entry_macd_hist': 'MACD_HIST',
            'entry_signal_strength': 'SIGNAL_STRENGTH',
            'volatility_10': 'VOLATILITY_10',
            'trend_strength': 'TREND_STRENGTH'
        }
        
        for feature, filter_name in key_features.items():
            if feature in analysis and 'optimal_range' in analysis[feature]:
                q25, q75 = analysis[feature]['optimal_range']
                filters[filter_name] = {
                    'min': q25,
                    'max': q75,
                    'mean': analysis[feature]['win_mean']
                }
        
        return filters

