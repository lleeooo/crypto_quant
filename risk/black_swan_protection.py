"""
黑天鹅事件防护模块
检测极端市场波动，提供紧急止损和风险控制
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class BlackSwanProtection:
    """黑天鹅事件防护"""
    
    def __init__(
        self,
        volatility_threshold: float = 3.0,  # 波动率阈值（标准差倍数）
        price_change_threshold: float = 0.10,  # 单根K线价格变化阈值（10%）
        volume_spike_threshold: float = 5.0,  # 成交量异常倍数
        max_drawdown_limit: float = 0.20,  # 最大回撤限制（20%）
        consecutive_loss_limit: int = 5,  # 连续亏损次数限制
        emergency_stop_loss_multiplier: float = 1.5  # 紧急止损倍数（相对于正常止损）
    ):
        self.volatility_threshold = volatility_threshold
        self.price_change_threshold = price_change_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.max_drawdown_limit = max_drawdown_limit
        self.consecutive_loss_limit = consecutive_loss_limit
        self.emergency_stop_loss_multiplier = emergency_stop_loss_multiplier
        
        # 状态跟踪
        self.consecutive_losses = 0
        self.peak_equity = None
        self.risk_level = 'normal'  # 'normal', 'high', 'extreme'
    
    def detect_black_swan(
        self, 
        df: pd.DataFrame, 
        current_idx: int,
        lookback: int = 20
    ) -> Dict[str, bool]:
        """
        检测黑天鹅事件
        
        Returns:
            检测结果字典
        """
        if current_idx < lookback:
            return {
                'detected': False,
                'reasons': [],
                'risk_level': 'normal'
            }
        
        row = df.iloc[current_idx]
        recent_df = df.iloc[max(0, current_idx - lookback):current_idx + 1]
        
        alerts = []
        risk_level = 'normal'
        
        # 1. 极端价格波动检测
        price_change = 0
        if current_idx > 0:
            prev_close = df.iloc[current_idx - 1]['close']
            if prev_close > 0:
                price_change = abs((row['close'] - prev_close) / prev_close)
        
        if price_change > self.price_change_threshold:
            alerts.append(f'极端价格波动: {price_change*100:.2f}%')
            risk_level = 'extreme'
        
        # 2. 波动率异常检测
        returns = recent_df['close'].pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std()
            mean_vol = returns.abs().mean()
            z_score = (vol - mean_vol) / (mean_vol + 1e-8)
            
            if z_score > self.volatility_threshold:
                alerts.append(f'波动率异常: Z-score={z_score:.2f}')
                risk_level = 'high' if risk_level == 'normal' else 'extreme'
        
        # 3. 成交量异常检测
        if 'volume' in row.index and len(recent_df) > 0:
            avg_volume = recent_df['volume'].iloc[:-1].mean()
            current_volume = row['volume']
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio > self.volume_spike_threshold:
                    alerts.append(f'成交量异常: {volume_ratio:.1f}x')
                    risk_level = 'high' if risk_level == 'normal' else 'extreme'
        
        # 4. 连续大幅下跌检测（恐慌性抛售）
        if len(recent_df) >= 3:
            recent_returns = recent_df['close'].pct_change().dropna()
            if len(recent_returns) >= 3:
                consecutive_drops = (recent_returns < -0.03).sum()  # 连续3%以上下跌
                if consecutive_drops >= 3:
                    alerts.append(f'连续大幅下跌: {consecutive_drops}次')
                    risk_level = 'extreme'
        
        # 5. 流动性危机检测（价差扩大）
        if 'high' in row.index and 'low' in row.index:
            price_range = (row['high'] - row['low']) / row['close']
            avg_range = (recent_df['high'] - recent_df['low']).abs() / recent_df['close']
            avg_range_mean = avg_range.mean()
            
            if avg_range_mean > 0:
                range_ratio = price_range / avg_range_mean
                if range_ratio > 3.0:
                    alerts.append(f'价差异常扩大: {range_ratio:.1f}x')
                    risk_level = 'high' if risk_level == 'normal' else 'extreme'
        
        detected = len(alerts) > 0
        
        return {
            'detected': detected,
            'reasons': alerts,
            'risk_level': risk_level
        }
    
    def calculate_emergency_stop_loss(
        self,
        entry_price: float,
        direction: int,  # 1: 多, -1: 空
        normal_stop_loss: float,
        atr: float,
        risk_level: str
    ) -> float:
        """
        计算紧急止损价格
        
        Args:
            entry_price: 入场价格
            direction: 方向
            normal_stop_loss: 正常止损价格
            atr: ATR值
            risk_level: 风险等级
        
        Returns:
            紧急止损价格
        """
        if risk_level == 'extreme':
            # 极端风险：更紧的止损
            multiplier = 0.8  # 比正常止损更紧
        elif risk_level == 'high':
            multiplier = 1.0  # 与正常止损相同
        else:
            return normal_stop_loss
        
        if direction == 1:  # 多头
            emergency_sl = entry_price - (entry_price - normal_stop_loss) * multiplier
        else:  # 空头
            emergency_sl = entry_price + (normal_stop_loss - entry_price) * multiplier
        
        return emergency_sl
    
    def should_reduce_position(
        self,
        current_equity: float,
        initial_equity: float,
        consecutive_losses: int
    ) -> Tuple[bool, float]:
        """
        判断是否应该减仓
        
        Returns:
            (是否减仓, 建议仓位比例)
        """
        # 检查回撤
        drawdown = (initial_equity - current_equity) / initial_equity if initial_equity > 0 else 0
        
        if drawdown > self.max_drawdown_limit:
            # 回撤超过限制，建议减仓50%
            return True, 0.5
        
        # 检查连续亏损
        if consecutive_losses >= self.consecutive_loss_limit:
            # 连续亏损过多，建议减仓70%
            return True, 0.3
        
        return False, 1.0
    
    def update_risk_state(
        self,
        trade_pnl: float,
        current_equity: float,
        initial_equity: float
    ):
        """更新风险状态"""
        # 更新连续亏损计数
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # 更新峰值权益
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # 计算当前回撤
        if self.peak_equity is not None and self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown_limit * 0.8:  # 接近限制时提高风险等级
                self.risk_level = 'high'
            elif drawdown > self.max_drawdown_limit:
                self.risk_level = 'extreme'
    
    def should_pause_trading(
        self,
        current_equity: float,
        initial_equity: float
    ) -> bool:
        """判断是否应该暂停交易"""
        drawdown = (initial_equity - current_equity) / initial_equity if initial_equity > 0 else 0
        
        # 回撤超过30%或连续亏损过多时暂停
        if drawdown > 0.30:
            return True
        
        if self.consecutive_losses >= self.consecutive_loss_limit * 2:
            return True
        
        return False
    
    def get_risk_adjusted_position_size(
        self,
        base_position_size: float,
        risk_level: str
    ) -> float:
        """
        根据风险等级调整仓位大小
        
        Args:
            base_position_size: 基础仓位大小
            risk_level: 风险等级
        
        Returns:
            调整后的仓位大小
        """
        if risk_level == 'extreme':
            return base_position_size * 0.3  # 极端风险：只用30%仓位
        elif risk_level == 'high':
            return base_position_size * 0.6  # 高风险：用60%仓位
        else:
            return base_position_size  # 正常风险：用100%仓位
    
    def reset(self):
        """重置状态"""
        self.consecutive_losses = 0
        self.peak_equity = None
        self.risk_level = 'normal'

