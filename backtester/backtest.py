"""
专业级回测引擎
特性：
1. 完整的交易统计指标
2. 止损止盈执行
3. 滑点和手续费模拟
4. 资金曲线和回撤分析
5. 可视化图表
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """交易记录"""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: int  # 1: 多, -1: 空
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'


@dataclass
class BacktestResult:
    """回测结果"""
    # 基础统计
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # 盈亏分析
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # 风险指标
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # K线数量
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 年化指标
    annual_return: float
    annual_volatility: float
    
    # 详细数据
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class Backtester:
    """专业级回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        use_stop_loss: bool = True,
        use_take_profit: bool = True
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.use_stop_loss = use_stop_loss
        self.use_take_profit = use_take_profit
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, BacktestResult]:
        """执行回测"""
        df = df.copy()
        
        cash = self.initial_capital
        position = 0  # 持仓数量
        position_direction = 0  # 1: 多, -1: 空
        entry_price = 0
        entry_time = None
        current_stop_loss = None
        current_take_profit = None
        
        trades: List[Trade] = []
        equity_curve = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']
            high = row['high']
            low = row['low']
            signal = row['signal'] if 'signal' in row.index else 0
            
            exit_reason = None
            
            # 检查止损止盈
            if position != 0 and self.use_stop_loss and current_stop_loss is not None:
                if position_direction == 1 and low <= current_stop_loss:
                    # 多头止损
                    exit_price = current_stop_loss * (1 - self.slippage)
                    exit_reason = 'stop_loss'
                elif position_direction == -1 and high >= current_stop_loss:
                    # 空头止损
                    exit_price = current_stop_loss * (1 + self.slippage)
                    exit_reason = 'stop_loss'
            
            if position != 0 and self.use_take_profit and current_take_profit is not None and exit_reason is None:
                if position_direction == 1 and high >= current_take_profit:
                    # 多头止盈
                    exit_price = current_take_profit * (1 - self.slippage)
                    exit_reason = 'take_profit'
                elif position_direction == -1 and low <= current_take_profit:
                    # 空头止盈
                    exit_price = current_take_profit * (1 + self.slippage)
                    exit_reason = 'take_profit'
            
            # 信号平仓
            if position != 0 and signal != 0 and signal != position_direction and exit_reason is None:
                exit_price = price * (1 - self.slippage * position_direction)
                exit_reason = 'signal'
            
            # 执行平仓
            if exit_reason is not None:
                pnl = position * (exit_price - entry_price) * position_direction
                pnl -= abs(position * exit_price) * self.fee_rate  # 手续费
                pnl_pct = pnl / (position * entry_price)
                
                cash += position * exit_price * (1 - self.fee_rate) if position_direction == 1 else \
                        (2 * position * entry_price - position * exit_price) * (1 - self.fee_rate)
                
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=df.index[i],
                    direction=position_direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=position,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason
                ))
                
                position = 0
                position_direction = 0
                current_stop_loss = None
                current_take_profit = None
            
            # 开仓
            if signal != 0 and position == 0:
                entry_price = price * (1 + self.slippage * signal)
                entry_time = df.index[i]
                
                # 获取仓位大小
                pos_size = row['position_size'] if 'position_size' in row.index and row['position_size'] > 0 else 1.0
                
                position = (cash * pos_size) / entry_price
                cash -= position * entry_price * (1 + self.fee_rate)
                position_direction = signal
                
                if 'stop_loss' in row.index and pd.notna(row['stop_loss']):
                    current_stop_loss = row['stop_loss']
                if 'take_profit' in row.index and pd.notna(row['take_profit']):
                    current_take_profit = row['take_profit']
            
            # 计算权益
            if position > 0:
                equity = cash + position * price
            else:
                equity = cash
            equity_curve.append(equity)
        
        # 强制平仓
        if position != 0:
            exit_price = df['close'].iloc[-1]
            pnl = position * (exit_price - entry_price) * position_direction
            pnl -= abs(position * exit_price) * self.fee_rate
            pnl_pct = pnl / (position * entry_price)
            
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=df.index[-1],
                direction=position_direction,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason='end_of_data'
            ))
            cash += position * exit_price * (1 - self.fee_rate)
        
        df['equity'] = equity_curve
        result = self._calculate_statistics(trades, equity_curve, df)
        
        return df, result
    
    def _calculate_statistics(
        self, 
        trades: List[Trade], 
        equity_curve: List[float],
        df: pd.DataFrame
    ) -> BacktestResult:
        """计算统计指标"""
        equity = np.array(equity_curve)
        
        # 基础统计
        final_capital = equity[-1] if len(equity) > 0 else self.initial_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # 交易统计
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏分析
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if total_trades > 0 else 0
        
        # 回撤分析
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown_pct = np.max(drawdown) if len(drawdown) > 0 else 0
        max_drawdown = np.max(peak - equity) if len(equity) > 0 else 0
        
        # 计算最大回撤持续期
        max_dd_duration = 0
        current_dd_duration = 0
        for i in range(len(equity)):
            if equity[i] < peak[i]:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # 收益率序列
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        returns = np.nan_to_num(returns, nan=0, posinf=0, neginf=0)
        
        # 年化指标 (假设1小时K线，一年约8760小时)
        periods_per_year = 8760
        n_periods = len(equity)
        
        annual_return = ((final_capital / self.initial_capital) ** (periods_per_year / n_periods) - 1) if n_periods > 0 else 0
        annual_volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 0 else 0
        
        # 夏普比率 (假设无风险利率 2%)
        risk_free_rate = 0.02
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        
        # 索提诺比率 (只考虑下行波动)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # 卡尔马比率
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown.tolist()
        )
    
    def print_report(self, result: BacktestResult):
        """打印回测报告"""
        print("\n" + "=" * 60)
        print("[回测报告]")
        print("=" * 60)
        
        print(f"\n[资金概览]")
        print(f"   初始资金: ${result.initial_capital:,.2f}")
        print(f"   最终资金: ${result.final_capital:,.2f}")
        print(f"   总收益: ${result.total_return:,.2f} ({result.total_return_pct*100:.2f}%)")
        
        print(f"\n[交易统计]")
        print(f"   总交易次数: {result.total_trades}")
        print(f"   盈利次数: {result.winning_trades}")
        print(f"   亏损次数: {result.losing_trades}")
        print(f"   胜率: {result.win_rate*100:.2f}%")
        
        print(f"\n[盈亏分析]")
        print(f"   平均盈利: ${result.avg_win:,.2f}")
        print(f"   平均亏损: ${result.avg_loss:,.2f}")
        print(f"   盈亏比: {abs(result.avg_win/result.avg_loss) if result.avg_loss != 0 else float('inf'):.2f}")
        print(f"   利润因子: {result.profit_factor:.2f}")
        print(f"   期望值: ${result.expectancy:,.2f}")
        
        print(f"\n[风险指标]")
        print(f"   最大回撤: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct*100:.2f}%)")
        print(f"   最大回撤持续期: {result.max_drawdown_duration} 根K线")
        print(f"   夏普比率: {result.sharpe_ratio:.2f}")
        print(f"   索提诺比率: {result.sortino_ratio:.2f}")
        print(f"   卡尔马比率: {result.calmar_ratio:.2f}")
        
        print(f"\n[年化指标]")
        print(f"   年化收益: {result.annual_return*100:.2f}%")
        print(f"   年化波动率: {result.annual_volatility*100:.2f}%")
        
        print("\n" + "=" * 60)


def plot_results(df: pd.DataFrame, result: BacktestResult, save_path: str = None):
    """绘制回测结果图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('Backtest Results', fontsize=14, fontweight='bold')
        
        # 1. 价格和信号
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Price', color='#2E86AB', linewidth=1)
        if 'EMA_fast' in df.columns:
            ax1.plot(df.index, df['EMA_fast'], label='EMA Fast', color='#F18F01', linewidth=0.8)
        if 'EMA_slow' in df.columns:
            ax1.plot(df.index, df['EMA_slow'], label='EMA Slow', color='#C73E1D', linewidth=0.8)
        
        # 标记买卖点
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='#2ECC71', s=100, label='Buy', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='#E74C3C', s=100, label='Sell', zorder=5)
        
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 权益曲线
        ax2 = axes[1]
        ax2.fill_between(df.index, result.equity_curve, alpha=0.3, color='#2E86AB')
        ax2.plot(df.index, result.equity_curve, color='#2E86AB', linewidth=1.5)
        ax2.axhline(y=result.initial_capital, color='#95A5A6', linestyle='--', linewidth=1)
        ax2.set_ylabel('Equity')
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤
        ax3 = axes[2]
        drawdown_pct = np.array(result.drawdown_curve) * 100
        ax3.fill_between(df.index, drawdown_pct, alpha=0.5, color='#E74C3C')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. RSI
        if 'RSI' in df.columns:
            ax4 = axes[3]
            ax4.plot(df.index, df['RSI'], color='#9B59B6', linewidth=1)
            ax4.axhline(y=70, color='#E74C3C', linestyle='--', linewidth=0.8)
            ax4.axhline(y=30, color='#2ECC71', linestyle='--', linewidth=0.8)
            ax4.axhline(y=50, color='#95A5A6', linestyle='--', linewidth=0.5)
            ax4.set_ylabel('RSI')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
        
        plt.xlabel('Date')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[图表] 已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("[警告] matplotlib未安装，跳过图表绘制")


# ============ 向后兼容的接口 ============

def backtest(df, initial_cash=10000):
    """向后兼容的简化接口"""
    backtester = Backtester(initial_capital=initial_cash, use_stop_loss=False, use_take_profit=False)
    df_result, _ = backtester.run(df)
    return df_result


def backtest_statistics(df, initial_cash=10000, fee_rate=0.001):
    """向后兼容的统计接口"""
    backtester = Backtester(initial_capital=initial_cash, fee_rate=fee_rate)
    df_result, result = backtester.run(df)
    
    # 转换为旧格式
    trades = [{
        'buy_price': t.entry_price,
        'sell_price': t.exit_price,
        'profit': t.pnl
    } for t in result.trades]
    
    return {
        'trades': trades,
        'total_trades': result.total_trades,
        'wins': result.winning_trades,
        'losses': result.losing_trades,
        'win_rate': result.win_rate,
        'total_profit': sum(t.pnl for t in result.trades if t.pnl > 0),
        'total_loss': sum(t.pnl for t in result.trades if t.pnl < 0),
        'avg_win': result.avg_win,
        'avg_loss': result.avg_loss,
        'profit_loss_ratio': abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else float('inf'),
        'final_capital': result.final_capital,
        'equity_curve': result.equity_curve,
        # 新增指标
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown_pct': result.max_drawdown_pct,
        'annual_return': result.annual_return
    }
