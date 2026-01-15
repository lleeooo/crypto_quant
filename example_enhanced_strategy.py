"""
增强版策略使用示例
展示如何使用基于盈利交易分析优化的策略和黑天鹅防护
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import default_config
from data.fetch_data import DataFetcher
from strategies.enhanced_ema_rsi_strategy import EnhancedEmaRsiStrategy
from backtester.backtest import Backtester
from datetime import datetime


def run_enhanced_strategy_example():
    """运行增强版策略示例"""
    print("\n" + "=" * 80)
    print("[增强版策略示例]")
    print("=" * 80)
    
    # 1. 获取数据
    print("\n>>> Step 1: 获取市场数据...")
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=default_config.proxy.to_dict() if default_config.proxy.enabled else None,
        cache_enabled=True
    )
    
    df = fetcher.fetch_ohlcv(
        symbol='BTC/USDT',
        timeframe='30m',
        limit=20000
    )
    print(f"   数据范围: {df.index[0]} ~ {df.index[-1]}")
    
    # 2. 创建增强版策略
    # 注意：这里使用示例的最优过滤条件
    # 实际使用时，应该先运行 analyze_winning_trades.py 获取真实的最优过滤条件
    print("\n>>> Step 2: 初始化增强版策略...")
    
    # 示例最优过滤条件（基于盈利交易分析）
    optimal_filters = {
        'RSI': {'min': 40, 'max': 65, 'mean': 52},
        'ADX': {'min': 25, 'max': 45, 'mean': 35},
        'VOLUME_RATIO': {'min': 1.0, 'max': 2.5, 'mean': 1.5},
        'ATR_PCT': {'min': 0.5, 'max': 3.0, 'mean': 1.5},
        'VOLATILITY_10': {'min': 0.5, 'max': 2.5, 'mean': 1.2},
        'TREND_STRENGTH': {'min': 0.3, 'max': 0.9, 'mean': 0.6}
    }
    
    strategy = EnhancedEmaRsiStrategy(
        risk_per_trade=default_config.strategy.risk_per_trade,
        use_volume_filter=default_config.strategy.use_volume_filter,
        enable_black_swan_protection=True,  # 启用黑天鹅防护
        optimal_filters=optimal_filters,  # 使用最优过滤条件
        rsi_optimal_range=(40, 65),  # 基于盈利交易的RSI区间
        adx_optimal_range=(25, 45),  # 基于盈利交易的ADX区间
        volume_optimal_range=(1.0, 2.5)  # 基于盈利交易的成交量区间
    )
    
    print("   ✓ 黑天鹅防护: 已启用")
    print("   ✓ 最优过滤条件: 已加载")
    print("   ✓ 动态风险调整: 已启用")
    
    # 3. 生成信号
    print("\n>>> Step 3: 生成交易信号...")
    df = strategy.generate_signals(df)
    
    buy_signals = len(df[df['signal'] == 1])
    sell_signals = len(df[df['signal'] == -1])
    black_swan_events = len(df[df['black_swan_flag'] == 1])
    
    print(f"   买入信号: {buy_signals}")
    print(f"   卖出信号: {sell_signals}")
    print(f"   黑天鹅事件: {black_swan_events}")
    
    # 4. 运行回测
    print("\n>>> Step 4: 执行回测...")
    backtester = Backtester(
        initial_capital=default_config.backtest.initial_capital,
        fee_rate=default_config.backtest.fee_rate,
        slippage=default_config.backtest.slippage,
        use_stop_loss=default_config.backtest.use_stop_loss,
        use_take_profit=default_config.backtest.use_take_profit
    )
    
    df, result = backtester.run(df)
    
    # 5. 输出报告
    backtester.print_report(result)
    
    # 6. 对比分析
    print("\n[策略改进对比]")
    print("-" * 80)
    print(f"   总交易次数: {result.total_trades}")
    print(f"   盈利次数: {result.winning_trades} ({result.win_rate*100:.2f}%)")
    print(f"   亏损次数: {result.losing_trades}")
    print(f"   平均盈利: ${result.avg_win:.2f}")
    print(f"   平均亏损: ${result.avg_loss:.2f}")
    print(f"   盈亏比: {abs(result.avg_win/result.avg_loss) if result.avg_loss != 0 else 0:.2f}")
    print(f"   最大回撤: {result.max_drawdown_pct*100:.2f}%")
    print(f"   夏普比率: {result.sharpe_ratio:.2f}")
    
    print("\n" + "=" * 80)
    print("[完成] 增强版策略回测完成!")
    print("=" * 80 + "\n")
    
    return df, result


if __name__ == '__main__':
    run_enhanced_strategy_example()

