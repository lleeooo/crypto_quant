"""
盈利交易分析主脚本
分析回测结果中的盈利交易，提取有效因子和策略模式
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import default_config
from data.fetch_data import DataFetcher
from strategies.ema_rsi_strategy import EmaRsiStrategy
from backtester.backtest import Backtester
from analysis.trade_analyzer import TradeAnalyzer
import pandas as pd


def analyze_winning_trades(
    symbol: str = 'BTC/USDT',
    timeframe: str = '30m',
    limit: int = 20000
):
    """分析盈利交易"""
    print("\n" + "=" * 80)
    print("[盈利交易分析]")
    print("=" * 80)
    print(f"交易对: {symbol}")
    print(f"时间框架: {timeframe}")
    print(f"K线数量: {limit}")
    print("=" * 80)
    
    # 1. 获取数据
    print("\n>>> Step 1: 获取市场数据...")
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=default_config.proxy.to_dict() if default_config.proxy.enabled else None,
        cache_enabled=True
    )
    
    df = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit
    )
    print(f"   数据范围: {df.index[0]} ~ {df.index[-1]}")
    
    # 2. 运行策略
    print("\n>>> Step 2: 运行策略生成信号...")
    strategy = EmaRsiStrategy(
        risk_per_trade=default_config.strategy.risk_per_trade,
        use_volume_filter=default_config.strategy.use_volume_filter
    )
    df = strategy.generate_signals(df)
    
    buy_signals = len(df[df['signal'] == 1])
    sell_signals = len(df[df['signal'] == -1])
    print(f"   买入信号: {buy_signals}")
    print(f"   卖出信号: {sell_signals}")
    
    # 3. 运行回测
    print("\n>>> Step 3: 执行回测...")
    backtester = Backtester(
        initial_capital=default_config.backtest.initial_capital,
        fee_rate=default_config.backtest.fee_rate,
        slippage=default_config.backtest.slippage,
        use_stop_loss=default_config.backtest.use_stop_loss,
        use_take_profit=default_config.backtest.use_take_profit
    )
    
    df, result = backtester.run(df)
    
    # 4. 打印回测结果
    backtester.print_report(result)
    
    # 5. 分析盈利交易
    print("\n>>> Step 4: 分析盈利交易特征...")
    analyzer = TradeAnalyzer()
    analysis = analyzer.analyze_trades(result.trades, df)
    
    # 6. 打印分析报告
    analyzer.print_analysis_report(analysis)
    
    # 7. 提取最优过滤条件
    print("\n>>> Step 5: 提取最优过滤条件...")
    optimal_filters = analyzer.extract_optimal_filters(analysis)
    
    print("\n[最优过滤条件]")
    print("-" * 80)
    for filter_name, filter_data in optimal_filters.items():
        print(f"{filter_name}:")
        print(f"   范围: [{filter_data['min']:.4f}, {filter_data['max']:.4f}]")
        print(f"   均值: {filter_data['mean']:.4f}")
    
    # 8. 保存分析结果
    print("\n>>> Step 6: 保存分析结果...")
    analysis_dir = 'analysis/results'
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # 保存为JSON
    import json
    from datetime import datetime
    
    result_file = os.path.join(analysis_dir, f'winning_trades_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # 转换分析结果为可序列化格式
    serializable_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, dict):
            serializable_analysis[key] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) 
                                         for k, v in value.items()}
        else:
            serializable_analysis[key] = value
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis': serializable_analysis,
            'optimal_filters': {k: {kk: float(vv) if isinstance(vv, (int, float, np.integer, np.floating)) else vv 
                                   for kk, vv in v.items()} 
                              for k, v in optimal_filters.items()},
            'backtest_summary': {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': float(result.win_rate),
                'total_return_pct': float(result.total_return_pct)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   分析结果已保存至: {result_file}")
    
    return analysis, optimal_filters, result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='盈利交易分析工具')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--timeframe', type=str, default='30m', help='时间框架')
    parser.add_argument('--limit', type=int, default=20000, help='K线数量')
    
    args = parser.parse_args()
    
    analyze_winning_trades(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit
    )

