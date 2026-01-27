"""
数据导出工具
将多时间框架的回测数据保存到本地，方便后续分析
"""
import os
import json
import pandas as pd
from datetime import datetime
from config.settings import Config
from data.fetch_data import DataFetcher
from strategies.ema_rsi_strategy import EmaRsiStrategy
from strategies.improved_strategy import TrailingStopStrategy
from backtester.backtest import Backtester


def export_all_timeframes(
    symbol: str = 'BTC/USDT',
    timeframes: list = ['30m', '1h', '2h', '4h'],
    limit: int = 2000,
    output_dir: str = 'localData'
):
    """
    导出多时间框架的回测数据
    
    Args:
        symbol: 交易对
        timeframes: 时间框架列表
        limit: K线数量
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config = Config()
    
    # 初始化数据获取器
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=config.proxy.to_dict(),
        cache_enabled=True
    )
    
    # 策略列表
    strategies = {
        '原版策略': EmaRsiStrategy(),
        '移动止损策略': TrailingStopStrategy(),
    }
    
    # 汇总结果
    all_results = {
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'initial_capital': config.backtest.initial_capital,
        'fee_rate': config.backtest.fee_rate,
        'timeframes': {}
    }
    
    print("=" * 80)
    print(f"[数据导出] 开始导出 {symbol} 多时间框架数据")
    print("=" * 80)
    
    for tf in timeframes:
        print(f"\n>>> 处理 {tf} 时间框架...")
        
        # 获取数据
        try:
            df = fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf,
                limit=limit
            )
        except Exception as e:
            print(f"   [错误] 获取数据失败: {e}")
            continue
        
        # 保存原始OHLCV数据
        ohlcv_file = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{tf}_ohlcv.csv')
        df.to_csv(ohlcv_file)
        print(f"   [保存] OHLCV数据: {ohlcv_file}")
        
        # 时间框架结果
        tf_results = {
            'data_range': {
                'start': str(df.index[0]),
                'end': str(df.index[-1]),
                'bars': len(df)
            },
            'price_stats': {
                'open': float(df['open'].iloc[0]),
                'close': float(df['close'].iloc[-1]),
                'high': float(df['high'].max()),
                'low': float(df['low'].min()),
                'price_change_pct': float((df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0] * 100),
                'avg_volume': float(df['volume'].mean()) if 'volume' in df.columns else 0,
            },
            'strategies': {}
        }
        
        # 对每个策略回测
        for strategy_name, strategy in strategies.items():
            print(f"   [回测] {strategy_name}...")
            
            df_test = strategy.generate_signals(df.copy())
            
            backtester = Backtester(
                initial_capital=config.backtest.initial_capital,
                fee_rate=config.backtest.fee_rate,
                slippage=config.backtest.slippage,
                use_stop_loss=True,
                use_take_profit=True
            )
            
            _, result = backtester.run(df_test)
            
            # 计算盈亏比
            pl_ratio = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0
            
            # 交易详情
            trades_detail = []
            for trade in result.trades:
                trades_detail.append({
                    'entry_time': str(trade.entry_time),
                    'exit_time': str(trade.exit_time),
                    'direction': '多' if trade.direction == 1 else '空',
                    'entry_price': round(trade.entry_price, 2),
                    'exit_price': round(trade.exit_price, 2),
                    'pnl': round(trade.pnl, 2),
                    'pnl_pct': round(trade.pnl_pct * 100, 2),
                    'exit_reason': trade.exit_reason
                })
            
            # 策略结果
            strategy_result = {
                'performance': {
                    'total_return_pct': round(result.total_return_pct * 100, 2),
                    'annual_return_pct': round(result.annual_return * 100, 2),
                    'max_drawdown_pct': round(result.max_drawdown_pct * 100, 2),
                    'sharpe_ratio': round(result.sharpe_ratio, 2),
                    'sortino_ratio': round(result.sortino_ratio, 2),
                    'calmar_ratio': round(result.calmar_ratio, 2),
                    'profit_factor': round(result.profit_factor, 2),
                },
                'trades': {
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate_pct': round(result.win_rate * 100, 2),
                    'avg_win': round(result.avg_win, 2),
                    'avg_loss': round(result.avg_loss, 2),
                    'profit_loss_ratio': round(pl_ratio, 2),
                    'expectancy': round(result.expectancy, 2),
                },
                'capital': {
                    'initial': round(result.initial_capital, 2),
                    'final': round(result.final_capital, 2),
                    'total_return': round(result.total_return, 2),
                },
                'trades_detail': trades_detail
            }
            
            tf_results['strategies'][strategy_name] = strategy_result
            
            print(f"      胜率: {result.win_rate*100:.2f}% | "
                  f"盈亏比: {pl_ratio:.2f} | "
                  f"收益: {result.total_return_pct*100:.2f}% | "
                  f"夏普: {result.sharpe_ratio:.2f}")
        
        all_results['timeframes'][tf] = tf_results
    
    # 保存汇总JSON
    summary_file = os.path.join(output_dir, f'{symbol.replace("/", "_")}_backtest_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 汇总数据: {summary_file}")
    
    # 生成可读的Markdown报告
    report_file = os.path.join(output_dir, f'{symbol.replace("/", "_")}_analysis_report.md')
    generate_markdown_report(all_results, report_file)
    print(f"[保存] 分析报告: {report_file}")
    
    print("\n" + "=" * 80)
    print("[完成] 所有数据已导出到 localData 目录")
    print("=" * 80)
    
    return all_results


def generate_markdown_report(data: dict, output_file: str):
    """生成Markdown格式的分析报告"""
    
    lines = [
        f"# {data['symbol']} 多时间框架回测分析报告",
        f"",
        f"**导出时间**: {data['export_time']}",
        f"**初始资金**: ${data['initial_capital']:,.2f}",
        f"**手续费率**: {data['fee_rate']*100:.2f}%",
        f"",
        f"---",
        f"",
        f"## 汇总对比",
        f"",
        f"| 时间框架 | 策略 | 胜率 | 盈亏比 | 总收益% | 最大回撤% | 夏普比 | 交易数 |",
        f"|----------|------|------|--------|---------|-----------|--------|--------|",
    ]
    
    for tf, tf_data in data['timeframes'].items():
        for strategy_name, s_data in tf_data['strategies'].items():
            perf = s_data['performance']
            trades = s_data['trades']
            lines.append(
                f"| {tf} | {strategy_name} | "
                f"{trades['win_rate_pct']}% | "
                f"{trades['profit_loss_ratio']} | "
                f"{perf['total_return_pct']}% | "
                f"{perf['max_drawdown_pct']}% | "
                f"{perf['sharpe_ratio']} | "
                f"{trades['total_trades']} |"
            )
    
    lines.extend([
        f"",
        f"---",
        f"",
    ])
    
    # 每个时间框架详情
    for tf, tf_data in data['timeframes'].items():
        lines.extend([
            f"## {tf} 时间框架详情",
            f"",
            f"**数据范围**: {tf_data['data_range']['start']} ~ {tf_data['data_range']['end']}",
            f"**K线数量**: {tf_data['data_range']['bars']}",
            f"",
            f"### 价格统计",
            f"- 开盘价: ${tf_data['price_stats']['open']:,.2f}",
            f"- 收盘价: ${tf_data['price_stats']['close']:,.2f}",
            f"- 最高价: ${tf_data['price_stats']['high']:,.2f}",
            f"- 最低价: ${tf_data['price_stats']['low']:,.2f}",
            f"- 涨跌幅: {tf_data['price_stats']['price_change_pct']:.2f}%",
            f"",
        ])
        
        for strategy_name, s_data in tf_data['strategies'].items():
            perf = s_data['performance']
            trades = s_data['trades']
            capital = s_data['capital']
            
            lines.extend([
                f"### {strategy_name}",
                f"",
                f"**资金概览**",
                f"- 初始资金: ${capital['initial']:,.2f}",
                f"- 最终资金: ${capital['final']:,.2f}",
                f"- 总收益: ${capital['total_return']:,.2f} ({perf['total_return_pct']}%)",
                f"",
                f"**交易统计**",
                f"- 总交易: {trades['total_trades']} 笔",
                f"- 盈利: {trades['winning_trades']} 笔",
                f"- 亏损: {trades['losing_trades']} 笔",
                f"- 胜率: {trades['win_rate_pct']}%",
                f"- 平均盈利: ${trades['avg_win']:.2f}",
                f"- 平均亏损: ${trades['avg_loss']:.2f}",
                f"- 盈亏比: {trades['profit_loss_ratio']}",
                f"- 期望值: ${trades['expectancy']:.2f}",
                f"",
                f"**风险指标**",
                f"- 夏普比率: {perf['sharpe_ratio']}",
                f"- 索提诺比率: {perf['sortino_ratio']}",
                f"- 卡尔马比率: {perf['calmar_ratio']}",
                f"- 利润因子: {perf['profit_factor']}",
                f"- 最大回撤: {perf['max_drawdown_pct']}%",
                f"- 年化收益: {perf['annual_return_pct']}%",
                f"",
                f"**交易明细**",
                f"",
                f"| 入场时间 | 方向 | 入场价 | 出场价 | 盈亏 | 盈亏% | 出场原因 |",
                f"|----------|------|--------|--------|------|-------|----------|",
            ])
            
            for trade in s_data['trades_detail'][-20:]:  # 最近20笔
                pnl_sign = '+' if trade['pnl'] > 0 else ''
                lines.append(
                    f"| {trade['entry_time'][:16]} | {trade['direction']} | "
                    f"${trade['entry_price']:,.2f} | ${trade['exit_price']:,.2f} | "
                    f"{pnl_sign}${trade['pnl']:.2f} | {pnl_sign}{trade['pnl_pct']:.2f}% | "
                    f"{trade['exit_reason']} |"
                )
            
            lines.extend([f"", f"---", f""])
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='导出多时间框架回测数据')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--limit', type=int, default=2000, help='K线数量')
    parser.add_argument('--output', type=str, default='localData', help='输出目录')
    
    args = parser.parse_args()
    
    export_all_timeframes(
        symbol=args.symbol,
        timeframes=['30m', '1h', '2h', '4h'],
        limit=args.limit,
        output_dir=args.output
    )

