"""
加密货币量化交易系统 - 主程序
Author: Quant Expert
Version: 3.0 (因子版)
"""
import sys
import argparse
from datetime import datetime
import io
import os

# 修复Windows终端编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 导入模块
from config.settings import Config, default_config
from data.fetch_data import DataFetcher
from strategies.ema_rsi_strategy import EmaRsiStrategy
from backtester.backtest import Backtester, plot_results


def run_backtest(config: Config = None, show_chart: bool = True, use_factor: bool = False):
    """运行回测"""
    if config is None:
        config = default_config
    
    strategy_name = "多因子策略" if use_factor else "EMA+RSI策略"
    
    print("\n" + "=" * 60)
    print(f"[*] 加密货币量化交易系统 v3.0")
    print("=" * 60)
    print(f"[日期] 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[策略] {strategy_name}")
    print(f"[交易对] {config.data.symbol}")
    print(f"[时间框架] {config.data.timeframe}")
    print(f"[K线数量] {config.data.limit}")
    print(f"[初始资金] ${config.backtest.initial_capital:,.2f}")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n>>> Step 1: 获取市场数据...")
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=config.proxy.to_dict(),
        cache_enabled=config.data.cache_enabled,
        cache_dir=config.data.cache_dir
    )
    
    df = fetcher.fetch_ohlcv(
        symbol=config.data.symbol,
        timeframe=config.data.timeframe,
        limit=config.data.limit
    )
    
    print(f"   数据范围: {df.index[0]} ~ {df.index[-1]}")
    
    # 2. 初始化策略
    print("\n>>> Step 2: 初始化策略...")
    
    if use_factor:
        # 使用多因子策略
        from factors.factor_engine import FactorStrategy
        strategy = FactorStrategy(
            lookback_period=100,
            signal_threshold=0.25,
            min_holding_period=12,
            stop_loss_atr=2.0,
            take_profit_atr=4.0,
            risk_per_trade=config.strategy.risk_per_trade
        )
        print(f"   回看周期: 100")
        print(f"   信号阈值: 0.25")
        print(f"   最小持仓: 12根K线")
        print(f"   止损倍数: 2.0x ATR")
        print(f"   止盈倍数: 4.0x ATR")
    else:
        # 使用EMA+RSI策略
        strategy = EmaRsiStrategy(
            risk_per_trade=config.strategy.risk_per_trade,
            use_volume_filter=config.strategy.use_volume_filter
        )
        print(f"   EMA周期: {strategy.ema_fast}/{strategy.ema_slow}/{strategy.ema_trend}")
        print(f"   RSI周期: {strategy.rsi_period} (超买:{strategy.rsi_overbought}/超卖:{strategy.rsi_oversold})")
        print(f"   ADX阈值: {strategy.adx_threshold}")
        print(f"   止损倍数: {strategy.atr_stop_multiplier}x ATR")
        print(f"   止盈倍数: {strategy.atr_take_profit_multiplier}x ATR")
        print(f"   移动止损: {'启用' if strategy.use_trailing_stop else '禁用'}")
    
    # 3. 生成信号
    print("\n>>> Step 3: 生成交易信号...")
    df = strategy.generate_signals(df)
    
    buy_signals = len(df[df['signal'] == 1])
    sell_signals = len(df[df['signal'] == -1])
    print(f"   买入信号: {buy_signals}")
    print(f"   卖出信号: {sell_signals}")
    
    # 显示因子得分（如果使用因子策略）
    if use_factor and 'factor_score' in df.columns:
        latest_score = df['factor_score'].iloc[-1]
        print(f"   当前因子得分: {latest_score:.3f}")
    
    # 4. 回测
    print("\n>>> Step 4: 执行回测...")
    backtester = Backtester(
        initial_capital=config.backtest.initial_capital,
        fee_rate=config.backtest.fee_rate,
        slippage=config.backtest.slippage,
        use_stop_loss=config.backtest.use_stop_loss,
        use_take_profit=config.backtest.use_take_profit
    )
    
    df, result = backtester.run(df)
    
    # 5. 输出报告
    backtester.print_report(result)
    
    # 6. 绘制图表
    if show_chart:
        print("\n>>> Step 5: 生成图表...")
        chart_path = f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_results(df, result, save_path=chart_path)
    
    # 7. 交易明细
    print("\n[交易明细] 最近10笔交易:")
    print("-" * 80)
    for trade in result.trades[-10:]:
        direction = "[多]" if trade.direction == 1 else "[空]"
        pnl_sign = "+" if trade.pnl > 0 else ""
        print(f"   {direction} | 入场: ${trade.entry_price:.2f} | 出场: ${trade.exit_price:.2f} | "
              f"盈亏: {pnl_sign}${trade.pnl:.2f} ({trade.pnl_pct*100:.2f}%) | {trade.exit_reason}")
    
    print("\n" + "=" * 60)
    print("[OK] 回测完成!")
    print("=" * 60 + "\n")
    
    return df, result


def run_factor_analysis(config: Config = None):
    """运行因子分析"""
    if config is None:
        config = default_config
    
    print("\n" + "=" * 60)
    print("[因子分析] 加密货币因子研究")
    print("=" * 60)
    
    # 获取数据
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=config.proxy.to_dict(),
        cache_enabled=True
    )
    
    df = fetcher.fetch_ohlcv(
        symbol=config.data.symbol,
        timeframe=config.data.timeframe,
        limit=config.data.limit
    )
    
    # 计算因子
    from factors.factor_lib import FactorCalculator
    from factors.factor_engine import FactorEngine
    
    calc = FactorCalculator(df)
    factors = calc.calculate_all()
    
    print(f"\n[信息] 计算了 {len(factors.columns)} 个因子")
    print(f"[信息] 数据范围: {df.index[0]} ~ {df.index[-1]}")
    
    # 计算未来收益率用于因子分析
    future_returns = df['close'].pct_change(10).shift(-10)  # 未来10期收益
    
    # 因子分析
    engine = FactorEngine()
    analysis = engine.factor_analysis(factors, future_returns)
    engine.print_factor_report(analysis)
    
    # 当前因子暴露
    normalized = engine.normalize_factors(factors)
    exposure = engine.get_factor_exposure(normalized)
    
    print("\n[当前因子暴露] (最新一期)")
    print("-" * 50)
    for idx, row in exposure.head(10).iterrows():
        direction = "+" if row['exposure'] > 0 else ""
        print(f"   {idx:<20} {direction}{row['exposure']:.3f} (权重:{row['weight']:.2f})")
    
    # 综合因子得分
    composite = engine.composite_factor(normalized)
    latest_score = composite.iloc[-1]
    
    print("\n" + "-" * 50)
    print(f"[综合因子得分] {latest_score:.3f}")
    
    if latest_score > 0.3:
        print("[信号] 强烈看多")
    elif latest_score > 0.1:
        print("[信号] 温和看多")
    elif latest_score < -0.3:
        print("[信号] 强烈看空")
    elif latest_score < -0.1:
        print("[信号] 温和看空")
    else:
        print("[信号] 中性观望")
    
    print("=" * 60 + "\n")
    
    return factors, analysis


def optimize_strategy(config: Config = None):
    """策略参数优化"""
    if config is None:
        config = default_config
    
    print("\n[优化] 开始参数优化...")
    
    # 获取数据
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=config.proxy.to_dict(),
        cache_enabled=True
    )
    df = fetcher.fetch_ohlcv(
        symbol=config.data.symbol,
        timeframe=config.data.timeframe,
        limit=config.data.limit
    )
    
    # 参数网格
    ema_fast_range = [10, 15, 20, 25]
    ema_slow_range = [40, 50, 60]
    rsi_period_range = [10, 14, 18]
    
    best_result = None
    best_params = None
    best_sharpe = float('-inf')
    
    total_combinations = len(ema_fast_range) * len(ema_slow_range) * len(rsi_period_range)
    current = 0
    
    for ema_fast in ema_fast_range:
        for ema_slow in ema_slow_range:
            if ema_fast >= ema_slow:
                continue
            for rsi_period in rsi_period_range:
                current += 1
                
                strategy = EmaRsiStrategy(
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    rsi_period=rsi_period
                )
                
                df_test = strategy.generate_signals(df.copy())
                
                backtester = Backtester(
                    initial_capital=config.backtest.initial_capital,
                    fee_rate=config.backtest.fee_rate
                )
                
                _, result = backtester.run(df_test)
                
                print(f"\r   进度: {current}/{total_combinations} | "
                      f"EMA({ema_fast}/{ema_slow}) RSI({rsi_period}) | "
                      f"Sharpe: {result.sharpe_ratio:.2f} | "
                      f"Return: {result.total_return_pct*100:.2f}%", end="")
                
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_result = result
                    best_params = {
                        'ema_fast': ema_fast,
                        'ema_slow': ema_slow,
                        'rsi_period': rsi_period
                    }
    
    print(f"\n\n[最优参数]:")
    print(f"   EMA Fast: {best_params['ema_fast']}")
    print(f"   EMA Slow: {best_params['ema_slow']}")
    print(f"   RSI Period: {best_params['rsi_period']}")
    print(f"   夏普比率: {best_result.sharpe_ratio:.2f}")
    print(f"   总收益: {best_result.total_return_pct*100:.2f}%")
    print(f"   最大回撤: {best_result.max_drawdown_pct*100:.2f}%")
    
    return best_params, best_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币量化交易系统 v3.0')
    parser.add_argument('--mode', type=str, default='backtest', 
                        choices=['backtest', 'optimize', 'factor', 'analysis'],
                        help='运行模式: backtest(回测), optimize(优化), factor(因子回测), analysis(因子分析)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', 
                        help='交易对 (默认: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h', 
                        help='时间框架 (默认: 1h)')
    parser.add_argument('--limit', type=int, default=1000, 
                        help='K线数量 (默认: 1000)')
    parser.add_argument('--capital', type=float, default=None, 
                        help='初始资金 (默认: 使用配置文件)')
    parser.add_argument('--no-chart', action='store_true', 
                        help='不显示图表')
    parser.add_argument('--no-proxy', action='store_true',
                        help='禁用代理')
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    config.data.symbol = args.symbol
    config.data.timeframe = args.timeframe
    config.data.limit = args.limit
    
    # 只有命令行指定了 --capital 才覆盖配置文件
    if args.capital is not None:
        config.backtest.initial_capital = args.capital
    
    if args.no_proxy:
        config.proxy.enabled = False
    
    # 确保目录存在
    for dir_path in ['logs', 'data/cache']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 运行
    if args.mode == 'backtest':
        run_backtest(config, show_chart=not args.no_chart, use_factor=False)
    elif args.mode == 'factor':
        run_backtest(config, show_chart=not args.no_chart, use_factor=True)
    elif args.mode == 'optimize':
        optimize_strategy(config)
    elif args.mode == 'analysis':
        run_factor_analysis(config)


if __name__ == '__main__':
    main()
