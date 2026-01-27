"""
加密货币量化交易系统 - 主程序
Author: Quant Expert
Version: 4.0 (实盘交易版)

功能:
- 回测分析
- 因子分析
- 策略对比
- 实盘交易
"""
import sys
import argparse
from datetime import datetime
import io
import os
import time

# 修复Windows终端编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 导入模块
from config.settings import Config, default_config
from data.fetch_data import DataFetcher
from strategies.ema_rsi_strategy import EmaRsiStrategy
from strategies.optimized_strategy import OptimizedStrategy, ConservativeStrategy, AggressiveStrategy, BalancedStrategy
from strategies.improved_strategy import ImprovedStrategy, HighWinRateStrategy, HighProfitRatioStrategy, OptimalStrategy, TrailingStopStrategy
from backtester.backtest import Backtester, plot_results


def load_local_data(symbol: str, timeframe: str, data_dir: str = 'localData') -> pd.DataFrame:
    """从本地加载数据"""
    import pandas as pd
    filename = f"{symbol.replace('/', '_')}_{timeframe}_ohlcv.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"本地数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def run_backtest(config: Config = None, show_chart: bool = True, use_factor: bool = False, 
                 strategy_type: str = 'original', use_local_data: bool = False):
    """运行回测
    
    Args:
        strategy_type: 'original' | 'optimized' | 'conservative' | 'aggressive'
        use_local_data: True使用本地数据，False使用线上API数据
    """
    if config is None:
        config = default_config
    
    strategy_names = {
        'original': "EMA+RSI策略(原版)",
        'optimized': "优化版策略 v4.0",
        'conservative': "保守版策略",
        'aggressive': "激进版策略",
        'factor': "多因子策略"
    }
    strategy_name = strategy_names.get(strategy_type, "EMA+RSI策略") if not use_factor else "多因子策略"
    
    data_source = "本地数据" if use_local_data else "线上API"
    
    print("\n" + "=" * 60)
    print(f"[*] 加密货币量化交易系统 v4.0")
    print("=" * 60)
    print(f"[日期] 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[策略] {strategy_name}")
    print(f"[交易对] {config.data.symbol}")
    print(f"[时间框架] {config.data.timeframe}")
    print(f"[数据源] {data_source}")
    print(f"[初始资金] ${config.backtest.initial_capital:,.2f}")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n>>> Step 1: 获取市场数据...")
    
    if use_local_data:
        # 从本地加载数据
        try:
            df = load_local_data(config.data.symbol, config.data.timeframe)
            print(f"   [本地] 加载成功: {len(df)} 根K线")
        except FileNotFoundError as e:
            print(f"   [错误] {e}")
            print(f"   [提示] 请先运行 python export_data.py 导出数据")
            return None, None
    else:
        # 从线上API获取数据
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
    print(f"   K线数量: {len(df)}")
    
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
    elif strategy_type == 'optimized':
        # 使用移动止损策略（保持原版入场，优化出场）
        strategy = TrailingStopStrategy()
        print(f"   EMA周期: {strategy.ema_fast}/{strategy.ema_slow}/{strategy.ema_trend}")
        print(f"   ADX趋势阈值: {strategy.adx_threshold}")
        print(f"   止损倍数: {strategy.atr_stop_multiplier}x ATR")
        print(f"   止盈倍数: {strategy.atr_take_profit_multiplier}x ATR")
        print(f"   保本触发: 盈利{strategy.breakeven_trigger}x ATR")
        print(f"   跟踪触发: 盈利{strategy.trailing_trigger}x ATR")
        print(f"   跟踪距离: {strategy.trailing_distance}x ATR")
    elif strategy_type == 'conservative':
        # 保守版策略
        strategy = ConservativeStrategy()
        print(f"   [保守模式] 更高胜率，更少交易")
        print(f"   入场评分阈值: {strategy.min_entry_score}")
        print(f"   止损倍数: {strategy.base_stop_multiplier}x ATR")
        print(f"   止盈倍数: {strategy.base_profit_multiplier}x ATR")
    elif strategy_type == 'aggressive':
        # 激进版策略
        strategy = AggressiveStrategy()
        print(f"   [激进模式] 更多交易机会")
        print(f"   入场评分阈值: {strategy.min_entry_score}")
        print(f"   止损倍数: {strategy.base_stop_multiplier}x ATR")
        print(f"   止盈倍数: {strategy.base_profit_multiplier}x ATR")
    else:
        # 使用EMA+RSI策略（原版）
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


def compare_strategies(config: Config = None, use_local_data: bool = False):
    """对比不同策略的表现"""
    if config is None:
        config = default_config
    
    data_source = "本地数据" if use_local_data else "线上API"
    
    print("\n" + "=" * 80)
    print("[策略对比分析] 多策略回测对比")
    print("=" * 80)
    
    # 获取数据
    if use_local_data:
        try:
            df = load_local_data(config.data.symbol, config.data.timeframe)
            print(f"[本地] 加载成功")
        except FileNotFoundError as e:
            print(f"[错误] {e}")
            print(f"[提示] 请先运行 python export_data.py 导出数据，或使用 --online 参数")
            return None
    else:
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
    
    print(f"[数据] {config.data.symbol} {config.data.timeframe} | {len(df)} 根K线")
    print(f"[来源] {data_source}")
    print(f"[日期] {df.index[0]} ~ {df.index[-1]}")
    
    # 测试不同策略
    strategies = [
        ('原版策略', EmaRsiStrategy()),
        ('移动止损策略', TrailingStopStrategy()),
        ('最优策略', OptimalStrategy()),
        ('高盈亏比策略', HighProfitRatioStrategy()),
    ]
    
    results = []
    
    print("\n" + "-" * 80)
    print(f"{'策略名称':<15} {'胜率':>8} {'盈亏比':>8} {'总收益%':>10} {'最大回撤%':>10} {'夏普比':>8} {'交易数':>8}")
    print("-" * 80)
    
    for name, strategy in strategies:
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
        pl_ratio = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else float('inf')
        
        print(f"{name:<15} {result.win_rate*100:>7.2f}% {pl_ratio:>8.2f} "
              f"{result.total_return_pct*100:>9.2f}% {result.max_drawdown_pct*100:>9.2f}% "
              f"{result.sharpe_ratio:>8.2f} {result.total_trades:>8}")
        
        results.append({
            'name': name,
            'win_rate': result.win_rate,
            'pl_ratio': pl_ratio,
            'total_return': result.total_return_pct,
            'max_drawdown': result.max_drawdown_pct,
            'sharpe': result.sharpe_ratio,
            'trades': result.total_trades,
            'result': result
        })
    
    print("-" * 80)
    
    # 找出最佳策略
    best_by_winrate = max(results, key=lambda x: x['win_rate'])
    best_by_return = max(results, key=lambda x: x['total_return'])
    best_by_sharpe = max(results, key=lambda x: x['sharpe'])
    
    print(f"\n[最佳表现]")
    print(f"   最高胜率: {best_by_winrate['name']} ({best_by_winrate['win_rate']*100:.2f}%)")
    print(f"   最高收益: {best_by_return['name']} ({best_by_return['total_return']*100:.2f}%)")
    print(f"   最高夏普: {best_by_sharpe['name']} ({best_by_sharpe['sharpe']:.2f})")
    
    print("\n" + "=" * 80)
    
    return results


def run_live_trading(config: Config = None, strategy_type: str = 'optimized', interval: int = 3600):
    """
    运行实盘交易
    
    Args:
        config: 配置对象
        strategy_type: 策略类型
        interval: 检查间隔（秒）
    """
    if config is None:
        config = default_config
    
    # 检查API配置
    if not config.exchange.api_key or not config.exchange.secret:
        print("\n" + "=" * 60)
        print("[错误] 请先配置API密钥!")
        print("=" * 60)
        print("\n方法1: 设置环境变量")
        print("  export EXCHANGE_API_KEY='your-api-key'")
        print("  export EXCHANGE_SECRET='your-secret-key'")
        print("  export EXCHANGE_PASSWORD='your-passphrase'")
        print("\n方法2: 创建 .env 文件")
        print("  EXCHANGE_API_KEY=your-api-key")
        print("  EXCHANGE_SECRET=your-secret-key")
        print("  EXCHANGE_PASSWORD=your-passphrase")
        print("\n方法3: 直接修改 config/settings.py")
        print("=" * 60)
        return
    
    # 导入交易模块
    from executor.trade_manager import LiveTrader, TradingConfig, TradeMode
    
    print("\n" + "=" * 60)
    print("[*] 加密货币量化交易系统 - 实盘交易模式")
    print("=" * 60)
    print(f"[日期] 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[模式] {'模拟盘' if config.exchange.sandbox else '实盘'}")
    print(f"[策略] {strategy_type}")
    print(f"[交易对] {config.data.symbol}")
    print(f"[时间框架] {config.data.timeframe}")
    print(f"[检查间隔] {interval}秒")
    print("=" * 60)
    
    # 安全确认
    if not config.exchange.sandbox:
        print("\n[警告] 您正在使用实盘模式，资金将面临真实风险!")
        confirm = input("确认继续? (输入 'YES' 确认): ")
        if confirm != 'YES':
            print("已取消")
            return
    
    # 创建交易配置
    trading_config = TradingConfig(
        symbol=config.data.symbol,
        timeframe=config.data.timeframe,
        trade_mode=TradeMode.CASH,  # 现货模式
        risk_per_trade=config.strategy.risk_per_trade,
        max_position_size=config.strategy.max_position_size,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_daily_trades=10,
        min_trade_interval=300
    )
    
    # 创建交易器
    trader = LiveTrader(
        api_key=config.exchange.api_key,
        secret_key=config.exchange.secret,
        passphrase=config.exchange.password,
        is_demo=config.exchange.sandbox,
        proxy=config.proxy.to_dict(),
        config=trading_config
    )
    
    # 设置数据获取器
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=config.proxy.to_dict(),
        cache_enabled=True
    )
    trader.set_data_fetcher(fetcher)
    
    # 选择策略
    if strategy_type == 'optimized':
        strategy = TrailingStopStrategy()
    elif strategy_type == 'conservative':
        strategy = ConservativeStrategy()
    elif strategy_type == 'aggressive':
        strategy = AggressiveStrategy()
    elif strategy_type == 'optimal':
        strategy = OptimalStrategy()
    else:
        strategy = EmaRsiStrategy()
    
    trader.set_strategy(strategy)
    
    print(f"\n[策略] 使用 {type(strategy).__name__}")
    print("[提示] 按 Ctrl+C 停止交易\n")
    
    try:
        # 启动自动交易
        trader.start(interval=interval)
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n[停止] 收到中断信号...")
        trader.stop()
        print("[OK] 交易已安全停止")


def run_trade_once(config: Config = None, strategy_type: str = 'optimized'):
    """
    执行一次交易检查（不循环）
    
    用于手动触发交易检查
    """
    if config is None:
        config = default_config
    
    if not config.exchange.api_key or not config.exchange.secret:
        print("[错误] 请先配置API密钥!")
        return
    
    from executor.trade_manager import LiveTrader, TradingConfig
    
    print("\n" + "=" * 60)
    print("[*] 执行单次交易检查")
    print("=" * 60)
    
    # 创建交易器
    trading_config = TradingConfig(
        symbol=config.data.symbol,
        timeframe=config.data.timeframe
    )
    
    trader = LiveTrader(
        api_key=config.exchange.api_key,
        secret_key=config.exchange.secret,
        passphrase=config.exchange.password,
        is_demo=config.exchange.sandbox,
        proxy=config.proxy.to_dict(),
        config=trading_config
    )
    
    # 设置数据获取器
    fetcher = DataFetcher(
        exchange_name='okx',
        proxy=config.proxy.to_dict(),
        cache_enabled=True
    )
    trader.set_data_fetcher(fetcher)
    
    # 选择策略
    if strategy_type == 'optimized':
        strategy = TrailingStopStrategy()
    else:
        strategy = EmaRsiStrategy()
    
    trader.set_strategy(strategy)
    
    # 执行一次
    trader.run_once()
    
    print("\n[OK] 检查完成")


def test_api_connection(config: Config = None):
    """测试API连接"""
    if config is None:
        config = default_config
    
    if not config.exchange.api_key or not config.exchange.secret:
        print("[错误] 请先配置API密钥!")
        return
    
    from executor.okx_executor import OKXExecutor
    
    print("\n" + "=" * 60)
    print("[*] 测试OKX API连接")
    print("=" * 60)
    
    executor = OKXExecutor(
        api_key=config.exchange.api_key,
        secret_key=config.exchange.secret,
        passphrase=config.exchange.password,
        is_demo=config.exchange.sandbox,
        proxy=config.proxy.to_dict()
    )
    
    # 测试连接
    if executor.test_connection():
        print("[OK] API连接成功!")
        
        # 打印账户信息
        executor.print_account_summary()
    else:
        print("[FAIL] API连接失败!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币量化交易系统 v4.0')
    parser.add_argument('--mode', type=str, default='backtest', 
                        choices=['backtest', 'optimize', 'factor', 'analysis', 'compare', 
                                 'optimized', 'conservative', 'aggressive',
                                 'live', 'trade-once', 'test-api'],
                        help='运行模式: backtest(原版回测), optimized(优化版), conservative(保守版), '
                             'aggressive(激进版), compare(策略对比), optimize(参数优化), '
                             'factor(因子回测), analysis(因子分析), '
                             'live(实盘交易), trade-once(单次检查), test-api(测试API)')
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
    parser.add_argument('--local', action='store_true',
                        help='使用本地数据（localData目录）')
    parser.add_argument('--online', action='store_true',
                        help='强制使用线上API数据')
    parser.add_argument('--interval', type=int, default=3600,
                        help='实盘交易检查间隔（秒，默认3600）')
    parser.add_argument('--strategy', type=str, default='optimized',
                        choices=['original', 'optimized', 'conservative', 'aggressive', 'optimal'],
                        help='实盘交易策略类型')
    parser.add_argument('--demo', action='store_true',
                        help='使用模拟盘（默认）')
    parser.add_argument('--real', action='store_true',
                        help='使用实盘（危险！）')
    
    args = parser.parse_args()
    
    # 配置 - 优先使用环境变量，否则使用配置文件默认值
    # 先尝试从环境变量加载
    env_api_key = os.getenv('EXCHANGE_API_KEY')
    
    if env_api_key:
        # 如果环境变量存在，使用 from_env()
        config = Config.from_env()
    else:
        # 否则使用配置文件的默认值（这样用户在settings.py中的修改会生效）
        config = default_config
    
    config.data.symbol = args.symbol
    config.data.timeframe = args.timeframe
    config.data.limit = args.limit
    
    # 只有命令行指定了 --capital 才覆盖配置文件
    if args.capital is not None:
        config.backtest.initial_capital = args.capital
    
    if args.no_proxy:
        config.proxy.enabled = False
    
    # 实盘/模拟盘设置（命令行参数优先）
    if args.real:
        config.exchange.sandbox = False
    elif args.demo:
        config.exchange.sandbox = True
    # 如果都没指定，则保持配置文件的设置
    
    # 确保目录存在
    for dir_path in ['logs', 'data/cache']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 确定数据源
    use_local = args.local and not args.online  # --local优先，但--online可覆盖
    
    # 运行
    if args.mode == 'backtest':
        run_backtest(config, show_chart=not args.no_chart, use_factor=False, 
                     strategy_type='original', use_local_data=use_local)
    elif args.mode == 'optimized':
        run_backtest(config, show_chart=not args.no_chart, use_factor=False, 
                     strategy_type='optimized', use_local_data=use_local)
    elif args.mode == 'conservative':
        run_backtest(config, show_chart=not args.no_chart, use_factor=False, 
                     strategy_type='conservative', use_local_data=use_local)
    elif args.mode == 'aggressive':
        run_backtest(config, show_chart=not args.no_chart, use_factor=False, 
                     strategy_type='aggressive', use_local_data=use_local)
    elif args.mode == 'compare':
        compare_strategies(config, use_local_data=use_local)
    elif args.mode == 'factor':
        run_backtest(config, show_chart=not args.no_chart, use_factor=True, use_local_data=use_local)
    elif args.mode == 'optimize':
        optimize_strategy(config)
    elif args.mode == 'analysis':
        run_factor_analysis(config)
    elif args.mode == 'live':
        # 实盘交易模式
        run_live_trading(config, strategy_type=args.strategy, interval=args.interval)
    elif args.mode == 'trade-once':
        # 单次交易检查
        run_trade_once(config, strategy_type=args.strategy)
    elif args.mode == 'test-api':
        # 测试API连接
        test_api_connection(config)


if __name__ == '__main__':
    main()
