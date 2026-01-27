"""
交易管理器
实现策略信号到实盘交易的转换

功能:
- 实时监控市场数据
- 策略信号执行
- 风险控制
- 交易日志记录
"""
import time
import json
import os
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import pandas as pd

from .okx_executor import OKXExecutor, TradeOrder, Position, OrderStatus
from .okx_client import OKXClient, TradeMode, OrderType, InstrumentType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingState(Enum):
    """交易状态"""
    IDLE = "idle"           # 空闲
    RUNNING = "running"     # 运行中
    PAUSED = "paused"       # 暂停
    STOPPED = "stopped"     # 停止


@dataclass
class TradingConfig:
    """交易配置"""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    trade_mode: TradeMode = TradeMode.CASH
    
    # 仓位控制
    max_position_size: float = 1.0  # 最大仓位占比
    risk_per_trade: float = 0.02    # 每笔交易风险
    
    # 止盈止损
    stop_loss_pct: float = 0.02     # 止损比例
    take_profit_pct: float = 0.04   # 止盈比例
    use_trailing_stop: bool = True   # 使用移动止损
    
    # 交易限制
    max_daily_trades: int = 10       # 每日最大交易次数
    min_trade_interval: int = 300    # 最小交易间隔（秒）
    max_drawdown: float = 0.1        # 最大回撤限制
    
    # 执行设置
    order_type: OrderType = OrderType.MARKET  # 订单类型
    slippage_tolerance: float = 0.001  # 滑点容忍度


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: str
    size: float
    price: float
    order_id: str
    signal_source: str
    pnl: float = 0
    fee: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'price': self.price,
            'order_id': self.order_id,
            'signal_source': self.signal_source,
            'pnl': self.pnl,
            'fee': self.fee
        }


class TradeManager:
    """
    交易管理器
    
    管理策略信号到实盘交易的转换
    
    使用示例:
    ```python
    manager = TradeManager(
        api_key="your-api-key",
        secret_key="your-secret-key",
        passphrase="your-passphrase",
        is_demo=True
    )
    
    # 配置交易参数
    manager.config.symbol = "BTC/USDT"
    manager.config.stop_loss_pct = 0.02
    
    # 执行买入信号
    manager.execute_signal(1, "EMA策略")
    
    # 执行卖出信号
    manager.execute_signal(-1, "EMA策略")
    ```
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        config: Optional[TradingConfig] = None,
        log_dir: str = "logs"
    ):
        """
        初始化交易管理器
        
        Args:
            api_key: API密钥
            secret_key: 私钥
            passphrase: API密码
            is_demo: 是否模拟盘
            proxy: 代理设置
            config: 交易配置
            log_dir: 日志目录
        """
        self.executor = OKXExecutor(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            is_demo=is_demo,
            proxy=proxy
        )
        
        self.config = config or TradingConfig()
        self.log_dir = log_dir
        
        # 交易状态
        self.state = TradingState.IDLE
        self.current_position: Optional[Position] = None
        self.last_trade_time: Optional[datetime] = None
        
        # 统计信息
        self.daily_trades = 0
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        
        # 交易历史
        self.trade_history: List[TradeRecord] = []
        
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logger.info(f"交易管理器初始化完成 [{'模拟盘' if is_demo else '实盘'}]")
    
    def _check_trade_allowed(self) -> tuple[bool, str]:
        """
        检查是否允许交易
        
        Returns:
            (是否允许, 原因)
        """
        # 检查状态
        if self.state != TradingState.RUNNING:
            return False, f"交易状态不正确: {self.state.value}"
        
        # 检查每日交易次数
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"已达每日最大交易次数: {self.config.max_daily_trades}"
        
        # 检查交易间隔
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.config.min_trade_interval:
                return False, f"交易间隔不足: {elapsed:.0f}s < {self.config.min_trade_interval}s"
        
        # 检查回撤
        if self.current_drawdown >= self.config.max_drawdown:
            return False, f"回撤超限: {self.current_drawdown:.2%} >= {self.config.max_drawdown:.2%}"
        
        return True, "允许交易"
    
    def _update_position(self):
        """更新当前持仓状态"""
        inst_id = self.config.symbol.replace('/', '-')
        positions = self.executor.get_positions(inst_id)
        
        if positions:
            self.current_position = positions[0]
        else:
            self.current_position = None
    
    def _update_statistics(self, trade: TradeRecord):
        """更新交易统计"""
        self.daily_trades += 1
        self.total_pnl += trade.pnl
        self.last_trade_time = trade.timestamp
        
        # 更新权益峰值和回撤
        balance = self.executor.get_account_balance()
        current_equity = balance.get('equity', balance.get('total', 0))
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def _save_trade_log(self, trade: TradeRecord):
        """保存交易日志"""
        log_file = os.path.join(self.log_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            # 读取现有日志
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            # 添加新交易
            trades.append(trade.to_dict())
            
            # 保存
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(trades, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存交易日志失败: {e}")
    
    def start(self):
        """启动交易"""
        if self.state == TradingState.RUNNING:
            logger.warning("交易已在运行中")
            return
        
        # 测试连接
        if not self.executor.test_connection():
            logger.error("API连接失败，无法启动交易")
            return
        
        # 初始化统计
        balance = self.executor.get_account_balance()
        self.peak_equity = balance.get('equity', balance.get('total', 0))
        
        # 更新持仓
        self._update_position()
        
        self.state = TradingState.RUNNING
        logger.info("交易已启动")
    
    def stop(self):
        """停止交易"""
        self.state = TradingState.STOPPED
        logger.info("交易已停止")
    
    def pause(self):
        """暂停交易"""
        self.state = TradingState.PAUSED
        logger.info("交易已暂停")
    
    def resume(self):
        """恢复交易"""
        if self.state == TradingState.PAUSED:
            self.state = TradingState.RUNNING
            logger.info("交易已恢复")
    
    def reset_daily_stats(self):
        """重置每日统计"""
        self.daily_trades = 0
        logger.info("每日统计已重置")
    
    def execute_signal(
        self,
        signal: int,
        signal_source: str = "strategy",
        size: Optional[float] = None
    ) -> Optional[TradeOrder]:
        """
        执行交易信号
        
        Args:
            signal: 交易信号 (1=买入/开多, -1=卖出/开空, 0=平仓)
            signal_source: 信号来源
            size: 交易数量（可选）
        
        Returns:
            订单对象
        """
        # 检查是否允许交易
        allowed, reason = self._check_trade_allowed()
        if not allowed:
            logger.warning(f"交易被拒绝: {reason}")
            return None
        
        # 更新持仓状态
        self._update_position()
        
        order = None
        
        if signal == 1:  # 买入/开多
            if self.current_position and self.current_position.is_short:
                # 先平空仓
                logger.info("平空仓后开多")
                self.executor.close_position(self.current_position)
                time.sleep(1)  # 等待成交
            
            if not self.current_position or not self.current_position.is_long:
                order = self.executor.open_long(
                    symbol=self.config.symbol,
                    size=size,
                    stop_loss_pct=self.config.stop_loss_pct,
                    take_profit_pct=self.config.take_profit_pct
                )
        
        elif signal == -1:  # 卖出/开空
            if self.current_position and self.current_position.is_long:
                # 现货模式下卖出持仓
                logger.info("卖出多仓")
                order = self.executor.close_position(self.current_position)
            elif self.config.trade_mode != TradeMode.CASH:
                # 合约模式可以开空
                if self.current_position and self.current_position.is_long:
                    logger.info("平多仓后开空")
                    self.executor.close_position(self.current_position)
                    time.sleep(1)
                
                if not self.current_position or not self.current_position.is_short:
                    order = self.executor.open_short(
                        symbol=self.config.symbol,
                        size=size,
                        stop_loss_pct=self.config.stop_loss_pct,
                        take_profit_pct=self.config.take_profit_pct,
                        trade_mode=self.config.trade_mode
                    )
        
        elif signal == 0:  # 平仓
            if self.current_position:
                order = self.executor.close_position(self.current_position)
        
        # 记录交易
        if order and order.status != OrderStatus.FAILED:
            # 等待订单成交
            order = self.executor.wait_order_filled(order, timeout=30)
            
            trade = TradeRecord(
                timestamp=datetime.now(),
                symbol=self.config.symbol,
                side=order.side.value,
                size=order.filled_size or order.size,
                price=order.avg_price or self.executor.get_current_price(self.config.symbol),
                order_id=order.order_id,
                signal_source=signal_source,
                pnl=order.pnl,
                fee=order.fee
            )
            
            self.trade_history.append(trade)
            self._update_statistics(trade)
            self._save_trade_log(trade)
            
            logger.info(f"交易执行完成: {trade.side} {trade.size} @ {trade.price}")
        
        return order
    
    def get_status(self) -> Dict:
        """获取交易状态"""
        self._update_position()
        balance = self.executor.get_account_balance()
        
        return {
            'state': self.state.value,
            'symbol': self.config.symbol,
            'balance': balance,
            'position': {
                'inst_id': self.current_position.inst_id if self.current_position else None,
                'side': 'long' if self.current_position and self.current_position.is_long else 
                        'short' if self.current_position and self.current_position.is_short else None,
                'size': self.current_position.size if self.current_position else 0,
                'unrealized_pnl': self.current_position.unrealized_pnl if self.current_position else 0
            },
            'statistics': {
                'daily_trades': self.daily_trades,
                'total_pnl': self.total_pnl,
                'current_drawdown': self.current_drawdown,
                'peak_equity': self.peak_equity
            },
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
        }
    
    def print_status(self):
        """打印交易状态"""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("[交易管理器状态]")
        print("=" * 60)
        print(f"运行状态: {status['state']}")
        print(f"交易对: {status['symbol']}")
        print(f"\n[账户余额]")
        print(f"  可用: {status['balance'].get('available', 0):.4f} USDT")
        print(f"  总值: {status['balance'].get('total', 0):.4f} USDT")
        
        print(f"\n[当前持仓]")
        pos = status['position']
        if pos['side']:
            print(f"  方向: {pos['side']}")
            print(f"  数量: {pos['size']}")
            print(f"  浮盈: {pos['unrealized_pnl']:.4f}")
        else:
            print("  无持仓")
        
        print(f"\n[统计信息]")
        stats = status['statistics']
        print(f"  今日交易: {stats['daily_trades']} 次")
        print(f"  总盈亏: {stats['total_pnl']:.4f}")
        print(f"  当前回撤: {stats['current_drawdown']:.2%}")
        
        if status['last_trade_time']:
            print(f"\n最后交易: {status['last_trade_time']}")
        
        print("=" * 60)


class LiveTrader:
    """
    实盘交易器
    
    整合策略和交易管理器，实现自动化实盘交易
    
    使用示例:
    ```python
    from strategies.optimized_strategy import OptimizedStrategy
    
    trader = LiveTrader(
        api_key="your-api-key",
        secret_key="your-secret-key",
        passphrase="your-passphrase",
        is_demo=True
    )
    
    # 设置策略
    strategy = OptimizedStrategy()
    trader.set_strategy(strategy)
    
    # 开始自动交易
    trader.start(interval=3600)  # 每小时检查一次
    ```
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        config: Optional[TradingConfig] = None
    ):
        """
        初始化实盘交易器
        
        Args:
            api_key: API密钥
            secret_key: 私钥
            passphrase: API密码
            is_demo: 是否模拟盘
            proxy: 代理设置
            config: 交易配置
        """
        self.manager = TradeManager(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            is_demo=is_demo,
            proxy=proxy,
            config=config
        )
        
        self.strategy = None
        self.data_fetcher = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info("实盘交易器初始化完成")
    
    def set_strategy(self, strategy):
        """
        设置交易策略
        
        Args:
            strategy: 策略对象（需要有 generate_signals 方法）
        """
        self.strategy = strategy
        logger.info(f"策略已设置: {type(strategy).__name__}")
    
    def set_data_fetcher(self, fetcher):
        """
        设置数据获取器
        
        Args:
            fetcher: DataFetcher对象
        """
        self.data_fetcher = fetcher
        logger.info("数据获取器已设置")
    
    def _fetch_latest_data(self, limit: int = 500) -> pd.DataFrame:
        """获取最新数据"""
        if self.data_fetcher:
            return self.data_fetcher.fetch_ohlcv(
                symbol=self.manager.config.symbol,
                timeframe=self.manager.config.timeframe,
                limit=limit
            )
        else:
            # 使用OKX客户端直接获取
            inst_id = self.manager.config.symbol.replace('/', '-')
            
            # 转换时间框架格式
            tf_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '2h': '2H', '4h': '4H', '1d': '1D'
            }
            bar = tf_map.get(self.manager.config.timeframe, '1H')
            
            candles = self.manager.executor.client.get_candlesticks(
                inst_id=inst_id,
                bar=bar,
                limit=min(limit, 300)
            )
            
            if not candles:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                        'volCcy', 'volCcyQuote', 'confirm']
            )
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            
            # 转换数值列
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 按时间排序（OKX返回的是倒序）
            df = df.sort_index()
            
            return df[['open', 'high', 'low', 'close', 'volume']]
    
    def _trading_loop(self, interval: int):
        """交易循环"""
        while self._running:
            try:
                # 获取最新数据
                logger.info("获取最新数据...")
                df = self._fetch_latest_data()
                
                if df.empty:
                    logger.warning("无法获取数据")
                    time.sleep(interval)
                    continue
                
                # 生成策略信号
                if self.strategy:
                    logger.info("生成策略信号...")
                    df = self.strategy.generate_signals(df)
                    
                    # 获取最新信号
                    latest_signal = df['signal'].iloc[-1] if 'signal' in df.columns else 0
                    
                    # 获取当前价格
                    current_price = df['close'].iloc[-1]
                    
                    logger.info(f"当前价格: {current_price:.2f}, 信号: {latest_signal}")
                    
                    # 执行信号
                    if latest_signal != 0:
                        strategy_name = type(self.strategy).__name__
                        self.manager.execute_signal(int(latest_signal), strategy_name)
                
                # 打印状态
                self.manager.print_status()
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
            
            # 等待下一个周期
            logger.info(f"等待 {interval} 秒后执行下一轮...")
            
            # 分段睡眠，以便能够响应停止信号
            for _ in range(interval):
                if not self._running:
                    break
                time.sleep(1)
    
    def start(self, interval: int = 3600):
        """
        启动自动交易
        
        Args:
            interval: 检查间隔（秒），默认1小时
        """
        if self._running:
            logger.warning("交易已在运行中")
            return
        
        if not self.strategy:
            logger.error("请先设置策略")
            return
        
        # 启动交易管理器
        self.manager.start()
        
        # 启动交易线程
        self._running = True
        self._thread = threading.Thread(target=self._trading_loop, args=(interval,))
        self._thread.daemon = True
        self._thread.start()
        
        logger.info(f"自动交易已启动，检查间隔: {interval}秒")
    
    def stop(self):
        """停止自动交易"""
        self._running = False
        self.manager.stop()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("自动交易已停止")
    
    def run_once(self):
        """执行一次交易检查"""
        if not self.strategy:
            logger.error("请先设置策略")
            return
        
        # 临时启动
        was_running = self.manager.state == TradingState.RUNNING
        if not was_running:
            self.manager.start()
        
        try:
            # 获取数据
            df = self._fetch_latest_data()
            
            if df.empty:
                logger.warning("无法获取数据")
                return
            
            # 生成信号
            df = self.strategy.generate_signals(df)
            latest_signal = df['signal'].iloc[-1] if 'signal' in df.columns else 0
            
            logger.info(f"信号: {latest_signal}")
            
            # 执行
            if latest_signal != 0:
                strategy_name = type(self.strategy).__name__
                self.manager.execute_signal(int(latest_signal), strategy_name)
            
            # 显示状态
            self.manager.print_status()
            
        finally:
            if not was_running:
                self.manager.stop()
    
    def emergency_close_all(self):
        """紧急平仓"""
        logger.warning("执行紧急平仓!")
        
        # 取消所有挂单
        self.manager.executor.cancel_all_orders()
        
        # 平掉所有仓位
        orders = self.manager.executor.close_all_positions()
        
        for order in orders:
            logger.info(f"平仓订单: {order.order_id}")
        
        self.stop()
        logger.warning("紧急平仓完成")


# ============ 便捷函数 ============

def create_trader_from_config(config) -> LiveTrader:
    """
    从配置对象创建实盘交易器
    
    Args:
        config: Config对象
    
    Returns:
        LiveTrader实例
    """
    trading_config = TradingConfig(
        symbol=config.data.symbol,
        timeframe=config.data.timeframe,
        risk_per_trade=config.strategy.risk_per_trade,
        max_position_size=config.strategy.max_position_size,
        stop_loss_pct=config.strategy.atr_stop_multiplier * 0.01,
        take_profit_pct=config.strategy.atr_take_profit_multiplier * 0.01
    )
    
    return LiveTrader(
        api_key=config.exchange.api_key,
        secret_key=config.exchange.secret,
        passphrase=config.exchange.password,
        is_demo=config.exchange.sandbox,
        proxy=config.proxy.to_dict(),
        config=trading_config
    )

