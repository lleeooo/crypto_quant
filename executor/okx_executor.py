"""
OKX交易执行器
封装交易逻辑，提供高级交易功能

功能:
- 智能下单（自动计算仓位大小）
- 止盈止损管理
- 订单状态跟踪
- 风险控制
"""
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .okx_client import (
    OKXClient, OKXError, OrderSide, OrderType, 
    PositionSide, TradeMode, InstrumentType
)


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"      # 等待中
    OPEN = "open"            # 已挂单
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"        # 完全成交
    CANCELED = "canceled"    # 已取消
    FAILED = "failed"        # 失败


@dataclass
class TradeOrder:
    """交易订单"""
    order_id: str
    client_order_id: str
    inst_id: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    pnl: float = 0.0
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    error_msg: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'inst_id': self.inst_id,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'size': self.size,
            'price': self.price,
            'status': self.status.value,
            'filled_size': self.filled_size,
            'avg_price': self.avg_price,
            'fee': self.fee,
            'pnl': self.pnl,
            'create_time': self.create_time.isoformat(),
            'update_time': self.update_time.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'error_msg': self.error_msg
        }


@dataclass
class Position:
    """持仓信息"""
    inst_id: str
    side: PositionSide
    size: float
    avg_price: float
    unrealized_pnl: float
    margin: float
    leverage: int
    liquidation_price: float
    
    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG or (self.side == PositionSide.NET and self.size > 0)
    
    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT or (self.side == PositionSide.NET and self.size < 0)


class OKXExecutor:
    """
    OKX交易执行器
    
    提供高级交易功能:
    - 智能仓位计算
    - 自动止盈止损
    - 订单管理
    - 风险控制
    
    使用示例:
    ```python
    from executor import OKXExecutor
    
    executor = OKXExecutor(
        api_key="your-api-key",
        secret_key="your-secret-key",
        passphrase="your-passphrase",
        is_demo=True
    )
    
    # 市价买入
    order = executor.buy_market("BTC-USDT", size=0.001)
    
    # 带止盈止损的限价单
    order = executor.buy_limit(
        "BTC-USDT",
        size=0.001,
        price=40000,
        stop_loss=39000,
        take_profit=42000
    )
    
    # 查看持仓
    positions = executor.get_positions()
    ```
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        max_position_size: float = 1.0,
        risk_per_trade: float = 0.02,
        default_leverage: int = 1
    ):
        """
        初始化执行器
        
        Args:
            api_key: API密钥
            secret_key: 私钥
            passphrase: API密码
            is_demo: 是否模拟盘
            proxy: 代理设置
            max_position_size: 最大仓位比例（占账户总值）
            risk_per_trade: 每笔交易风险比例
            default_leverage: 默认杠杆倍数
        """
        self.client = OKXClient(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            is_demo=is_demo,
            proxy=proxy
        )
        
        self.is_demo = is_demo
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.default_leverage = default_leverage
        
        # 订单跟踪
        self.orders: Dict[str, TradeOrder] = {}
        
        # 产品信息缓存
        self._instrument_cache: Dict[str, Dict] = {}
        
        logger.info(f"OKX执行器初始化完成 [{'模拟盘' if is_demo else '实盘'}]")
    
    def _generate_client_order_id(self) -> str:
        """生成客户端订单ID"""
        return f"quant_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def _convert_symbol(self, symbol: str) -> str:
        """
        转换交易对格式
        
        BTC/USDT -> BTC-USDT
        """
        return symbol.replace('/', '-')
    
    def _get_instrument_info(self, inst_id: str) -> Dict:
        """获取产品信息（带缓存）"""
        if inst_id not in self._instrument_cache:
            # 判断产品类型
            if 'SWAP' in inst_id:
                inst_type = InstrumentType.SWAP
            elif 'FUTURES' in inst_id:
                inst_type = InstrumentType.FUTURES
            else:
                inst_type = InstrumentType.SPOT
            
            instruments = self.client.get_instruments(inst_type, inst_id)
            if instruments:
                self._instrument_cache[inst_id] = instruments[0]
        
        return self._instrument_cache.get(inst_id, {})
    
    def _parse_order_response(self, response: List[Dict], order: TradeOrder) -> TradeOrder:
        """解析订单响应"""
        if response and len(response) > 0:
            data = response[0]
            order.order_id = data.get('ordId', '')
            
            # 检查下单是否成功
            s_code = data.get('sCode', '0')
            if s_code != '0':
                order.status = OrderStatus.FAILED
                order.error_msg = data.get('sMsg', 'Unknown error')
            else:
                order.status = OrderStatus.OPEN
        
        return order
    
    def _parse_order_status(self, data: Dict) -> OrderStatus:
        """解析订单状态"""
        state = data.get('state', '')
        status_map = {
            'live': OrderStatus.OPEN,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'canceling': OrderStatus.CANCELED
        }
        return status_map.get(state, OrderStatus.PENDING)
    
    # ==================== 账户查询 ====================
    
    def get_account_balance(self, currency: str = "USDT") -> Dict:
        """
        获取账户余额
        
        Args:
            currency: 币种
        
        Returns:
            余额信息
        """
        try:
            balances = self.client.get_account_balance(currency)
            if balances and len(balances) > 0:
                details = balances[0].get('details', [])
                for detail in details:
                    if detail.get('ccy') == currency:
                        return {
                            'currency': currency,
                            'available': float(detail.get('availBal', 0)),
                            'frozen': float(detail.get('frozenBal', 0)),
                            'total': float(detail.get('cashBal', 0)),
                            'equity': float(detail.get('eqUsd', 0))
                        }
            return {
                'currency': currency,
                'available': 0,
                'frozen': 0,
                'total': 0,
                'equity': 0
            }
        except OKXError as e:
            logger.error(f"获取余额失败: {e}")
            return {}
    
    def get_positions(self, inst_id: Optional[str] = None) -> List[Position]:
        """
        获取持仓列表
        
        Args:
            inst_id: 产品ID（可选）
        
        Returns:
            持仓列表
        """
        try:
            positions_data = self.client.get_positions(inst_id=inst_id)
            positions = []
            
            for pos in positions_data:
                if float(pos.get('pos', 0)) != 0:
                    side_str = pos.get('posSide', 'net')
                    side_map = {
                        'long': PositionSide.LONG,
                        'short': PositionSide.SHORT,
                        'net': PositionSide.NET
                    }
                    
                    positions.append(Position(
                        inst_id=pos.get('instId', ''),
                        side=side_map.get(side_str, PositionSide.NET),
                        size=float(pos.get('pos', 0)),
                        avg_price=float(pos.get('avgPx', 0)),
                        unrealized_pnl=float(pos.get('upl', 0)),
                        margin=float(pos.get('margin', 0)),
                        leverage=int(float(pos.get('lever', 1))),
                        liquidation_price=float(pos.get('liqPx', 0)) if pos.get('liqPx') else 0
                    ))
            
            return positions
        except OKXError as e:
            logger.error(f"获取持仓失败: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """
        获取当前价格
        
        Args:
            symbol: 交易对
        
        Returns:
            当前价格
        """
        inst_id = self._convert_symbol(symbol)
        try:
            ticker = self.client.get_ticker(inst_id)
            if ticker and len(ticker) > 0:
                return float(ticker[0].get('last', 0))
        except OKXError as e:
            logger.error(f"获取价格失败: {e}")
        return 0
    
    # ==================== 下单功能 ====================
    
    def buy_market(
        self,
        symbol: str,
        size: float,
        trade_mode: TradeMode = TradeMode.CASH,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeOrder:
        """
        市价买入
        
        Args:
            symbol: 交易对（如 BTC/USDT 或 BTC-USDT）
            size: 数量
            trade_mode: 交易模式
            stop_loss: 止损价
            take_profit: 止盈价
        
        Returns:
            订单对象
        """
        return self._place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=size,
            trade_mode=trade_mode,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def sell_market(
        self,
        symbol: str,
        size: float,
        trade_mode: TradeMode = TradeMode.CASH,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeOrder:
        """
        市价卖出
        
        Args:
            symbol: 交易对
            size: 数量
            trade_mode: 交易模式
            stop_loss: 止损价
            take_profit: 止盈价
        
        Returns:
            订单对象
        """
        return self._place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=size,
            trade_mode=trade_mode,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def buy_limit(
        self,
        symbol: str,
        size: float,
        price: float,
        trade_mode: TradeMode = TradeMode.CASH,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeOrder:
        """
        限价买入
        
        Args:
            symbol: 交易对
            size: 数量
            price: 价格
            trade_mode: 交易模式
            stop_loss: 止损价
            take_profit: 止盈价
        
        Returns:
            订单对象
        """
        return self._place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            trade_mode=trade_mode,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def sell_limit(
        self,
        symbol: str,
        size: float,
        price: float,
        trade_mode: TradeMode = TradeMode.CASH,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeOrder:
        """
        限价卖出
        
        Args:
            symbol: 交易对
            size: 数量
            price: 价格
            trade_mode: 交易模式
            stop_loss: 止损价
            take_profit: 止盈价
        
        Returns:
            订单对象
        """
        return self._place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            trade_mode=trade_mode,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def _place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: Optional[float] = None,
        trade_mode: TradeMode = TradeMode.CASH,
        pos_side: Optional[PositionSide] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeOrder:
        """
        下单核心方法
        
        Args:
            symbol: 交易对
            side: 方向
            order_type: 订单类型
            size: 数量
            price: 价格
            trade_mode: 交易模式
            pos_side: 持仓方向
            stop_loss: 止损价
            take_profit: 止盈价
        
        Returns:
            订单对象
        """
        inst_id = self._convert_symbol(symbol)
        client_order_id = self._generate_client_order_id()
        
        # 创建订单对象
        order = TradeOrder(
            order_id='',
            client_order_id=client_order_id,
            inst_id=inst_id,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        try:
            # 记录下单日志
            logger.info(f"下单: {inst_id} {side.value} {order_type.value} "
                       f"数量={size} 价格={price or '市价'}")
            
            # 调用API下单
            response = self.client.place_order(
                inst_id=inst_id,
                side=side,
                order_type=order_type,
                sz=str(size),
                td_mode=trade_mode,
                px=str(price) if price else None,
                pos_side=pos_side,
                cl_ord_id=client_order_id,
                sl_trigger_px=str(stop_loss) if stop_loss else None,
                sl_ord_px=str(stop_loss) if stop_loss else None,
                tp_trigger_px=str(take_profit) if take_profit else None,
                tp_ord_px=str(take_profit) if take_profit else None
            )
            
            # 解析响应
            order = self._parse_order_response(response, order)
            
            if order.status == OrderStatus.OPEN:
                logger.info(f"下单成功: {order.order_id}")
            else:
                logger.error(f"下单失败: {order.error_msg}")
            
        except OKXError as e:
            order.status = OrderStatus.FAILED
            order.error_msg = str(e)
            logger.error(f"下单异常: {e}")
        
        # 保存订单
        self.orders[order.client_order_id] = order
        
        return order
    
    # ==================== 订单管理 ====================
    
    def cancel_order(self, order: TradeOrder) -> bool:
        """
        取消订单
        
        Args:
            order: 订单对象
        
        Returns:
            是否成功
        """
        try:
            self.client.cancel_order(
                inst_id=order.inst_id,
                ord_id=order.order_id,
                cl_ord_id=order.client_order_id
            )
            order.status = OrderStatus.CANCELED
            order.update_time = datetime.now()
            logger.info(f"订单已取消: {order.order_id}")
            return True
        except OKXError as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        取消所有订单
        
        Args:
            symbol: 交易对（可选，不传则取消所有）
        
        Returns:
            取消的订单数量
        """
        inst_id = self._convert_symbol(symbol) if symbol else None
        
        try:
            pending_orders = self.client.get_pending_orders(inst_id=inst_id)
            canceled_count = 0
            
            for order_data in pending_orders:
                try:
                    self.client.cancel_order(
                        inst_id=order_data['instId'],
                        ord_id=order_data['ordId']
                    )
                    canceled_count += 1
                except OKXError:
                    pass
            
            logger.info(f"已取消 {canceled_count} 个订单")
            return canceled_count
        except OKXError as e:
            logger.error(f"取消订单失败: {e}")
            return 0
    
    def get_order_status(self, order: TradeOrder) -> TradeOrder:
        """
        查询订单状态
        
        Args:
            order: 订单对象
        
        Returns:
            更新后的订单对象
        """
        try:
            order_data = self.client.get_order(
                inst_id=order.inst_id,
                ord_id=order.order_id,
                cl_ord_id=order.client_order_id
            )
            
            if order_data and len(order_data) > 0:
                data = order_data[0]
                order.status = self._parse_order_status(data)
                order.filled_size = float(data.get('accFillSz', 0))
                order.avg_price = float(data.get('avgPx', 0)) if data.get('avgPx') else 0
                order.fee = float(data.get('fee', 0))
                order.pnl = float(data.get('pnl', 0))
                order.update_time = datetime.now()
            
        except OKXError as e:
            logger.error(f"查询订单失败: {e}")
        
        return order
    
    def wait_order_filled(self, order: TradeOrder, timeout: int = 60) -> TradeOrder:
        """
        等待订单成交
        
        Args:
            order: 订单对象
            timeout: 超时时间（秒）
        
        Returns:
            更新后的订单对象
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            order = self.get_order_status(order)
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.FAILED]:
                break
            
            time.sleep(1)
        
        return order
    
    # ==================== 智能交易 ====================
    
    def calculate_position_size(
        self,
        symbol: str,
        stop_loss_pct: float = 0.02,
        account_risk_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算仓位大小（基于风险管理）
        
        Args:
            symbol: 交易对
            stop_loss_pct: 止损比例
            account_risk_pct: 账户风险比例（默认使用self.risk_per_trade）
        
        Returns:
            (仓位大小, 止损价)
        """
        if account_risk_pct is None:
            account_risk_pct = self.risk_per_trade
        
        # 获取账户余额
        balance = self.get_account_balance()
        available = balance.get('available', 0)
        
        if available <= 0:
            logger.warning("账户余额不足")
            return 0, 0
        
        # 获取当前价格
        current_price = self.get_current_price(symbol)
        if current_price <= 0:
            logger.warning("无法获取当前价格")
            return 0, 0
        
        # 计算止损价
        stop_loss = current_price * (1 - stop_loss_pct)
        
        # 计算风险金额
        risk_amount = available * account_risk_pct
        
        # 计算仓位大小
        # 仓位 = 风险金额 / (入场价 - 止损价)
        price_risk = current_price - stop_loss
        if price_risk <= 0:
            logger.warning("止损设置无效")
            return 0, 0
        
        position_size = risk_amount / price_risk
        
        # 限制最大仓位
        max_size = (available * self.max_position_size) / current_price
        position_size = min(position_size, max_size)
        
        logger.info(f"计算仓位: {symbol} 价格={current_price:.2f} "
                   f"仓位={position_size:.6f} 止损={stop_loss:.2f}")
        
        return position_size, stop_loss
    
    def open_long(
        self,
        symbol: str,
        size: Optional[float] = None,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        order_type: OrderType = OrderType.MARKET
    ) -> TradeOrder:
        """
        开多仓
        
        Args:
            symbol: 交易对
            size: 仓位大小（不传则自动计算）
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
            order_type: 订单类型
        
        Returns:
            订单对象
        """
        current_price = self.get_current_price(symbol)
        
        # 计算仓位
        if size is None:
            size, stop_loss = self.calculate_position_size(symbol, stop_loss_pct)
        else:
            stop_loss = current_price * (1 - stop_loss_pct)
        
        # 计算止盈
        take_profit = current_price * (1 + take_profit_pct)
        
        if size <= 0:
            order = TradeOrder(
                order_id='',
                client_order_id=self._generate_client_order_id(),
                inst_id=self._convert_symbol(symbol),
                side=OrderSide.BUY,
                order_type=order_type,
                size=0,
                status=OrderStatus.FAILED,
                error_msg="仓位计算失败"
            )
            return order
        
        logger.info(f"开多仓: {symbol} 数量={size:.6f} "
                   f"止损={stop_loss:.2f} 止盈={take_profit:.2f}")
        
        return self.buy_market(
            symbol=symbol,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def open_short(
        self,
        symbol: str,
        size: Optional[float] = None,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        trade_mode: TradeMode = TradeMode.CROSS,
        order_type: OrderType = OrderType.MARKET
    ) -> TradeOrder:
        """
        开空仓（合约）
        
        Args:
            symbol: 交易对
            size: 仓位大小
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
            trade_mode: 交易模式
            order_type: 订单类型
        
        Returns:
            订单对象
        """
        current_price = self.get_current_price(symbol)
        
        # 计算止损止盈（空头方向相反）
        stop_loss = current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 - take_profit_pct)
        
        if size is None:
            # 空头仓位计算
            balance = self.get_account_balance()
            available = balance.get('available', 0)
            risk_amount = available * self.risk_per_trade
            price_risk = stop_loss - current_price
            if price_risk > 0:
                size = risk_amount / price_risk
            else:
                size = 0
        
        if size <= 0:
            order = TradeOrder(
                order_id='',
                client_order_id=self._generate_client_order_id(),
                inst_id=self._convert_symbol(symbol),
                side=OrderSide.SELL,
                order_type=order_type,
                size=0,
                status=OrderStatus.FAILED,
                error_msg="仓位计算失败"
            )
            return order
        
        logger.info(f"开空仓: {symbol} 数量={size:.6f} "
                   f"止损={stop_loss:.2f} 止盈={take_profit:.2f}")
        
        return self.sell_market(
            symbol=symbol,
            size=size,
            trade_mode=trade_mode,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def close_position(self, position: Position) -> TradeOrder:
        """
        平仓
        
        Args:
            position: 持仓对象
        
        Returns:
            订单对象
        """
        if position.is_long:
            return self.sell_market(
                symbol=position.inst_id,
                size=abs(position.size)
            )
        else:
            return self.buy_market(
                symbol=position.inst_id,
                size=abs(position.size)
            )
    
    def close_all_positions(self) -> List[TradeOrder]:
        """
        平掉所有仓位
        
        Returns:
            订单列表
        """
        positions = self.get_positions()
        orders = []
        
        for position in positions:
            order = self.close_position(position)
            orders.append(order)
        
        return orders
    
    # ==================== 测试功能 ====================
    
    def test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            是否连接成功
        """
        try:
            is_connected = self.client.test_connectivity()
            if is_connected:
                logger.info("API连接成功")
                
                # 获取账户配置
                config = self.client.get_account_config()
                if config:
                    logger.info(f"账户模式: {config[0].get('acctLv', 'unknown')}")
                    logger.info(f"持仓模式: {config[0].get('posMode', 'unknown')}")
            else:
                logger.error("API连接失败")
            
            return is_connected
        except Exception as e:
            logger.error(f"连接测试异常: {e}")
            return False
    
    def print_account_summary(self):
        """打印账户摘要"""
        print("\n" + "=" * 50)
        print("[账户摘要]")
        print("=" * 50)
        
        # 余额
        balance = self.get_account_balance()
        print(f"可用余额: {balance.get('available', 0):.4f} USDT")
        print(f"冻结余额: {balance.get('frozen', 0):.4f} USDT")
        print(f"总余额: {balance.get('total', 0):.4f} USDT")
        
        # 持仓
        positions = self.get_positions()
        if positions:
            print("\n[持仓]")
            for pos in positions:
                direction = "多" if pos.is_long else "空"
                print(f"  {pos.inst_id}: {direction} {abs(pos.size):.6f} @ {pos.avg_price:.2f} "
                      f"浮盈: {pos.unrealized_pnl:.4f}")
        else:
            print("\n[持仓] 无")
        
        print("=" * 50)

