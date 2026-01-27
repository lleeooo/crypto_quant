"""
OKX API v5 客户端
实现REST API认证、签名和基础请求功能

API文档: https://www.okx.com/docs-v5/zh/
"""
import hmac
import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode
from decimal import Decimal
import requests


class OKXEnvironment(Enum):
    """OKX环境"""
    PRODUCTION = "https://www.okx.com"
    DEMO = "https://www.okx.com"  # 模拟盘用同一个URL，通过header区分


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"        # 市价单
    LIMIT = "limit"          # 限价单
    POST_ONLY = "post_only"  # 只做maker单
    FOK = "fok"              # 全部成交或立即取消
    IOC = "ioc"              # 立即成交并取消剩余


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"    # 多头
    SHORT = "short"  # 空头
    NET = "net"      # 单向持仓（买卖模式）


class TradeMode(Enum):
    """交易模式"""
    CASH = "cash"        # 现货
    CROSS = "cross"      # 全仓保证金
    ISOLATED = "isolated"  # 逐仓保证金


class InstrumentType(Enum):
    """产品类型"""
    SPOT = "SPOT"        # 现货
    MARGIN = "MARGIN"    # 杠杆
    SWAP = "SWAP"        # 永续合约
    FUTURES = "FUTURES"  # 交割合约
    OPTION = "OPTION"    # 期权


@dataclass
class APICredentials:
    """API凭证"""
    api_key: str
    secret_key: str
    passphrase: str
    is_demo: bool = False  # 是否为模拟盘


class OKXError(Exception):
    """OKX API错误"""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"OKX Error [{code}]: {message}")


class OKXClient:
    """
    OKX REST API v5 客户端
    
    功能:
    - API签名认证
    - 账户信息查询
    - 订单管理
    - 持仓查询
    - 市场数据获取
    
    使用示例:
    ```python
    client = OKXClient(
        api_key="your-api-key",
        secret_key="your-secret-key",
        passphrase="your-passphrase",
        is_demo=True  # 使用模拟盘
    )
    
    # 获取账户余额
    balance = client.get_account_balance()
    
    # 下单
    order = client.place_order(
        inst_id="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        sz="0.001"
    )
    ```
    """
    
    @staticmethod
    def format_number(value: float, precision: int = 8) -> str:
        """
        格式化数字到指定精度（移除尾随零）
        
        Args:
            value: 数值
            precision: 精度（小数位数）
        
        Returns:
            格式化后的字符串
        """
        decimal_value = Decimal(str(value))
        formatted = f"{decimal_value:.{precision}f}"
        # 移除尾随零和小数点
        return formatted.rstrip('0').rstrip('.')
    
    @staticmethod
    def validate_api_credentials(api_key: str, secret: str, passphrase: str) -> tuple[bool, str]:
        """
        验证API凭证是否完整
        
        Returns:
            (是否有效, 错误信息)
        """
        if not api_key or api_key.strip() == '':
            return False, "API Key 不能为空"
        if not secret or secret.strip() == '':
            return False, "Secret Key 不能为空"
        if not passphrase or passphrase.strip() == '':
            return False, "Passphrase 不能为空"
        return True, ""
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = False,
        proxy: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        """
        初始化OKX客户端
        
        Args:
            api_key: API密钥
            secret_key: 私钥
            passphrase: API密码短语
            is_demo: 是否使用模拟盘（True=模拟盘，False=实盘）
            proxy: 代理设置 {"http": "...", "https": "..."}
            timeout: 请求超时时间（秒）
        """
        # 验证凭证
        is_valid, error_msg = self.validate_api_credentials(api_key, secret_key, passphrase)
        if not is_valid:
            raise ValueError(f"API凭证无效: {error_msg}")
        
        self.credentials = APICredentials(api_key, secret_key, passphrase, is_demo)
        self.base_url = OKXEnvironment.PRODUCTION.value
        self.proxy = proxy
        self.timeout = timeout
        self.session = requests.Session()
        
        if proxy:
            self.session.proxies = proxy
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        生成API签名
        
        签名规则: Base64(HMAC-SHA256(timestamp + method + requestPath + body, secretKey))
        
        Args:
            timestamp: ISO格式时间戳
            method: HTTP方法 (GET/POST)
            request_path: 请求路径 (如 /api/v5/account/balance)
            body: 请求体JSON字符串（POST请求时）
        
        Returns:
            Base64编码的签名
        """
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.credentials.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _get_timestamp(self) -> str:
        """获取ISO格式时间戳"""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """
        构建请求头
        
        Args:
            method: HTTP方法
            request_path: 请求路径
            body: 请求体
        
        Returns:
            请求头字典
        """
        timestamp = self._get_timestamp()
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        headers = {
            'OK-ACCESS-KEY': self.credentials.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.credentials.passphrase,
            'Content-Type': 'application/json'
        }
        
        # 模拟盘需要添加特殊header
        if self.credentials.is_demo:
            headers['x-simulated-trading'] = '1'
        
        return headers
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        发送API请求
        
        Args:
            method: HTTP方法
            endpoint: API端点 (如 /api/v5/account/balance)
            params: URL参数（GET请求）
            data: 请求体（POST请求）
        
        Returns:
            API响应数据
        
        Raises:
            OKXError: API返回错误
        """
        url = self.base_url + endpoint
        body = ""
        
        # 构建请求路径（包含查询参数）
        request_path = endpoint
        if params:
            query_string = urlencode(sorted(params.items()))  # 使用urlencode并排序
            request_path = f"{endpoint}?{query_string}"
        
        if data:
            body = json.dumps(data)
        
        headers = self._get_headers(method, request_path, body)
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=self.timeout
                )
            else:
                response = self.session.post(
                    url, 
                    data=body, 
                    headers=headers, 
                    timeout=self.timeout
                )
            
            result = response.json()
            
            # 检查API错误
            if result.get('code') != '0':
                raise OKXError(result.get('code', 'unknown'), result.get('msg', 'Unknown error'))
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise OKXError('network_error', str(e))
    
    # ==================== 账户相关 ====================
    
    def get_account_balance(self, ccy: Optional[str] = None) -> Dict:
        """
        获取账户余额
        
        Args:
            ccy: 币种，不传则返回所有币种
        
        Returns:
            账户余额信息
        
        API: GET /api/v5/account/balance
        """
        params = {}
        if ccy:
            params['ccy'] = ccy
        
        result = self._request('GET', '/api/v5/account/balance', params=params)
        return result.get('data', [])
    
    def get_positions(
        self,
        inst_type: Optional[InstrumentType] = None,
        inst_id: Optional[str] = None
    ) -> List[Dict]:
        """
        获取持仓信息
        
        Args:
            inst_type: 产品类型
            inst_id: 产品ID（如BTC-USDT-SWAP）
        
        Returns:
            持仓列表
        
        API: GET /api/v5/account/positions
        """
        params = {}
        if inst_type:
            params['instType'] = inst_type.value
        if inst_id:
            params['instId'] = inst_id
        
        result = self._request('GET', '/api/v5/account/positions', params=params)
        return result.get('data', [])
    
    def get_account_config(self) -> Dict:
        """
        获取账户配置
        
        Returns:
            账户配置信息
        
        API: GET /api/v5/account/config
        """
        result = self._request('GET', '/api/v5/account/config')
        return result.get('data', [])
    
    def set_leverage(
        self,
        inst_id: str,
        lever: str,
        mgn_mode: TradeMode,
        pos_side: Optional[PositionSide] = None
    ) -> Dict:
        """
        设置杠杆倍数
        
        Args:
            inst_id: 产品ID
            lever: 杠杆倍数
            mgn_mode: 保证金模式
            pos_side: 持仓方向（双向持仓时必填）
        
        Returns:
            设置结果
        
        API: POST /api/v5/account/set-leverage
        """
        data = {
            'instId': inst_id,
            'lever': lever,
            'mgnMode': mgn_mode.value
        }
        if pos_side:
            data['posSide'] = pos_side.value
        
        result = self._request('POST', '/api/v5/account/set-leverage', data=data)
        return result.get('data', [])
    
    # ==================== 交易相关 ====================
    
    def place_order(
        self,
        inst_id: str,
        side: OrderSide,
        order_type: OrderType,
        sz: str,
        td_mode: TradeMode = TradeMode.CASH,
        px: Optional[str] = None,
        pos_side: Optional[PositionSide] = None,
        cl_ord_id: Optional[str] = None,
        reduce_only: bool = False,
        tp_trigger_px: Optional[str] = None,
        tp_ord_px: Optional[str] = None,
        sl_trigger_px: Optional[str] = None,
        sl_ord_px: Optional[str] = None,
        attach_algo_ords: Optional[List[Dict]] = None
    ) -> Dict:
        """
        下单
        
        Args:
            inst_id: 产品ID（如 BTC-USDT, BTC-USDT-SWAP）
            side: 订单方向
            order_type: 订单类型
            sz: 委托数量（现货为币数量，合约为张数）
            td_mode: 交易模式
            px: 委托价格（限价单必填）
            pos_side: 持仓方向（双向持仓必填）
            cl_ord_id: 客户自定义订单ID
            reduce_only: 只减仓
            tp_trigger_px: 止盈触发价
            tp_ord_px: 止盈委托价
            sl_trigger_px: 止损触发价
            sl_ord_px: 止损委托价
        
        Returns:
            订单信息
        
        API: POST /api/v5/trade/order
        """
        data = {
            'instId': inst_id,
            'tdMode': td_mode.value,
            'side': side.value,
            'ordType': order_type.value,
            'sz': sz
        }
        
        if px:
            data['px'] = px
        if pos_side:
            data['posSide'] = pos_side.value
        if cl_ord_id:
            data['clOrdId'] = cl_ord_id
        if reduce_only:
            data['reduceOnly'] = 'true'
        
        # 止盈止损 - 现货不支持直接下单时附加，需要用attachAlgoOrds
        if td_mode == TradeMode.CASH and (tp_trigger_px or sl_trigger_px):
            # 现货使用条件单数组
            if attach_algo_ords is None:
                attach_algo_ords = []
            if tp_trigger_px:
                attach_algo_ords.append({
                    'tpTriggerPx': tp_trigger_px,
                    'tpOrdPx': tp_ord_px or '-1'  # -1表示市价
                })
            if sl_trigger_px:
                attach_algo_ords.append({
                    'slTriggerPx': sl_trigger_px,
                    'slOrdPx': sl_ord_px or '-1'
                })
        else:
            # 合约可以直接使用止盈止损参数
            if tp_trigger_px:
                data['tpTriggerPx'] = tp_trigger_px
            if tp_ord_px:
                data['tpOrdPx'] = tp_ord_px
            if sl_trigger_px:
                data['slTriggerPx'] = sl_trigger_px
            if sl_ord_px:
                data['slOrdPx'] = sl_ord_px
        
        if attach_algo_ords:
            data['attachAlgoOrds'] = attach_algo_ords
        
        result = self._request('POST', '/api/v5/trade/order', data=data)
        return result.get('data', [])
    
    def cancel_order(
        self,
        inst_id: str,
        ord_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None
    ) -> Dict:
        """
        撤单
        
        Args:
            inst_id: 产品ID
            ord_id: 订单ID（ord_id和cl_ord_id至少填一个）
            cl_ord_id: 客户自定义订单ID
        
        Returns:
            撤单结果
        
        API: POST /api/v5/trade/cancel-order
        """
        data = {'instId': inst_id}
        if ord_id:
            data['ordId'] = ord_id
        if cl_ord_id:
            data['clOrdId'] = cl_ord_id
        
        result = self._request('POST', '/api/v5/trade/cancel-order', data=data)
        return result.get('data', [])
    
    def cancel_batch_orders(self, orders: List[Dict]) -> Dict:
        """
        批量撤单
        
        Args:
            orders: 订单列表，每个订单包含instId和ordId或clOrdId
        
        Returns:
            批量撤单结果
        
        API: POST /api/v5/trade/cancel-batch-orders
        """
        result = self._request('POST', '/api/v5/trade/cancel-batch-orders', data=orders)
        return result.get('data', [])
    
    def amend_order(
        self,
        inst_id: str,
        ord_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        new_sz: Optional[str] = None,
        new_px: Optional[str] = None
    ) -> Dict:
        """
        修改订单
        
        Args:
            inst_id: 产品ID
            ord_id: 订单ID
            cl_ord_id: 客户自定义订单ID
            new_sz: 新数量
            new_px: 新价格
        
        Returns:
            修改结果
        
        API: POST /api/v5/trade/amend-order
        """
        data = {'instId': inst_id}
        if ord_id:
            data['ordId'] = ord_id
        if cl_ord_id:
            data['clOrdId'] = cl_ord_id
        if new_sz:
            data['newSz'] = new_sz
        if new_px:
            data['newPx'] = new_px
        
        result = self._request('POST', '/api/v5/trade/amend-order', data=data)
        return result.get('data', [])
    
    def get_order(
        self,
        inst_id: str,
        ord_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None
    ) -> Dict:
        """
        查询订单详情
        
        Args:
            inst_id: 产品ID
            ord_id: 订单ID
            cl_ord_id: 客户自定义订单ID
        
        Returns:
            订单详情
        
        API: GET /api/v5/trade/order
        """
        params = {'instId': inst_id}
        if ord_id:
            params['ordId'] = ord_id
        if cl_ord_id:
            params['clOrdId'] = cl_ord_id
        
        result = self._request('GET', '/api/v5/trade/order', params=params)
        return result.get('data', [])
    
    def get_pending_orders(
        self,
        inst_type: Optional[InstrumentType] = None,
        inst_id: Optional[str] = None,
        order_type: Optional[OrderType] = None,
        state: Optional[str] = None
    ) -> List[Dict]:
        """
        获取未成交订单列表
        
        Args:
            inst_type: 产品类型
            inst_id: 产品ID
            order_type: 订单类型
            state: 订单状态
        
        Returns:
            订单列表
        
        API: GET /api/v5/trade/orders-pending
        """
        params = {}
        if inst_type:
            params['instType'] = inst_type.value
        if inst_id:
            params['instId'] = inst_id
        if order_type:
            params['ordType'] = order_type.value
        if state:
            params['state'] = state
        
        result = self._request('GET', '/api/v5/trade/orders-pending', params=params)
        return result.get('data', [])
    
    def get_orders_history(
        self,
        inst_type: InstrumentType,
        inst_id: Optional[str] = None,
        order_type: Optional[OrderType] = None,
        state: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取历史订单
        
        Args:
            inst_type: 产品类型
            inst_id: 产品ID
            order_type: 订单类型
            state: 订单状态
            limit: 返回数量
        
        Returns:
            订单历史列表
        
        API: GET /api/v5/trade/orders-history
        """
        params = {
            'instType': inst_type.value,
            'limit': str(limit)
        }
        if inst_id:
            params['instId'] = inst_id
        if order_type:
            params['ordType'] = order_type.value
        if state:
            params['state'] = state
        
        result = self._request('GET', '/api/v5/trade/orders-history', params=params)
        return result.get('data', [])
    
    # ==================== 市场数据 ====================
    
    def get_ticker(self, inst_id: str) -> Dict:
        """
        获取单个产品行情
        
        Args:
            inst_id: 产品ID
        
        Returns:
            行情数据
        
        API: GET /api/v5/market/ticker
        """
        result = self._request('GET', '/api/v5/market/ticker', params={'instId': inst_id})
        return result.get('data', [])
    
    def get_tickers(self, inst_type: InstrumentType) -> List[Dict]:
        """
        获取所有产品行情
        
        Args:
            inst_type: 产品类型
        
        Returns:
            行情数据列表
        
        API: GET /api/v5/market/tickers
        """
        result = self._request('GET', '/api/v5/market/tickers', params={'instType': inst_type.value})
        return result.get('data', [])
    
    def get_orderbook(self, inst_id: str, sz: int = 20) -> Dict:
        """
        获取深度数据
        
        Args:
            inst_id: 产品ID
            sz: 深度档位数（1-400）
        
        Returns:
            深度数据
        
        API: GET /api/v5/market/books
        """
        result = self._request('GET', '/api/v5/market/books', params={'instId': inst_id, 'sz': str(sz)})
        return result.get('data', [])
    
    def get_candlesticks(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[List]:
        """
        获取K线数据
        
        Args:
            inst_id: 产品ID
            bar: 时间粒度（1m/3m/5m/15m/30m/1H/2H/4H/6H/12H/1D/1W/1M）
            limit: 返回数量（默认100，最大300）
            after: 请求此时间戳之前的数据
            before: 请求此时间戳之后的数据
        
        Returns:
            K线数据列表 [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
        
        API: GET /api/v5/market/candles
        """
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        if after:
            params['after'] = after
        if before:
            params['before'] = before
        
        result = self._request('GET', '/api/v5/market/candles', params=params)
        return result.get('data', [])
    
    # ==================== 工具方法 ====================
    
    def get_instruments(
        self,
        inst_type: InstrumentType,
        inst_id: Optional[str] = None
    ) -> List[Dict]:
        """
        获取产品信息
        
        Args:
            inst_type: 产品类型
            inst_id: 产品ID
        
        Returns:
            产品信息列表
        
        API: GET /api/v5/public/instruments
        """
        params = {'instType': inst_type.value}
        if inst_id:
            params['instId'] = inst_id
        
        result = self._request('GET', '/api/v5/public/instruments', params=params)
        return result.get('data', [])
    
    def get_system_time(self) -> int:
        """
        获取服务器时间
        
        Returns:
            服务器时间戳（毫秒）
        
        API: GET /api/v5/public/time
        """
        result = self._request('GET', '/api/v5/public/time')
        return int(result.get('data', [{}])[0].get('ts', 0))
    
    def test_connectivity(self) -> bool:
        """
        测试API连接
        
        Returns:
            连接是否成功
        """
        try:
            self.get_system_time()
            return True
        except Exception:
            return False


# 便捷函数
def create_client_from_config(config) -> OKXClient:
    """
    从配置对象创建OKX客户端
    
    Args:
        config: Config对象
    
    Returns:
        OKXClient实例
    """
    return OKXClient(
        api_key=config.exchange.api_key,
        secret_key=config.exchange.secret,
        passphrase=config.exchange.password,
        is_demo=config.exchange.sandbox,
        proxy=config.proxy.to_dict()
    )

