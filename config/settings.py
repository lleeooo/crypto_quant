"""
量化系统配置文件
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import os


@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str = "okx"
    api_key: str = ""
    secret: str = ""
    password: str = ""  # OKX需要
    sandbox: bool = True  # 沙盒模式
    

@dataclass
class ProxyConfig:
    """代理配置"""
    enabled: bool = True
    http: str = "http://127.0.0.1:7890"
    https: str = "http://127.0.0.1:7890"
    
    def to_dict(self) -> Optional[Dict]:
        if not self.enabled:
            return None
        return {
            'http': self.http,
            'https': self.https
        }


@dataclass
class DataConfig:
    """数据配置"""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 1000
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    cache_expiry_hours: int = 1  # 缓存过期时间


@dataclass
class StrategyConfig:
    """策略配置"""
    # EMA参数
    ema_fast: int = 20
    ema_slow: int = 50
    
    # RSI参数
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    
    # ATR参数
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    atr_take_profit_multiplier: float = 3.0
    
    # 仓位管理
    risk_per_trade: float = 0.02  # 每笔交易风险比例
    max_position_size: float = 1.0  # 最大仓位比例
    
    # 过滤器
    use_volume_filter: bool = True
    volume_ma_period: int = 20
    min_signal_strength: float = 0.2


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000
    fee_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    use_stop_loss: bool = True
    use_take_profit: bool = True


@dataclass
class LogConfig:
    """日志配置"""
    enabled: bool = True
    level: str = "INFO"
    log_dir: str = "logs"
    log_to_console: bool = True
    log_to_file: bool = True


@dataclass
class Config:
    """主配置类"""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        config = cls()
        
        # 交易所配置
        config.exchange.api_key = os.getenv('EXCHANGE_API_KEY', '')
        config.exchange.secret = os.getenv('EXCHANGE_SECRET', '')
        config.exchange.password = os.getenv('EXCHANGE_PASSWORD', '')
        
        # 代理配置
        proxy_enabled = os.getenv('PROXY_ENABLED', 'true').lower() == 'true'
        config.proxy.enabled = proxy_enabled
        config.proxy.http = os.getenv('HTTP_PROXY', 'http://127.0.0.1:7890')
        config.proxy.https = os.getenv('HTTPS_PROXY', 'http://127.0.0.1:7890')
        
        return config


# 默认配置实例
default_config = Config()

