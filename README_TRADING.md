# OKX 自动交易系统使用指南

本系统已集成 OKX 交易所 API v5，支持自动化交易。

## 🚀 快速开始

### 1. 获取API密钥

1. 登录 [OKX](https://www.okx.com)
2. 进入 `个人中心` -> `API`
3. 创建新的API密钥，设置权限：
   - ✅ 读取（必需）
   - ✅ 交易（必需）
   - ❌ 提币（建议关闭）
4. 记录 `API Key`、`Secret Key` 和 `Passphrase`

### 2. 配置API密钥

**方法一：环境变量（推荐）**

```bash
# Linux/Mac
export EXCHANGE_API_KEY='your-api-key'
export EXCHANGE_SECRET='your-secret-key'
export EXCHANGE_PASSWORD='your-passphrase'

# Windows CMD
set EXCHANGE_API_KEY=your-api-key
set EXCHANGE_SECRET=your-secret-key
set EXCHANGE_PASSWORD=your-passphrase

# Windows PowerShell
$env:EXCHANGE_API_KEY="your-api-key"
$env:EXCHANGE_SECRET="your-secret-key"
$env:EXCHANGE_PASSWORD="your-passphrase"
```

**方法二：.env 文件**

复制 `.env.example` 为 `.env` 并填入你的密钥：

```bash
cp .env.example .env
# 编辑 .env 文件填入密钥
```

### 3. 测试API连接

```bash
python main.py --mode test-api
```

成功输出示例：
```
[*] 测试OKX API连接
==================================================
API连接成功
账户模式: 1
持仓模式: long_short_mode
==================================================
[账户摘要]
==================================================
可用余额: 1000.0000 USDT
冻结余额: 0.0000 USDT
总余额: 1000.0000 USDT
[持仓] 无
==================================================
```

## 📊 使用模式

### 模拟盘交易（推荐新手）

```bash
# 启动模拟盘自动交易
python main.py --mode live --demo

# 使用特定策略
python main.py --mode live --demo --strategy optimized

# 设置检查间隔（秒）
python main.py --mode live --demo --interval 1800
```

### 实盘交易（⚠️ 有风险）

```bash
# 启动实盘交易（需要输入确认）
python main.py --mode live --real

# 指定交易对
python main.py --mode live --real --symbol ETH/USDT
```

### 单次检查

```bash
# 执行一次交易检查，不循环
python main.py --mode trade-once
```

## 🛡️ 风险控制

系统内置多重风险控制：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `risk_per_trade` | 2% | 每笔交易最大风险 |
| `max_position_size` | 100% | 最大仓位占比 |
| `stop_loss_pct` | 2% | 止损比例 |
| `take_profit_pct` | 4% | 止盈比例 |
| `max_daily_trades` | 10 | 每日最大交易次数 |
| `min_trade_interval` | 300秒 | 最小交易间隔 |
| `max_drawdown` | 10% | 最大回撤限制 |

## 📁 策略选择

| 策略 | 命令参数 | 特点 |
|------|----------|------|
| 原版EMA+RSI | `--strategy original` | 基础策略 |
| 优化版（移动止损） | `--strategy optimized` | 默认推荐 |
| 保守版 | `--strategy conservative` | 更高胜率 |
| 激进版 | `--strategy aggressive` | 更多交易 |
| 最优版 | `--strategy optimal` | 均衡表现 |

## 📝 日志与监控

交易日志保存在 `logs/` 目录：

- `trades_YYYYMMDD.json` - 每日交易记录
- 控制台实时输出交易状态

## ⚙️ 高级配置

### 修改交易配置

编辑 `config/settings.py`:

```python
@dataclass
class ExchangeConfig:
    name: str = "okx"
    api_key: str = ""
    secret: str = ""
    password: str = ""
    sandbox: bool = True  # True=模拟盘, False=实盘

@dataclass
class StrategyConfig:
    risk_per_trade: float = 0.02  # 每笔风险2%
    max_position_size: float = 1.0  # 最大仓位100%
```

### 代理设置

```python
@dataclass
class ProxyConfig:
    enabled: bool = True
    http: str = "http://127.0.0.1:7890"
    https: str = "http://127.0.0.1:7890"
```

## 🔧 API功能

### OKXClient (底层API)

```python
from executor.okx_client import OKXClient, OrderSide, OrderType

client = OKXClient(
    api_key="xxx",
    secret_key="xxx",
    passphrase="xxx",
    is_demo=True
)

# 获取账户余额
balance = client.get_account_balance()

# 获取行情
ticker = client.get_ticker("BTC-USDT")

# 下单
order = client.place_order(
    inst_id="BTC-USDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    sz="0.001"
)
```

### OKXExecutor (交易执行器)

```python
from executor.okx_executor import OKXExecutor

executor = OKXExecutor(
    api_key="xxx",
    secret_key="xxx",
    passphrase="xxx",
    is_demo=True
)

# 市价买入
order = executor.buy_market("BTC/USDT", size=0.001)

# 带止盈止损的开仓
order = executor.open_long(
    "BTC/USDT",
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)

# 平仓
executor.close_all_positions()
```

### TradeManager (交易管理器)

```python
from executor.trade_manager import TradeManager

manager = TradeManager(
    api_key="xxx",
    secret_key="xxx",
    passphrase="xxx",
    is_demo=True
)

# 启动交易
manager.start()

# 执行信号 (1=买入, -1=卖出, 0=平仓)
manager.execute_signal(1, "EMA策略")

# 查看状态
manager.print_status()
```

### LiveTrader (实盘交易器)

```python
from executor.trade_manager import LiveTrader
from strategies.optimized_strategy import OptimizedStrategy

trader = LiveTrader(
    api_key="xxx",
    secret_key="xxx",
    passphrase="xxx",
    is_demo=True
)

# 设置策略
trader.set_strategy(OptimizedStrategy())

# 启动自动交易 (每小时检查)
trader.start(interval=3600)

# 紧急平仓
trader.emergency_close_all()
```

## ⚠️ 重要提示

1. **先用模拟盘测试** - 确保策略正常工作后再切换实盘
2. **控制仓位** - 不要投入超过你能承受损失的资金
3. **监控运行** - 定期检查交易日志和账户状态
4. **网络稳定** - 确保网络连接稳定，避免断线
5. **API安全** - 不要泄露API密钥，定期更换

## 🆘 常见问题

### Q: API连接失败？

检查：
1. API密钥是否正确
2. 网络是否需要代理
3. IP是否在白名单中

### Q: 下单失败？

检查：
1. 账户余额是否充足
2. API权限是否包含交易
3. 交易对格式是否正确 (BTC/USDT 或 BTC-USDT)

### Q: 策略没有信号？

检查：
1. 市场条件可能不满足策略入场条件
2. 增加数据量 `--limit 2000`
3. 尝试不同的时间框架

## 📚 更多资源

- [OKX API文档](https://www.okx.com/docs-v5/zh/)
- [项目README](./README.md)
- [分析报告说明](./README_ANALYSIS.md)

