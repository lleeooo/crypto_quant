"""
数据获取模块
特性：
1. 支持多交易所
2. 数据缓存
3. 分批获取大量历史数据
4. 异常处理
"""
import ccxt
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict


class DataFetcher:
    """专业级数据获取器"""
    
    # 时间框架对应的毫秒数
    TIMEFRAME_MS = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
        '1M': 30 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(
        self,
        exchange_name: str = 'okx',
        proxy: Optional[Dict] = None,
        cache_dir: str = 'data/cache',
        cache_enabled: bool = True,
        cache_expiry_hours: int = 1
    ):
        self.exchange_name = exchange_name
        self.proxy = proxy
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_expiry_hours = cache_expiry_hours
        
        # 初始化交易所
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'proxies': proxy
        })
        
        # 创建缓存目录
        if cache_enabled and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        """生成缓存文件路径"""
        cache_key = f"{self.exchange_name}_{symbol}_{timeframe}".replace('/', '_')
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")
    
    def _get_cache_meta_path(self, symbol: str, timeframe: str) -> str:
        """缓存元数据路径"""
        cache_key = f"{self.exchange_name}_{symbol}_{timeframe}".replace('/', '_')
        return os.path.join(self.cache_dir, f"{cache_key}_meta.json")
    
    def _is_cache_valid(self, symbol: str, timeframe: str, required_limit: int) -> bool:
        """检查缓存是否有效"""
        if not self.cache_enabled:
            return False
        
        cache_path = self._get_cache_path(symbol, timeframe)
        meta_path = self._get_cache_meta_path(symbol, timeframe)
        
        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return False
        
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            last_update = datetime.fromisoformat(meta['last_update'])
            if datetime.now() - last_update > timedelta(hours=self.cache_expiry_hours):
                return False
            
            # 检查缓存数据量是否足够
            if meta.get('rows', 0) < required_limit:
                return False
            
            return True
        except Exception:
            return False
    
    def _load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """加载缓存数据"""
        try:
            cache_path = self._get_cache_path(symbol, timeframe)
            return pd.read_parquet(cache_path)
        except Exception:
            return None
    
    def _save_cache(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """保存缓存数据"""
        try:
            cache_path = self._get_cache_path(symbol, timeframe)
            meta_path = self._get_cache_meta_path(symbol, timeframe)
            
            df.to_parquet(cache_path)
            
            meta = {
                'symbol': symbol,
                'timeframe': timeframe,
                'last_update': datetime.now().isoformat(),
                'rows': len(df)
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"[警告] 缓存保存失败: {e}")
    
    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        limit: int = 1000,
        since: Optional[int] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取OHLCV数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架 (1m, 5m, 15m, 1h, 4h, 1d)
            limit: 获取K线数量
            since: 起始时间戳(毫秒)，如果为None则从当前向过去获取
            use_cache: 是否使用缓存
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 检查缓存
        if use_cache and self._is_cache_valid(symbol, timeframe, limit):
            cached_df = self._load_cache(symbol, timeframe)
            if cached_df is not None and len(cached_df) >= limit:
                print(f"[缓存] 使用缓存数据: {symbol} {timeframe} ({len(cached_df)} 根)")
                return cached_df.tail(limit).copy()
        
        print(f"[网络] 从 {self.exchange_name} 获取数据: {symbol} {timeframe} (目标: {limit} 根)...")
        
        try:
            all_data = []
            remaining = limit
            batch_size = 100  # OKX 每次最多返回 100 根（部分接口是 300）
            
            # 获取时间框架的毫秒数
            tf_ms = self.TIMEFRAME_MS.get(timeframe, 60 * 60 * 1000)
            
            # 计算起始时间：从当前时间向过去推
            if since is None:
                # 从当前时间开始，向过去获取
                end_time = int(datetime.now().timestamp() * 1000)
                # 计算需要的起始时间
                start_time = end_time - (limit * tf_ms)
                current_since = start_time
            else:
                current_since = since
                end_time = current_since + (limit * tf_ms)
            
            batch_count = 0
            max_batches = (limit // batch_size) + 10  # 防止无限循环
            
            while remaining > 0 and batch_count < max_batches:
                batch_count += 1
                fetch_limit = min(remaining + 1, batch_size)  # +1 因为可能有边界重复
                
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=current_since, 
                        limit=fetch_limit
                    )
                except Exception as e:
                    print(f"[警告] 批次 {batch_count} 获取失败: {e}")
                    time.sleep(1)
                    continue
                
                if not ohlcv:
                    print(f"[信息] 批次 {batch_count}: 无更多数据")
                    break
                
                all_data.extend(ohlcv)
                
                # 显示进度
                if batch_count % 10 == 0:
                    print(f"[进度] 已获取 {len(all_data)} 根K线...")
                
                # 检查是否到达数据末尾
                if len(ohlcv) < fetch_limit:
                    print(f"[信息] 已到达数据末尾")
                    break
                
                # 下一批从最后一根K线之后开始
                last_timestamp = ohlcv[-1][0]
                current_since = last_timestamp + tf_ms
                
                # 检查是否超过当前时间
                if current_since > int(datetime.now().timestamp() * 1000):
                    break
                
                remaining = limit - len(all_data)
                
                # 交易所速率限制
                time.sleep(0.1)
            
            if not all_data:
                raise Exception("未获取到数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(
                all_data, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 去重（按时间戳）
            df = df.drop_duplicates(subset=['timestamp'])
            
            # 转换时间戳
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df = df.drop(columns=['timestamp'])
            
            # 排序
            df = df.sort_index()
            
            # 只保留需要的数量
            if len(df) > limit:
                df = df.tail(limit)
            
            # 保存缓存
            if use_cache and self.cache_enabled:
                self._save_cache(df, symbol, timeframe)
            
            print(f"[OK] 获取完成: {len(df)} 根K线 (范围: {df.index[0]} ~ {df.index[-1]})")
            return df
            
        except Exception as e:
            print(f"[错误] 数据获取失败: {e}")
            
            # 尝试使用过期缓存
            cached_df = self._load_cache(symbol, timeframe)
            if cached_df is not None:
                print(f"[警告] 使用过期缓存数据 ({len(cached_df)} 根)")
                return cached_df.tail(limit).copy()
            
            raise
    
    def fetch_ohlcv_by_date(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        按日期范围获取OHLCV数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 起始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')，默认为当前时间
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 解析日期
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            since = int(start_dt.timestamp() * 1000)
        else:
            since = None
        
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            end_ts = int(end_dt.timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)
        
        # 计算需要获取的K线数量
        tf_ms = self.TIMEFRAME_MS.get(timeframe, 60 * 60 * 1000)
        if since:
            limit = int((end_ts - since) / tf_ms) + 1
        else:
            limit = 1000
        
        print(f"[信息] 日期范围: {start_date or '最早'} ~ {end_date or '现在'}, 预计 {limit} 根K线")
        
        return self.fetch_ohlcv(symbol, timeframe, limit, since, use_cache=False)
    
    def fetch_multiple_timeframes(
        self,
        symbol: str = 'BTC/USDT',
        timeframes: list = ['1h', '4h', '1d'],
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """获取多时间框架数据"""
        result = {}
        for tf in timeframes:
            try:
                result[tf] = self.fetch_ohlcv(symbol, tf, limit)
            except Exception as e:
                print(f"[警告] {tf} 数据获取失败: {e}")
        return result
    
    def get_available_symbols(self) -> list:
        """获取交易所可用的交易对列表"""
        try:
            self.exchange.load_markets()
            return list(self.exchange.markets.keys())
        except Exception as e:
            print(f"[错误] 获取交易对失败: {e}")
            return []


# ============ 向后兼容的简化接口 ============

def fetch_btc_ohlcv(proxy=None, limit=500):
    """向后兼容的简化接口"""
    fetcher = DataFetcher(proxy=proxy)
    return fetcher.fetch_ohlcv('BTC/USDT', '1h', limit)
