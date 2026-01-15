"""
因子计算库 - 加密货币现货专用
包含：动量因子、波动率因子、成交量因子、趋势因子、反转因子

因子命名规则：
- 因子值为正 → 看多信号
- 因子值为负 → 看空信号
- 因子值接近0 → 中性
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FactorCalculator:
    """因子计算器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化因子计算器
        
        Args:
            df: 包含 open, high, low, close, volume 的DataFrame
        """
        self.df = df.copy()
        self.factors = pd.DataFrame(index=df.index)
    
    # ==================== 动量因子 ====================
    
    def momentum_return(self, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        收益率动量因子
        过去N期的累计收益率
        """
        for p in periods:
            self.factors[f'MOM_{p}'] = self.df['close'].pct_change(p)
        return self.factors
    
    def momentum_rank(self, period: int = 20) -> pd.Series:
        """
        动量排名因子
        过去N期收益在滚动窗口中的排名（0-1）
        """
        returns = self.df['close'].pct_change(period)
        rank = returns.rolling(period * 2).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )
        self.factors['MOM_RANK'] = rank
        return rank
    
    def roc(self, period: int = 12) -> pd.Series:
        """
        变动率 (Rate of Change)
        """
        roc = (self.df['close'] - self.df['close'].shift(period)) / self.df['close'].shift(period) * 100
        self.factors[f'ROC_{period}'] = roc
        return roc
    
    def acceleration(self, period: int = 10) -> pd.Series:
        """
        动量加速度因子
        动量的变化率，用于捕捉趋势加速
        """
        mom = self.df['close'].pct_change(period)
        acc = mom.diff(period)
        self.factors['ACCELERATION'] = acc
        return acc
    
    # ==================== 波动率因子 ====================
    
    def volatility(self, periods: List[int] = [10, 20, 60]) -> pd.DataFrame:
        """
        历史波动率因子
        收益率的标准差（年化）
        """
        returns = self.df['close'].pct_change()
        for p in periods:
            vol = returns.rolling(p).std() * np.sqrt(365 * 24)  # 假设1小时K线
            self.factors[f'VOL_{p}'] = vol
        return self.factors
    
    def volatility_ratio(self, short: int = 10, long: int = 60) -> pd.Series:
        """
        波动率比率因子
        短期波动率 / 长期波动率
        >1 表示波动扩张，<1 表示波动收缩
        """
        returns = self.df['close'].pct_change()
        vol_short = returns.rolling(short).std()
        vol_long = returns.rolling(long).std()
        ratio = vol_short / vol_long
        self.factors['VOL_RATIO'] = ratio
        return ratio
    
    def atr_factor(self, period: int = 14) -> pd.Series:
        """
        ATR百分比因子
        ATR占价格的百分比，衡量相对波动
        """
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        atr_pct = atr / self.df['close'] * 100
        self.factors['ATR_PCT'] = atr_pct
        return atr_pct
    
    def parkinson_vol(self, period: int = 20) -> pd.Series:
        """
        Parkinson波动率
        使用最高价和最低价估计波动率，比收盘价更准确
        """
        log_hl = np.log(self.df['high'] / self.df['low'])
        parkinson = np.sqrt(1 / (4 * np.log(2)) * (log_hl ** 2).rolling(period).mean())
        self.factors['PARKINSON_VOL'] = parkinson
        return parkinson
    
    # ==================== 成交量因子 ====================
    
    def volume_momentum(self, period: int = 20) -> pd.Series:
        """
        成交量动量因子
        当前成交量相对于过去N期均值的偏离
        """
        vol_ma = self.df['volume'].rolling(period).mean()
        vol_mom = (self.df['volume'] - vol_ma) / vol_ma
        self.factors['VOL_MOM'] = vol_mom
        return vol_mom
    
    def price_volume_trend(self, period: int = 20) -> pd.Series:
        """
        价量趋势因子
        价格变化 * 成交量变化，正值表示量价齐升/齐跌
        """
        price_change = self.df['close'].pct_change()
        vol_change = self.df['volume'].pct_change()
        pvt = (price_change * vol_change).rolling(period).sum()
        self.factors['PVT'] = pvt
        return pvt
    
    def obv_slope(self, period: int = 20) -> pd.Series:
        """
        OBV斜率因子
        能量潮的变化趋势
        """
        obv = pd.Series(index=self.df.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.df['volume'].iloc[i]
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # OBV斜率（标准化）
        obv_slope = obv.diff(period) / obv.rolling(period).std()
        self.factors['OBV_SLOPE'] = obv_slope
        return obv_slope
    
    def vwap_deviation(self, period: int = 20) -> pd.Series:
        """
        VWAP偏离因子
        当前价格与VWAP的偏离程度
        正值表示价格高于VWAP，负值表示低于
        """
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (tp * self.df['volume']).rolling(period).sum() / self.df['volume'].rolling(period).sum()
        deviation = (self.df['close'] - vwap) / vwap * 100
        self.factors['VWAP_DEV'] = deviation
        return deviation
    
    def mfi(self, period: int = 14) -> pd.Series:
        """
        资金流量指数 (Money Flow Index)
        类似RSI，但考虑成交量
        """
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_money_flow = tp * self.df['volume']
        
        positive_flow = raw_money_flow.where(tp > tp.shift(), 0).rolling(period).sum()
        negative_flow = raw_money_flow.where(tp < tp.shift(), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.nan)))
        # 标准化到 -1 到 1
        mfi_normalized = (mfi - 50) / 50
        self.factors['MFI'] = mfi_normalized
        return mfi_normalized
    
    # ==================== 趋势因子 ====================
    
    def trend_strength(self, period: int = 20) -> pd.Series:
        """
        趋势强度因子
        使用线性回归的R²衡量趋势的确定性
        """
        def calc_r2(prices):
            if len(prices) < 2:
                return 0
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            fitted = slope * x + intercept
            ss_res = np.sum((prices - fitted) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            if ss_tot == 0:
                return 0
            r2 = 1 - (ss_res / ss_tot)
            # 用斜率的符号表示方向
            return r2 * np.sign(slope)
        
        trend = self.df['close'].rolling(period).apply(calc_r2, raw=True)
        self.factors['TREND_STRENGTH'] = trend
        return trend
    
    def ema_cross_strength(self, fast: int = 12, slow: int = 26) -> pd.Series:
        """
        EMA交叉强度因子
        快慢EMA的差值，标准化
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        diff = ema_fast - ema_slow
        atr = self._calc_atr(14)
        
        strength = diff / atr
        self.factors['EMA_CROSS'] = strength
        return strength
    
    def adx_factor(self, period: int = 14) -> pd.Series:
        """
        ADX趋势因子
        结合趋势强度和方向
        """
        plus_dm = self.df['high'].diff()
        minus_dm = -self.df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._calc_atr(period)
        
        plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)
        
        # 方向由DI差值决定，强度由ADX决定
        direction = np.sign(plus_di - minus_di)
        adx_factor = (adx / 50) * direction  # 归一化并加方向
        
        self.factors['ADX_FACTOR'] = adx_factor
        return adx_factor
    
    def price_channel_position(self, period: int = 20) -> pd.Series:
        """
        价格通道位置因子
        当前价格在N期高低通道中的相对位置
        1表示在最高点，-1表示在最低点
        """
        high_max = self.df['high'].rolling(period).max()
        low_min = self.df['low'].rolling(period).min()
        
        position = 2 * (self.df['close'] - low_min) / (high_max - low_min) - 1
        self.factors['CHANNEL_POS'] = position
        return position
    
    # ==================== 反转因子 ====================
    
    def rsi_factor(self, period: int = 14) -> pd.Series:
        """
        RSI反转因子
        极端RSI值预示反转
        """
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # 转换为因子：超买为负，超卖为正
        rsi_factor = (50 - rsi) / 50
        self.factors['RSI_FACTOR'] = rsi_factor
        return rsi_factor
    
    def bollinger_factor(self, period: int = 20, num_std: float = 2) -> pd.Series:
        """
        布林带位置因子
        价格在布林带中的相对位置
        >1 表示超买（突破上轨），<-1 表示超卖（突破下轨）
        """
        ma = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        
        z_score = (self.df['close'] - ma) / (std * num_std)
        self.factors['BB_FACTOR'] = z_score
        return z_score
    
    def mean_reversion(self, period: int = 20) -> pd.Series:
        """
        均值回归因子
        价格偏离均值的程度，用于捕捉回归机会
        """
        ma = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        
        deviation = (self.df['close'] - ma) / std
        # 反向，偏离越大，回归动力越强
        reversion = -deviation
        self.factors['MEAN_REV'] = reversion
        return reversion
    
    def stochastic_factor(self, k_period: int = 14, d_period: int = 3) -> pd.Series:
        """
        随机指标因子
        类似RSI的超买超卖信号
        """
        low_min = self.df['low'].rolling(k_period).min()
        high_max = self.df['high'].rolling(k_period).max()
        
        k = 100 * (self.df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        d = k.rolling(d_period).mean()
        
        # 转换为因子
        stoch_factor = (50 - d) / 50
        self.factors['STOCH_FACTOR'] = stoch_factor
        return stoch_factor
    
    # ==================== 复合因子 ====================
    
    def smart_money(self, period: int = 20) -> pd.Series:
        """
        聪明钱因子
        结合价格和成交量，识别机构行为
        """
        # 价格变化
        price_change = self.df['close'].pct_change()
        
        # 成交量变化
        vol_change = self.df['volume'] / self.df['volume'].rolling(period).mean()
        
        # 聪明钱指标：价格上涨+放量 或 价格下跌+缩量
        smart = price_change * np.log1p(vol_change)
        smart_ma = smart.rolling(period).sum()
        
        self.factors['SMART_MONEY'] = smart_ma
        return smart_ma
    
    def momentum_quality(self, period: int = 20) -> pd.Series:
        """
        动量质量因子
        结合动量和波动率，高质量动量 = 高收益/低波动
        """
        returns = self.df['close'].pct_change(period)
        volatility = self.df['close'].pct_change().rolling(period).std()
        
        quality = returns / volatility
        self.factors['MOM_QUALITY'] = quality
        return quality
    
    # ==================== 工具方法 ====================
    
    def _calc_atr(self, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()
    
    def calculate_all(self) -> pd.DataFrame:
        """计算所有因子"""
        # 动量因子
        self.momentum_return([5, 10, 20])
        self.roc(12)
        self.acceleration(10)
        
        # 波动率因子
        self.volatility([10, 20])
        self.volatility_ratio(10, 60)
        self.atr_factor(14)
        
        # 成交量因子
        self.volume_momentum(20)
        self.obv_slope(20)
        self.vwap_deviation(20)
        self.mfi(14)
        
        # 趋势因子
        self.trend_strength(20)
        self.ema_cross_strength(12, 26)
        self.adx_factor(14)
        self.price_channel_position(20)
        
        # 反转因子
        self.rsi_factor(14)
        self.bollinger_factor(20)
        self.mean_reversion(20)
        self.stochastic_factor(14)
        
        # 复合因子
        self.smart_money(20)
        self.momentum_quality(20)
        
        return self.factors
    
    def get_factors(self) -> pd.DataFrame:
        """获取所有已计算的因子"""
        return self.factors

