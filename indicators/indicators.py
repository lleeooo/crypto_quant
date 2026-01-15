"""
专业级技术指标库
包含：趋势指标、动量指标、波动率指标、成交量指标
"""
import pandas as pd
import numpy as np
from typing import Tuple


# ============ 趋势指标 ============

def EMA(series: pd.Series, window: int) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=window, adjust=False).mean()


def SMA(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均线"""
    return series.rolling(window).mean()


def WMA(series: pd.Series, window: int) -> pd.Series:
    """加权移动平均线"""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def DEMA(series: pd.Series, window: int) -> pd.Series:
    """双重指数移动平均线"""
    ema1 = EMA(series, window)
    ema2 = EMA(ema1, window)
    return 2 * ema1 - ema2


def TEMA(series: pd.Series, window: int) -> pd.Series:
    """三重指数移动平均线"""
    ema1 = EMA(series, window)
    ema2 = EMA(ema1, window)
    ema3 = EMA(ema2, window)
    return 3 * ema1 - 3 * ema2 + ema3


def KAMA(series: pd.Series, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """考夫曼自适应移动平均线"""
    change = abs(series - series.shift(window))
    volatility = abs(series.diff()).rolling(window).sum()
    
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[window-1] = series.iloc[:window].mean()
    
    for i in range(window, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    
    return kama


# ============ 动量指标 ============

def RSI(series: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指数 (Wilder's RSI)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # 使用Wilder's平滑方法
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def StochRSI(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """随机RSI"""
    rsi = RSI(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    stoch_rsi = stoch_rsi.fillna(0.5)
    
    k = stoch_rsi.rolling(3).mean()  # %K
    d = k.rolling(3).mean()  # %D
    
    return k, d


def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD指标"""
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def Stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """随机指标 (KDJ)"""
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k.fillna(50)
    d = k.rolling(d_period).mean()
    
    return k, d


def Williams_R(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """威廉指标"""
    high_max = df['high'].rolling(window).max()
    low_min = df['low'].rolling(window).min()
    
    wr = -100 * (high_max - df['close']) / (high_max - low_min).replace(0, np.nan)
    return wr.fillna(-50)


def CCI(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """商品通道指数"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    
    cci = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
    return cci.fillna(0)


def MFI(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """资金流量指数"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = tp * df['volume']
    
    positive_flow = raw_money_flow.where(tp > tp.shift(), 0).rolling(window).sum()
    negative_flow = raw_money_flow.where(tp < tp.shift(), 0).rolling(window).sum()
    
    mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.nan)))
    return mfi.fillna(50)


def ROC(series: pd.Series, window: int = 12) -> pd.Series:
    """变动率指标"""
    return ((series - series.shift(window)) / series.shift(window)) * 100


def Momentum(series: pd.Series, window: int = 10) -> pd.Series:
    """动量指标"""
    return series - series.shift(window)


# ============ 波动率指标 ============

def ATR(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr


def Bollinger(series: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林带"""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def Bollinger_Width(series: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
    """布林带宽度 (衡量波动率)"""
    ma, upper, lower = Bollinger(series, window, num_std)
    width = (upper - lower) / ma
    return width


def Bollinger_PctB(series: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
    """布林带%B (价格在带内的相对位置)"""
    ma, upper, lower = Bollinger(series, window, num_std)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pct_b.fillna(0.5)


def Keltner_Channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """肯特纳通道"""
    middle = EMA(df['close'], ema_period)
    atr = ATR(df, atr_period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return middle, upper, lower


def Donchian_Channel(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """唐奇安通道"""
    upper = df['high'].rolling(window).max()
    lower = df['low'].rolling(window).min()
    middle = (upper + lower) / 2
    return middle, upper, lower


def Historical_Volatility(series: pd.Series, window: int = 20, annualize: bool = True, periods_per_year: int = 252) -> pd.Series:
    """历史波动率"""
    log_returns = np.log(series / series.shift(1))
    vol = log_returns.rolling(window).std()
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


# ============ 成交量指标 ============

def OBV(df: pd.DataFrame) -> pd.Series:
    """能量潮指标"""
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def VWAP(df: pd.DataFrame, window: int = None) -> pd.Series:
    """成交量加权平均价格"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    if window:
        vwap = (tp * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    else:
        cum_vol = df['volume'].cumsum()
        cum_tp_vol = (tp * df['volume']).cumsum()
        vwap = cum_tp_vol / cum_vol
    
    return vwap


def AD_Line(df: pd.DataFrame) -> pd.Series:
    """累积/派发线"""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    clv = clv.fillna(0)
    ad = (clv * df['volume']).cumsum()
    return ad


def CMF(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """蔡金资金流量"""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    clv = clv.fillna(0)
    
    cmf = (clv * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    return cmf


def Force_Index(df: pd.DataFrame, window: int = 13) -> pd.Series:
    """强力指数"""
    fi = df['close'].diff() * df['volume']
    return EMA(fi, window)


def Volume_Ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """成交量比率"""
    return df['volume'] / df['volume'].rolling(window).mean()


# ============ 趋势强度指标 ============

def ADX(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """平均趋向指数"""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = ATR(df, window)
    
    plus_di = 100 * EMA(plus_dm, window) / atr
    minus_di = 100 * EMA(minus_dm, window) / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = EMA(dx.fillna(0), window)
    
    return adx, plus_di, minus_di


def Aroon(df: pd.DataFrame, window: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """阿隆指标"""
    aroon_up = df['high'].rolling(window + 1).apply(lambda x: x.argmax(), raw=True) / window * 100
    aroon_down = df['low'].rolling(window + 1).apply(lambda x: x.argmin(), raw=True) / window * 100
    aroon_osc = aroon_up - aroon_down
    
    return aroon_up, aroon_down, aroon_osc


def SuperTrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> Tuple[pd.Series, pd.Series]:
    """超级趋势指标"""
    atr = ATR(df, period)
    hl2 = (df['high'] + df['low']) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
    
    return supertrend, direction


def Ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict:
    """一目均衡表"""
    tenkan_sen = (df['high'].rolling(tenkan).max() + df['low'].rolling(tenkan).min()) / 2
    kijun_sen = (df['high'].rolling(kijun).max() + df['low'].rolling(kijun).min()) / 2
    
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((df['high'].rolling(senkou_b).max() + df['low'].rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = df['close'].shift(-kijun)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b_line,
        'chikou': chikou
    }
