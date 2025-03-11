import math
import numpy as np
import pandas as pd
import ta
import math
import requests

def get_n_columns(df, columns, n=1):
    dt = df.copy()
    for col in columns:
        dt["n"+str(n)+"_"+col] = dt[col].shift(n)
    return dt

def rma(input_data: pd.Series, period: int) -> pd.Series:
    data = input_data.copy()
    alpha = 1 / period
    rma = data.ewm(alpha=alpha, adjust=False).mean()
    return rma

def chop(high, low, close, window=14):
    ''' Choppiness indicator
    '''
    tr1 = pd.DataFrame(high - low).rename(columns={0: 'tr1'})
    tr2 = pd.DataFrame(abs(high - close.shift(1))
                       ).rename(columns={0: 'tr2'})
    tr3 = pd.DataFrame(abs(low - close.shift(1))
                       ).rename(columns={0: 'tr3'})
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').dropna().max(axis=1)
    atr = tr.rolling(1).mean()
    highh = high.rolling(window).max()
    lowl = low.rolling(window).min()
    chop_serie = 100 * np.log10((atr.rolling(window).sum()) /
                          (highh - lowl)) / np.log10(window)
    return pd.Series(chop_serie, name="CHOP")

def fear_and_greed(close):
    ''' Fear and greed indicator
    '''
    response = requests.get("https://api.alternative.me/fng/?limit=0&format=json")
    dataResponse = response.json()['data']
    fear = pd.DataFrame(dataResponse, columns = ['timestamp', 'value'])

    fear = fear.set_index(fear['timestamp'])
    fear.index = pd.to_datetime(fear.index, unit='s')
    del fear['timestamp']
    df = pd.DataFrame(close, columns = ['close'])
    df['fearResult'] = fear['value']
    df['FEAR'] = df['fearResult'].ffill()
    df['FEAR'] = df.FEAR.astype(float)
    return pd.Series(df['FEAR'], name="FEAR")


class Trix():
    """ Trix indicator

        Args:
            close(pd.Series): dataframe 'close' columns,
            trix_length(int): the window length for each mooving average of the trix,
            trix_signal_length(int): the window length for the signal line
    """

    def __init__(
        self,
        close: pd.Series,
        trix_length: int = 9,
        trix_signal_length: int = 21,
        trix_signal_type: str = "sma" # or ema
    ):
        self.close = close
        self.trix_length = trix_length
        self.trix_signal_length = trix_signal_length
        self.trix_signal_type = trix_signal_type
        self._run()

    def _run(self):
        self.trix_line = ta.trend.ema_indicator(
            ta.trend.ema_indicator(
                ta.trend.ema_indicator(
                    close=self.close, window=self.trix_length),
                window=self.trix_length), window=self.trix_length)
        
        self.trix_pct_line = self.trix_line.pct_change()*100

        if self.trix_signal_type == "sma":
            self.trix_signal_line = ta.trend.sma_indicator(
                close=self.trix_pct_line, window=self.trix_signal_length)
        elif self.trix_signal_type == "ema":
            self.trix_signal_line = ta.trend.ema_indicator(
                close=self.trix_pct_line, window=self.trix_signal_length)
            
        self.trix_histo = self.trix_pct_line - self.trix_signal_line

    def get_trix_line(self) -> pd.Series:
        return pd.Series(self.trix_line, name="trix_line")

    def get_trix_pct_line(self) -> pd.Series:
        return pd.Series(self.trix_pct_line, name="trix_pct_line")

    def get_trix_signal_line(self) -> pd.Series:
        return pd.Series(self.trix_signal_line, name="trix_signal_line")

    def get_trix_histo(self) -> pd.Series:
        return pd.Series(self.trix_histo, name="trix_histo")


class VMC():
    """ VuManChu Cipher B + Divergences 

        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            wtChannelLen(int): n period.
            wtAverageLen(int): n period.
            wtMALen(int): n period.
            rsiMFIperiod(int): n period.
            rsiMFIMultiplier(int): n period.
            rsiMFIPosY(int): n period.
    """

    def __init__(
        self: pd.Series,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        wtChannelLen: int = 9,
        wtAverageLen: int = 12,
        wtMALen: int = 3,
        rsiMFIperiod: int = 60,
        rsiMFIMultiplier: int = 150,
        rsiMFIPosY: int = 2.5
    ) -> None:
        self._high = high
        self._low = low
        self._close = close
        self._open = open
        self._wtChannelLen = wtChannelLen
        self._wtAverageLen = wtAverageLen
        self._wtMALen = wtMALen
        self._rsiMFIperiod = rsiMFIperiod
        self._rsiMFIMultiplier = rsiMFIMultiplier
        self._rsiMFIPosY = rsiMFIPosY

        self._run()
        self.wave_1()

    def _run(self) -> None:
        self.hlc3 = (self._close + self._high + self._low)
        self._esa = ta.trend.ema_indicator(
            close=self.hlc3, window=self._wtChannelLen)
        self._de = ta.trend.ema_indicator(
            close=abs(self.hlc3 - self._esa), window=self._wtChannelLen)
        self._rsi = ta.trend.sma_indicator(self._close, self._rsiMFIperiod)
        self._ci = (self.hlc3 - self._esa) / (0.015 * self._de)

    def wave_1(self) -> pd.Series:
        """VMC Wave 1 

        Returns:
            pandas.Series: New feature generated.
        """
        wt1 = ta.trend.ema_indicator(self._ci, self._wtAverageLen)
        return pd.Series(wt1, name="wt1")

    def wave_2(self) -> pd.Series:
        """VMC Wave 2

        Returns:
            pandas.Series: New feature generated.
        """
        wt2 = ta.trend.sma_indicator(self.wave_1(), self._wtMALen)
        return pd.Series(wt2, name="wt2")

    def money_flow(self) -> pd.Series:
        """VMC Money Flow

        Returns:
            pandas.Series: New feature generated.
        """
        mfi = ((self._close - self._open) /
               (self._high - self._low)) * self._rsiMFIMultiplier
        rsi = ta.trend.sma_indicator(mfi, self._rsiMFIperiod)
        money_flow = rsi - self._rsiMFIPosY
        return pd.Series(money_flow, name="money_flow")


def heikinAshiDf(df):
    df['HA_Close'] = (df.open + df.high + df.low + df.close)/4
    ha_open = [(df.open[0] + df.close[0]) / 2]
    [ha_open.append((ha_open[i] + df.HA_Close.values[i]) / 2)
     for i in range(0, len(df)-1)]
    df['HA_Open'] = ha_open
    df['HA_High'] = df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
    return df

class SmoothedHeikinAshi():
    def __init__(self, open, high, low, close, smooth1=5, smooth2=3):
        self.open = open.copy()
        self.high = high.copy()
        self.low = low.copy()
        self.close = close.copy()
        self.smooth1 = smooth1
        self.smooth2 = smooth2
        self._run()

    def _calculate_ha_open(self):
        ha_open = pd.Series(np.nan, index=self.open.index)
        start = 0
        for i in range(1, len(ha_open)):
            if np.isnan(self.smooth_open.iloc[i]):
                continue
            else:
                ha_open.iloc[i] = (self.smooth_open.iloc[i] + self.smooth_close.iloc[i]) / 2
                start = i
                break

        for i in range(start + 1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + self.ha_close.iloc[i-1]) / 2

        return ha_open

    def _run(self):
        self.smooth_open = ta.trend.ema_indicator(self.open, self.smooth1)
        self.smooth_high = ta.trend.ema_indicator(self.high, self.smooth1)
        self.smooth_low = ta.trend.ema_indicator(self.low, self.smooth1)
        self.smooth_close = ta.trend.ema_indicator(self.close, self.smooth1)

        self.ha_close = (self.smooth_open + self.smooth_high + self.smooth_low + self.smooth_close) / 4
        self.ha_open = self._calculate_ha_open()
        

        self.smooth_ha_close = ta.trend.ema_indicator(self.ha_close, self.smooth2)
        self.smooth_ha_open = ta.trend.ema_indicator(self.ha_open, self.smooth2)
    
    def smoothed_ha_close(self):
        return self.smooth_ha_close
    def smoothed_ha_open(self):
        return self.smooth_ha_open


def volume_anomality(df, volume_window=10):
    dfInd = df.copy()
    dfInd["VolAnomaly"] = 0
    dfInd["PreviousClose"] = dfInd["close"].shift(1)
    dfInd['MeanVolume'] = dfInd['volume'].rolling(volume_window).mean()
    dfInd['MaxVolume'] = dfInd['volume'].rolling(volume_window).max()
    dfInd.loc[dfInd['volume'] > 1.5 * dfInd['MeanVolume'], "VolAnomaly"] = 1
    dfInd.loc[dfInd['volume'] > 2 * dfInd['MeanVolume'], "VolAnomaly"] = 2
    dfInd.loc[dfInd['volume'] >= dfInd['MaxVolume'], "VolAnomaly"] = 3
    dfInd.loc[dfInd['PreviousClose'] > dfInd['close'],
              "VolAnomaly"] = (-1) * dfInd["VolAnomaly"]
    return dfInd["VolAnomaly"]

class SuperTrend():
    def __init__(
        self,
        high,
        low,
        close,
        atr_window=10,
        atr_multi=3
    ):
        self.high = high
        self.low = low
        self.close = close
        self.atr_window = atr_window
        self.atr_multi = atr_multi
        self._run()
        
    def _run(self):
        # calculate ATR
        price_diffs = [self.high - self.low, 
                    self.high - self.close.shift(), 
                    self.close.shift() - self.low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
        atr = true_range.ewm(alpha=1/self.atr_window,min_periods=self.atr_window).mean() 
        # atr = ta.volatility.average_true_range(high, low, close, atr_period)
        # df['atr'] = df['tr'].rolling(atr_period).mean()
        
        # HL2 is simply the average of high and low prices
        hl2 = (self.high + self.low) / 2
        # upperband and lowerband calculation
        # notice that final bands are set to be equal to the respective bands
        final_upperband = upperband = hl2 + (self.atr_multi * atr)
        final_lowerband = lowerband = hl2 - (self.atr_multi * atr)
        
        # initialize Supertrend column to True
        supertrend = [True] * len(self.close)
        
        for i in range(1, len(self.close)):
            curr, prev = i, i-1
            
            # if current close price crosses above upperband
            if self.close[curr] > final_upperband[prev]:
                supertrend[curr] = True
            # if current close price crosses below lowerband
            elif self.close[curr] < final_lowerband[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]
                
                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan
                
        self.st = pd.DataFrame({
            'Supertrend': supertrend,
            'Final Lowerband': final_lowerband,
            'Final Upperband': final_upperband
        })
        
    def super_trend_upper(self):
        return self.st['Final Upperband']
        
    def super_trend_lower(self):
        return self.st['Final Lowerband']
        
    def super_trend_direction(self):
        return self.st['Supertrend']
    
class MaSlope():
    """ Slope adaptative moving average
    """

    def __init__(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        long_ma: int = 200,
        major_length: int = 14,
        minor_length: int = 6,
        slope_period: int = 34,
        slope_ir: int = 25
    ):
        self.close = close
        self.high = high
        self.low = low
        self.long_ma = long_ma
        self.major_length = major_length
        self.minor_length = minor_length
        self.slope_period = slope_period
        self.slope_ir = slope_ir
        self._run()

    def _run(self):
        minAlpha = 2 / (self.minor_length + 1)
        majAlpha = 2 / (self.major_length + 1)
        # df = pd.DataFrame(data = [self.close, self.high, self.low], columns = ['close','high','low'])
        df = pd.DataFrame(data = {"close": self.close, "high": self.high, "low":self.low})
        df['hh'] = df['high'].rolling(window=self.long_ma+1).max()
        df['ll'] = df['low'].rolling(window=self.long_ma+1).min()
        df = df.fillna(0)
        df.loc[df['hh'] == df['ll'],'mult'] = 0
        df.loc[df['hh'] != df['ll'],'mult'] = abs(2 * df['close'] - df['ll'] - df['hh']) / (df['hh'] - df['ll'])
        df['final'] = df['mult'] * (minAlpha - majAlpha) + majAlpha

        ma_first = (df.iloc[0]['final']**2) * df.iloc[0]['close']

        col_ma = [ma_first]
        for i in range(1, len(df)):
            ma1 = col_ma[i-1]
            col_ma.append(ma1 + (df.iloc[i]['final']**2) * (df.iloc[i]['close'] - ma1))

        df['ma'] = col_ma
        pi = math.atan(1) * 4
        df['hh1'] = df['high'].rolling(window=self.slope_period).max()
        df['ll1'] = df['low'].rolling(window=self.slope_period).min()
        df['slope_range'] = self.slope_ir / (df['hh1'] - df['ll1']) * df['ll1']
        df['dt'] = (df['ma'].shift(2) - df['ma']) / df['close'] * df['slope_range'] 
        df['c'] = (1+df['dt']*df['dt'])**0.5
        df['xangle'] = round(180*np.arccos(1/df['c']) / pi)
        df.loc[df['dt']>0,"xangle"] = - df['xangle']
        self.df = df
        # print(df)

    def ma_line(self) -> pd.Series:
        """ ma_line

            Returns:
                pd.Series: ma_line
        """
        return self.df['ma']

    def x_angle(self) -> pd.Series:
        """ x_angle

            Returns:
                pd.Series: x_angle
        """
        return self.df['xangle']
        
def ichimoku(
    df,
    high_col="high",
    low_col="low",
    close_col="close",
    conversion_period=9,
    base_period=26,
    span_b_period=52,
    displacement=26
):
    """
    Calculate Ichimoku Cloud indicator lines and add them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'high', 'low', 'close' columns (by default).
    high_col : str
        Column name for the high prices.
    low_col : str
        Column name for the low prices.
    close_col : str
        Column name for the close prices.
    conversion_period : int
        Lookback period for the Conversion Line (Tenkan-sen).
    base_period : int
        Lookback period for the Base Line (Kijun-sen).
    span_b_period : int
        Lookback period for the Leading Span B (Senkou Span B).
    displacement : int
        Number of periods to shift forward for the leading spans.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with the following new columns added:
        - 'tenkan_sen'
        - 'kijun_sen'
        - 'senkou_span_a'
        - 'senkou_span_b'
        - 'chikou_span'
    """

    # Tenkan-sen (Conversion Line)
    df["tenkan_sen"] = (
        df[high_col].rolling(window=conversion_period).max() +
        df[low_col].rolling(window=conversion_period).min()
    ) / 2.0

    # Kijun-sen (Base Line)
    df["kijun_sen"] = (
        df[high_col].rolling(window=base_period).max() +
        df[low_col].rolling(window=base_period).min()
    ) / 2.0

    # Senkou Span A (Leading Span A) = (Tenkan-sen + Kijun-sen) / 2, shifted forward
    df["span_a"] = (
        (df["tenkan_sen"] + df["kijun_sen"]) / 2
    ).shift(displacement)

    # Senkou Span B (Leading Span B) = (Highest high + Lowest low) / 2 over span_b_period, shifted forward
    highest_high_span_b = df[high_col].rolling(window=span_b_period).max()
    lowest_low_span_b = df[low_col].rolling(window=span_b_period).min()
    df["span_b"] = ((highest_high_span_b + lowest_low_span_b) / 2).shift(displacement)

    # Chikou Span (Lagging Span) = close shifted backward by displacement
    df["chikou_span"] = df[close_col]

    return df

def ema(
    df,
    close_col="close",
    period=9,
    column_name=None
):
    """
    Calculate the Exponential Moving Average (EMA) for a given period and
    add it as a new column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    close_col : str
        Column name for the close price data.
    period : int
        Lookback period for the EMA.
    column_name : str, optional
        Name of the output EMA column. If None, defaults to 'ema_{period}'.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with the EMA column added.
    """
    if column_name is None:
        column_name = f"ema_{period}"

    df[column_name] = df[close_col].ewm(span=period, adjust=False).mean()
    return df

def bollinger_bands_width(
    df,
    close_col="close",
    period=20,
    std_dev=2.0,
    add_bands=True
):
    """
    Calculate Bollinger Bands (upper, middle, lower) and Bollinger Band Width (bbw).
    Optionally add all bands to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    close_col : str
        The column name for the close price.
    period : int
        Rolling window for the SMA and standard deviation.
    std_dev : float
        How many standard deviations away from the middle band.
    add_bands : bool
        If True, adds 'bb_upper', 'bb_middle', and 'bb_lower' to the DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with a new column 'bbw', and optionally
        'bb_upper', 'bb_middle', 'bb_lower' if add_bands=True.
    """
    # Middle Band (Simple Moving Average)
    df["bb_middle"] = df[close_col].rolling(window=period).mean()

    # Rolling Standard Deviation
    df["bb_std"] = df[close_col].rolling(window=period).std()

    # Upper Band
    df["bb_upper"] = df["bb_middle"] + std_dev * df["bb_std"]

    # Lower Band
    df["bb_lower"] = df["bb_middle"] - std_dev * df["bb_std"]

    # Bollinger Band Width = (Upper - Lower) / Middle
    # (Some definitions vary: e.g. (upper - lower)/(2 * middle). Adjust as you prefer.)
    df["bbw"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100

    # If you want to keep your DataFrame clean, you can choose to drop intermediate columns,
    # or skip adding them. We'll keep them for reference here.
    if not add_bands:
        df.drop(["bb_upper", "bb_middle", "bb_lower", "bb_std"], axis=1, inplace=True)
    else:
        df.drop("bb_std", axis=1, inplace=True)

    return df

def adx(
    df,
    high_col="high",
    low_col="low",
    close_col="close",
    period=14,
    adx_col_name="adx"
):
    """
    Calculate the Average Directional Index (ADX) along with +DI and -DI.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with columns for high, low, and close.
    high_col : str
        Column name for high prices.
    low_col : str
        Column name for low prices.
    close_col : str
        Column name for close prices.
    period : int
        Lookback period for ADX.
    adx_col_name : str
        Name for the ADX column in the DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with new columns:
            '+di', '-di', 'dx', and the final ADX column (adx_col_name).
    """
    # Make a copy so we don't override original columns if you prefer
    # But we can also work directly in df.

    # 1) Calculate True Range (TR)
    df["prev_close"] = df[close_col].shift(1)
    df["high-low"] = df[high_col] - df[low_col]
    df["high-pclose"] = (df[high_col] - df["prev_close"]).abs()
    df["low-pclose"] = (df[low_col] - df["prev_close"]).abs()

    df["tr"] = df[["high-low", "high-pclose", "low-pclose"]].max(axis=1)

    # 2) Calculate directional movement +DM, -DM
    df["up_move"] = df[high_col].diff()
    df["down_move"] = -df[low_col].diff()

    df["+dm"] = (
        df["up_move"].where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), 0)
    )
    df["-dm"] = (
        df["down_move"].where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), 0)
    )

    # 3) Wilder's smoothing for TR, +DM, and -DM over 'period' bars
    # Initialize first value by summing first 'period' values, then do the Wilder smoothing
    df["tr_smoothed"] = 0.0
    df["+dm_smoothed"] = 0.0
    df["-dm_smoothed"] = 0.0

    # We can't do standard .rolling().sum() easily for the Wilder approach, so let's do it in a loop.
    # If you prefer a simpler approach, just do a rolling sum or ewm. This approach matches many charting tools.

    # First, we need enough rows, so if the DataFrame is shorter than 'period', handle carefully
    if len(df) < period:
        # Not enough data to do a full ADX calculation
        return df

    # Initialize the sums for the first 'period'
    first_idx = df.index[0]
    start_idx = df.index[period - 1]  # the index of the row that completes the first window

    # Sum over that initial window
    df.loc[start_idx, "tr_smoothed"] = df["tr"].iloc[:period].sum()
    df.loc[start_idx, "+dm_smoothed"] = df["+dm"].iloc[:period].sum()
    df.loc[start_idx, "-dm_smoothed"] = df["-dm"].iloc[:period].sum()

    # Wilder's smoothing: subsequent bars
    for i in range(period, len(df)):
        df.loc[df.index[i], "tr_smoothed"] = (
            df["tr_smoothed"].iloc[i - 1]
            - (df["tr_smoothed"].iloc[i - 1] / period)
            + df["tr"].iloc[i]
        )

        df.loc[df.index[i], "+dm_smoothed"] = (
            df["+dm_smoothed"].iloc[i - 1]
            - (df["+dm_smoothed"].iloc[i - 1] / period)
            + df["+dm"].iloc[i]
        )

        df.loc[df.index[i], "-dm_smoothed"] = (
            df["-dm_smoothed"].iloc[i - 1]
            - (df["-dm_smoothed"].iloc[i - 1] / period)
            + df["-dm"].iloc[i]
        )

    # 4) +DI and -DI
    df["+di"] = 100 * (df["+dm_smoothed"] / df["tr_smoothed"])
    df["-di"] = 100 * (df["-dm_smoothed"] / df["tr_smoothed"])

    # 5) DX = |+DI - -DI| / (+DI + -DI) * 100
    df["dx"] = (
        (df["+di"] - df["-di"]).abs()
        / (df["+di"] + df["-di"]).replace(0, float("nan"))  # avoid div by zero
        * 100
    )

    # 6) ADX = Wilder's smoothing of DX
    df[adx_col_name] = 0.0
    # Initialize first ADX (the row at 'start_idx') as the average of DX in that first period
    df.loc[start_idx, adx_col_name] = df["dx"].iloc[:period].mean()

    for i in range(period, len(df)):
        df.loc[df.index[i], adx_col_name] = (
            (df[adx_col_name].iloc[i - 1] * (period - 1) + df["dx"].iloc[i]) / period
        )

    # Clean up columns if you want to keep only certain ones
    # For clarity, let's just keep ADX, +DI, -DI, and drop intermediate columns
    # COMMENT THIS OUT if you want to debug intermediate steps
    df.drop(
        [
            "prev_close", "high-low", "high-pclose", "low-pclose", "tr",
            "up_move", "down_move", "+dm", "-dm",
            "tr_smoothed", "+dm_smoothed", "-dm_smoothed", "dx"
        ],
        axis=1, inplace=True
    )

    return df

def atr(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
    atr_col_name: str = "atr"
):
    """
    Calculate the Average True Range (ATR) and add it as a new column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for high, low, close.
    high_col, low_col, close_col : str
        Column names for the 'high', 'low', 'close' prices.
    period : int
        The ATR lookback period (default=14).
    atr_col_name : str
        The name of the column where ATR values will be stored.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an additional column for ATR.
    """

    # True Range (TR)
    df["previous_close"] = df[close_col].shift(1)
    df["high-low"] = df[high_col] - df[low_col]
    df["high-pc"] = (df[high_col] - df["previous_close"]).abs()
    df["low-pc"] = (df[low_col] - df["previous_close"]).abs()

    df["tr"] = df[["high-low", "high-pc", "low-pc"]].max(axis=1)

    # Wilder's ATR calculation
    # 1) First ATR value is simple average of first 'period' TR values
    # 2) Then ATR = [ (previous ATR * (period - 1)) + current TR ] / period

    df[atr_col_name] = np.nan  # initialize the column
    # If there's not enough data for a full period, just return
    if len(df) < period:
        return df

    # The first ATR value
    first_idx = df.index[period - 1]
    df.loc[first_idx, atr_col_name] = df["tr"].iloc[:period].mean()

    # Wilder's smoothing for subsequent values
    for i in range(period, len(df)):
        prev_atr = df.loc[df.index[i - 1], atr_col_name]
        curr_tr = df.loc[df.index[i], "tr"]
        df.loc[df.index[i], atr_col_name] = ((prev_atr * (period - 1)) + curr_tr) / period

    # Clean up intermediate columns if desired
    df.drop(["previous_close", "high-low", "high-pc", "low-pc", "tr"], axis=1, inplace=True)

    return df
