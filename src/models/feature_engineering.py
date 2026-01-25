"""
Feature Engineering Module for Stock Price Analysis

This module provides functions to calculate technical indicators
for stock price prediction.
"""

import pandas as pd
import numpy as np


def calculate_sma(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA) for each company.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock data (must contain 'name' column for grouping)
    column : str
        Column name to calculate SMA on
    window : int
        Window size for moving average

    Returns:
    --------
    pd.Series
        Series containing SMA values
    """
    return df.groupby('name')[column].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )


def calculate_rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) for each company.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the specified period

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock data (must contain 'name' column for grouping)
    column : str
        Column name to calculate RSI on
    period : int
        Period for RSI calculation (typically 14)

    Returns:
    --------
    pd.Series
        Series containing RSI values (0-100 scale)
    """
    def rsi_calc(prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI

    return df.groupby('name')[column].transform(rsi_calc)


def calculate_volatility(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation) for each company.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock data
    column : str
        Column name to calculate volatility on
    window : int
        Window size for rolling std

    Returns:
    --------
    pd.Series
        Series containing volatility values
    """
    return df.groupby('name')[column].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )


def calculate_daily_return(df: pd.DataFrame) -> pd.Series:
    """
    Calculate daily return percentage for each company.

    Returns:
    --------
    pd.Series
        Series containing daily return percentages
    """
    return df.groupby('name')['close'].transform(
        lambda x: x.pct_change() * 100
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators and features to the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data with columns: date, open, high, low, close, volume, name

    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators:
        - sma_20: 20-day Simple Moving Average
        - sma_50: 50-day Simple Moving Average
        - rsi_14: 14-day Relative Strength Index
        - volatility: 20-day rolling volatility
        - daily_return: Daily return percentage
        - price_momentum: Difference between close and SMA 20
        - next_close: Next day's closing price (target variable)
    """
    # Make a copy and ensure proper sorting
    df = df.copy()
    df = df.sort_values(['name', 'date']).reset_index(drop=True)

    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Calculate technical indicators
    df['sma_20'] = calculate_sma(df, 'close', 20)
    df['sma_50'] = calculate_sma(df, 'close', 50)
    df['rsi_14'] = calculate_rsi(df, 'close', 14)
    df['volatility'] = calculate_volatility(df, 'close', 20)
    df['daily_return'] = calculate_daily_return(df)
    df['price_momentum'] = df['close'] - df['sma_20']

    # Target variable: next day's closing price
    df['next_close'] = df.groupby('name')['close'].shift(-1)

    return df


# Feature columns used for modeling
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'rsi_14', 'volatility', 'price_momentum'
]
