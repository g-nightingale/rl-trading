import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
from scipy.stats import ks_2samp


def get_stock_data(ticker, start_date, end_date, frequency='1d'):
    """
    Get stock data from Yahoo Finance.
    
    :param ticker: Stock ticker symbol.
    :param start_date: Start date for the data in 'YYYY-MM-DD' format.
    :param end_date: End date for the data in 'YYYY-MM-DD' format.
    :param frequency: Data frequency ('1d' for daily, '1wk' for weekly, '1mo' for monthly). Default is daily.
    :return: DataFrame with stock data.
    """

    # Fetch data
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(start=start_date, end=end_date, interval=frequency)
    
    # Create DataFrame for Close prices
    raw_data = pd.DataFrame(hist_data['Close'])

    return raw_data

def sliding_window_ks_test(data, window_size, step_size=1):
    """
    Perform a sliding window analysis and KS test on a numpy array.

    Parameters:
    data (numpy array): The data to analyze.
    window_size (int): The size of the sliding window.
    step_size (int): The number of elements to move the window at each step. Default is 1.

    Returns:
    list of tuples: Each tuple contains (window_start_index, window_end_index, ks_statistic, p_value).
    """
    n = len(data)
    results = []

    for start in range(0, n - window_size, step_size):
        end = start + window_size
        if end + window_size > n:
            break

        window1 = data[start:end]
        window2 = data[end:end + window_size]

        ks_statistic, p_value = ks_2samp(window1, window2)
        results.append((start, end, ks_statistic, p_value))

    return results
