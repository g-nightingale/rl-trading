import pandas as pd
import yfinance as yf
import numpy as np

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
    raw_data = pd.DataFrame(hist_data)

    return raw_data

def standard_scale(df, column_names):
    """
    Apply standard scaling to a specified column in a Pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to be scaled.

    Returns:
    pd.Series: A Series with the scaled values.
    """
    for column_name in column_names:
        column = df[column_name]
        mean = column.mean()
        std_dev = column.std()
        df[column_name] = (column - mean) / std_dev
    return df

class RlTradingEnv:
    def __init__(self,
                 ticker,
                 start_date,
                 end_date=pd.Timestamp.now(),
                 frequency='1d',
                 starting_balance=10000,
                 position_percent=0.1,
                 num_samples=1000,
                 window_size=40,
                 refresh_synthetic_data=True):
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.starting_balance = starting_balance
        self.position_percent = position_percent
        self.num_samples = num_samples
        self.window_size = window_size
        self.refresh_synthetic_data = refresh_synthetic_data
        self.data = None
        self.step_reward = 0.0
        self.first = True
        self.reset()

    def get_data(self):
        """
        Get the data.
        """
        raw_data = get_stock_data(self.ticker, 
                                  self.start_date, 
                                  self.end_date, 
                                  frequency=self.frequency)

        data = raw_data.copy()
        data['rtn'] = data['Close'].pct_change()
        data.dropna(inplace=True)

        self.data = data
        self.mean = data['rtn'].mean()
        self.std_dev = data['rtn'].std()
        self.base_value = data['Close'].iloc[0]

        # Construct a histogram (discrete probability distribution) with 100 bins
        # self.hist, self.bin_edges = np.histogram(data['rtn'].values, bins=100, density=True)

        self.hist, self.rtn_edges, self.vol_edges = np.histogram2d(data['rtn'].values, data['Volume'].values, bins=100, density=True)

        # Calculate cumulative distribution
        cum_values = np.cumsum(self.hist.ravel())
        cum_values /= cum_values[-1]  # Normalize to make the sum equal to 1
        self.cum_values = cum_values

    def generate_synthetic_data(self, num_samples):
        """
        Generate synthetic returns.
        """
        returns = np.zeros(num_samples)
        volume = np.zeros(num_samples)
        for i in range(num_samples):
            # Choose a bin based on the cumulative distribution
            random_choice = np.random.rand()  # Generate a random number between 0 and 1
            bin_index = np.searchsorted(self.cum_values, random_choice)
            rtn_bin, vol_bin = np.unravel_index(bin_index, self.hist.shape)

            # Sample a value uniformly from the chosen bin
            rtn_lower_bound = self.rtn_edges[rtn_bin]
            rtn_upper_bound = self.rtn_edges[rtn_bin + 1]
            returns[i] = np.random.uniform(rtn_lower_bound, rtn_upper_bound)

            vol_lower_bound = self.vol_edges[vol_bin]
            vol_upper_bound = self.vol_edges[vol_bin + 1]
            volume[i] = np.random.uniform(vol_lower_bound, vol_upper_bound)

        return returns, volume

    def generate_price_data(self):
        """
        Generate synthetic price data.
        """

        # Generate samples from a normal distribution
        # samples = np.random.normal(self.mean, self.std_dev, self.num_samples)
        synthetic_data = np.zeros((self.num_samples, 2))
        returns, volume = self.generate_synthetic_data(self.num_samples)
        multiplicative_returns = 1.0 + returns
        cumulative_returns = np.cumprod(multiplicative_returns)

        synthetic_data[:, 0] = self.base_value * cumulative_returns 
        synthetic_data[:, 1] = volume
        return synthetic_data

    def reset(self):
        """
        Reset the environment.
        """
        self.t = 0
        self.terminal = False
        self.in_trade = 0
        self.balance = self.starting_balance
        self.balance_history = []
        self.position_history = []
        self.trade_history = []
        self.units_held = 0
        if self.data is None:
            self.get_data()
        if self.first or self.refresh_synthetic_data:
            self.first = False
            self.synthetic_data = self.generate_price_data()
        self.prime_data()

    def prime_data(self):
        """
        Prime the data.
        """
        for _ in range(40):
            _, _, _ = self.step(0)

    def step(self, action, use_synthetic_data=True):
        """
        Step the environment
        """

        if self.t == self.num_samples:
            self.terminal = True

        # Initialise reward
        reward = self.step_reward

        # Take actions
        # Do nothing
        if action==0:
            pass
        # Long
        elif action==1 and self.in_trade!=1:
            reward += self.enter_trade(1, use_synthetic_data)
        # Short
        elif action==2 and self.in_trade!=2:
            reward += self.enter_trade(2, use_synthetic_data)

        self.t += 1

        # Append histories
        self.balance_history.append(self.balance)
        self.position_history.append(self.in_trade)

        return self.get_state(), reward, self.terminal

    def get_state(self, use_synthetic_data=True):
        """
        Get the state representation.
        """

        STANDARD_SCALER_COLUMNS = ['close', 'volume', 'avg5', 'avg10', 'avg20']
        if use_synthetic_data:
            _state = pd.DataFrame(self.synthetic_data[:self.t, :], columns=['close', 'volume'])
        else:
            _state = self.data[['Close', 'Volume']].iloc[:self.t]
            _state.columns = ['close', 'volume']

        _state['avg5'] = _state['close'].rolling(window=5, min_periods=1).mean()
        _state['avg10'] = _state['close'].rolling(window=10, min_periods=1).mean()
        _state['avg20'] = _state['close'].rolling(window=20, min_periods=1).mean()
        _state['balance'] = [x/self.starting_balance for x in self.balance_history]
        _state['position'] = [x - 1 for x  in self.position_history]

        state = _state.iloc[self.t-self.window_size:self.t]
        state = _state.iloc[self.t-self.window_size:self.t].copy()
        state = standard_scale(state, STANDARD_SCALER_COLUMNS)

        return np.expand_dims(state.values, axis=0)

    def exit_trade(self, use_synthetic_data=True):
        """
        Exit a trade.
        """
        if use_synthetic_data:
            exit_price = self.synthetic_data[self.t, 0]
        else:
            exit_price = self.data['Close'].iloc[self.t]

        if self.in_trade==1:
            profit = (self.units_held * exit_price) - (self.units_held * self.entry_price)
        else:
            profit = -(self.units_held * exit_price) + (self.units_held * self.entry_price)

        self.trade_history.append({'t':self.t,
                                   'position':'long' if self.in_trade==1 else 'short',
                                   'type':'exit',
                                   'price':exit_price,
                                   'units':self.units_held,
                                   'profit':profit,
                                   'balance':self.balance + profit})
        
        self.in_trade = 0
        self.units_held = 0
        self.balance += profit

        return profit

    def enter_trade(self, trade_type, use_synthetic_data=True):
        """
        Enter a trade.
        """
        reward = 0
        if (trade_type==1 and self.in_trade==2) or (trade_type==2 and self.in_trade==1):
            reward = self.exit_trade(use_synthetic_data)

        if use_synthetic_data:
            entry_price =  self.synthetic_data[self.t, 0]
        else:
            entry_price = self.data['Close'].iloc[self.t]

        self.units_held = (self.balance * self.position_percent) / entry_price
        self.entry_price = entry_price
        self.in_trade = trade_type

        self.trade_history.append({'t':self.t,
                                    'position':'long' if self.in_trade==1 else 'short',
                                    'type':'entry',
                                    'price':entry_price,
                                    'units':self.units_held,
                                    'profit':0.0,
                                    'balance':self.balance
                                    })

        return reward
    
    def get_trade_history(self):
        """
        Get trade history.
        """

        return pd.DataFrame(self.trade_history)