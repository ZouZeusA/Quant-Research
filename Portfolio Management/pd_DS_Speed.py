import pandas_ta as ta
import pandas as pd
import numpy as np
import time

stock_data = pd.read_csv("Daily_EUSTOXX_DOW.csv", index_col=["Date","Ticker"],parse_dates=['Date'])
macd_period = 25

#! Group by Ticker in index [Date, Ticker]
start = time.time()
def compute_macd(close):
    macd = ta.macd(close=close, length=macd_period).iloc[:,0] # return MACD days, hours and seconds
    return macd.sub(macd.mean()).div(macd.std())

stock_data[f'MACD_{macd_period}'] = stock_data.groupby(level=1, group_keys=False)['Adj_Close'].apply(compute_macd)
print('goupby:', time.time() - start)
# goupby: 0.8369805812835693

#! Option 1: Features at Level 0, Tickers at Level 1
stock_data_wide = stock_data.unstack()
stock_data_wide.columns.names = ["Features","Tickers"]

start = time.time()
adj_close_values_wide = stock_data_wide['Adj_Close']
# Assuming multi-level columns with (Price_Type, Ticker)
# adj_close_values_wide = stock_data_wide.xs('Adj_Close', level=0, axis=1,drop_level=False)

# macd = adj_close_values.apply(lambda x: ta.macd(close=x, length=macd_period).iloc[:, 0])
# Calculate MACD using pandas operations
short_ema_wide = adj_close_values_wide.ewm(span=12, adjust=False).mean()
long_ema_wide = adj_close_values_wide.ewm(span=26, adjust=False).mean()
macd_wide = short_ema_wide - long_ema_wide

macd_wide = macd_wide.sub(macd_wide.mean()).div(macd_wide.std())
macd_wide.columns = pd.MultiIndex.from_product([[f'MACD_{macd_period}'],macd_wide.columns ])

stock_data_wide = pd.concat([stock_data_wide, macd_wide], axis=1)
option1_time = time.time() - start
print('Option 1 time:', option1_time)
# Option 1 Backtest: 

#! Option 2 columns: (Tickers at Level 0, Features at Level 1) index datetime
df = stock_data_wide.copy()
df.columns = df.columns.swaplevel(0, 1)
df = df.sort_index(axis=1)

# returns = np.log(df.xs('Close', level=1, axis=1,drop_level=False)).diff()
# returns.columns = pd.MultiIndex.from_product([returns.columns.get_level_values(0), ['Log_Returns']])

# # Use assign method to add all columns at once
# df = pd.concat([df, returns], axis=1,names=["Tickers","Features"]).sort_index(axis=1)
# df.columns.names = ["Tickers","Features"]

start = time.time()
# Using xs (cross-section) to get adjusted close prices across all assets
adj_close_values = df.xs('Adj_Close', level=1, axis=1,drop_level=True)

# Vectorized MACD calculation
# macd = adj_close_values.apply(lambda x: ta.macd(close=x, length=macd_period).iloc[:, 0])
short_ema = adj_close_values.ewm(span=12, adjust=False).mean()
long_ema = adj_close_values.ewm(span=26, adjust=False).mean()
macd = short_ema - long_ema

macd = macd.sub(macd.mean()).div(macd.std())
macd.columns = pd.MultiIndex.from_product([macd.columns, [f'MACD_{macd_period}']])

# Use assign method to add all columns at once
df = pd.concat([df, macd], axis=1).sort_index(axis=1)
option2_time = time.time() - start
print('Option 2 time:', option2_time)
# Option 2 Backtest: 0.05068206787109375

print("Execution speed of option1/option2: ",option1_time/option2_time)