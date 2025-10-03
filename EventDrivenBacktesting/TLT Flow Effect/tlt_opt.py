import traceback
import sys
import os

# Add the parent directory to the Python path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MyBTstrategy import MyStrategy
from MyBTengine import run_strategy
import backtrader as bt
import pandas as pd

# import yfinance as yf
# yf.download("SPY", start="2004-01-01", end="2024-12-03",auto_adjust=True).xs("SPY",axis=1,level=1).to_csv("SPY.csv")
# yf.download("TLT", start="2004-01-01", end="2024-12-03",auto_adjust=True).xs("TLT",axis=1,level=1).to_csv("TLT.csv")
tlt=pd.read_csv("TLT.csv",index_col=0,parse_dates=True)
tlt["month"] = tlt.index.month
# get the last day of each month 
tlt["month"] = tlt["month"].diff().shift(-1)

# Incremental range logic
tlt['incremental'] = (tlt['month'] != 0).cumsum()  # Create groups for each reset
tlt['incremental'] = tlt.groupby('incremental').cumcount(ascending=False) +1
tlt['incremental']= tlt['incremental'].shift(1).fillna(0)

class FlowTimingStrategy(MyStrategy):
    params = {
        'printlog': False,
        'day': 9
    }
  
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open

        # To keep track of pending orders 
        self.orefs = list()
        self.bit=0

        self.bar_executed = 0
        self.trades=[]

        self.i = 0
        self.end_of_month = tlt[tlt["month"]==1].index

        self.j = 0
        self.day7 = tlt[tlt["incremental"]==self.params.day].index
    
    def next(self):
 
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.orefs:
            return
        
        # Get current date
        current_date = self.datas[0].datetime.datetime(0)
        
        # Prevent self.i from exceeding the size of your self.end_of_month array
        # Check if it's first day of month (short entry)
        if self.i < len(self.end_of_month) and current_date.day == self.end_of_month[self.i].day:
            if not self.position:  # If no position is open
                # the order will be executed on the first day open
                self.orefs.append(self.sell(exectype=bt.Order.Close))  # Enter short position
                self.i += 1
                
        # Check if it's 5 days after month start (short exit)
        elif len(self) >= (self.bar_executed + 4):
            if self.position.size < 0:  # If we're in a short position
                self.orefs.append(self.close(exectype=bt.Order.Close))  # Exit short position
        
        # Prevent self.j from exceeding the size of your self.day7 array
        # Check if it's 7 days before month end (long entry)
        if self.j < len(self.day7) and current_date.day == self.day7[self.j].day:
            
            if not self.position:  # If no position is open
                self.orefs.append(self.buy(exectype=bt.Order.Close))  # Enter long position
                self.j += 1
                
        # Check if it's 1 day before month end (long exit)
        elif len(self) >= (self.bar_executed + 6): 
            if self.position.size > 0:  # If we're in a long position
                self.orefs.append(self.close(exectype=bt.Order.Close))  # Exit long position

if __name__ == '__main__':     
    try:
        # "args" can be a list of strings that will be parsed as command-line arguments ["--cash", "50000"]
        # can add sizer_class that extends bt.sizers or CustomSizer from MyBTclasses.py
        run_strategy(FlowTimingStrategy, args=["--strat","printlog=False,day=range(1,15)","--optimization","--cash","10000","--commission","0","--slippage","0","--dataname", "TLT.csv","--sizer","percents=90"])
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = traceback.extract_tb(exc_tb)[-1][0]
        line = traceback.extract_tb(exc_tb)[-1][1]
        print(f"Error: {e}")
        print(f"File: {fname}")
        print(f"Line: {line}")
        print("Full traceback:")
        traceback.print_exc()


