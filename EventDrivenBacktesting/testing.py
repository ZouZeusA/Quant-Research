import traceback
import sys
import os

# Add the parent directory to the Python path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MyBTstrategy import MyStrategy
from MyBTengine import run_strategy
import backtrader as bt

#* Check all the args for the script using: python testing.py -h

class CustomStrategy(MyStrategy):
    params = {
        'period': 10,
        'ind': 'sma',
        'printlog': False,
        'exitbars': 3
    }

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open

        self.orefs = list()

        self.bit=0
        self.trades=[]

        # if self.p.ind == "sma":
        #     self.sma_small_tf = btind.SMA(
        #         self.data.close, period=self.p.period, plotname="S_SMA"
        #     )
        # elif self.p.ind == "..."

        # self.sizer.setsizing(self.params.stake)
        # We could have also called buy and sell with a stake parameter

        #* Add a timer to trigger at a certain time
        # UTC (default behavior)
        # Add a timer for New York session open (9:30 AM US/Eastern) ny_timezone  = pytz.timezone('US/Eastern')
        # self.timer = self.add_timer(
        #     when=datetime.time(9, 30), # or bt.timer.SESSION_START 
        #     offset=datetime.timedelta(minutes=15), # Trigger at 09:45 AM
        #     repeat=datetime.timedelta(days=1),  # Repeat every day
        #     weekdays=[1, 2, 3, 4, 5],  # Trigger Monday to Friday
        # )

    def next(self):
 
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.orefs:
            return
        
        # Check if we are in the market?
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                # current close less than previous close

                if self.dataclose[-1] < self.dataclose[-2]:
                    # previous close less than the previous close

                    # BUY, BUY, BUY!!! (with default parameters)
                    # Keep track of the created order to avoid a 2nd order
                    #! Simple Order
                    self.orefs.append(self.buy(data=self.datas[0], size=None, # None can be taken from getsizer 
                                            #  amount of shares, contracts in the portfolio of a specific asset
                                            price=None, # None is valid for Market and Close orders | trigger point for Limit, Stop, 'Stop'Limit and 'Stop'TrailLimit
                                            plimit= None, # ex. self.data.close[0] * 1.02, Only applicable to Stop'Limit' or StopTrail'Limit' orders.
                                            exectype=bt.Order.Market or None, # bt.Order.Close can be used to execute at the second next close
                                            valid=None, # Good till cancel
                                                        # Order.DAY or 0 for Day orders
                                                        # datetime.datetime or datetime.date: Good till date
                                            # oco="anotherorder", # One cancel others: If any order in the group is executed, cancelled or expires, the other orders will be cancelled
                                            trailamount=None, # for Order.Stop'Trail' or Stop'Trail'Limit
                                            trailpercent=None, # for Order.Stop'Trail' or Stop'Trail'Limit
                                            # for manual bracket order creation
                                            parent=None, # The low/high side orders must have parent=main_side_order
                                            transmit=True # The main_side order must be created 1st and have transmit=False
                                                          # 1st low/high side order must have transmit=False
                                                          # The last order to be created (either the low or high side) sets transmit=True
                                                          # Specify the execution types and low/high side order size=main_side.size
                                        ))
                    
                    #! Bracket Order
                    # Returns a list containing the 3 orders in this order: [main, stop, limit] limit as take profit and stop as stop loss
                    # If high/low orders have been suppressed the return value will still contain 3 orders, but those suppressed will have a value of None
                    # close = self.data.close[0]
                    # p1 = close #* (1.0 - 0.005)
                    # p2 = p1 - 0.02 * close
                    # p3 = p1 + 0.1 * close 
                    # valid1 = datetime.timedelta(3)
                    # valid2 = valid3 = datetime.timedelta(500)
                    # self.orefs=[o for o in self.buy_bracket(data=None, 
                    #                  size=None, # The same size is applied to all 3 orders of the bracket
                    #                  price=p1,
                    #                  plimit=None, 
                    #                  exectype=0, # Default 2 or bt.Order.Limit
                    #                  valid=valid1, 
                    #                  tradeid=0, 
                    #                  trailamount=None, 
                    #                  trailpercent=None, 
                    #                  oargs={}, 
                    #                  stopprice=p2, # Specific price for the *low side* stop order 
                    #                  stopexec=3, # default bt.Order.Stop
                    #                  stopargs=dict(valid=valid2), 
                    #                  limitprice=p3, # Specific price for the *high side* stop order
                    #                  limitexec=2, # Default bt.Order.Limit
                    #                  limitargs={'valid':valid3},)]                    
                    
                    self.log('{} {} {} ORDER {} Created'.format(
                                    self.orefs[0].data._name,
                                    'BUY' * self.orefs[0].isbuy() or 'SELL',
                                    self.orefs[0].getordername(),
                                    self.orefs[0].ref), color="yellow" * self.orefs[0].issell() or "cyan", italic=True)
                    # self.broker.submit(order)

        # self.close(data=None, size=None, **kwargs same as buy) # submits the opposite order to close the current open position, regardless of whether it's long or short.

        else:

            # Already in the market ... we might sell
            # Exit after 5 bars (on the 6th bar) have elapsed for good or for worse
            if len(self) >= (self.bar_executed + self.params.exitbars):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                # Keep track of the created order to avoid a 2nd order
                self.orefs.append(self.sell(data=self.datas[0], 
                                       size=self.position.size,
                                       price=None, 
                                       plimit= None, 
                                       exectype=bt.Order.Market or None, 
                                       valid=None, 
                                       oco=None,
                                       trailamount=None, 
                                       trailpercent=None, 
                                       parent=None, 
                                       transmit=True
                                       ))
                self.log('{} {} {} ORDER {} Created'.format(
                                self.orefs[0].data._name,
                                'BUY' * self.orefs[0].isbuy() or 'SELL',
                                self.orefs[0].getordername(),
                                self.orefs[0].ref), color="yellow" * self.orefs[0].issell() or "cyan", italic=True)

if __name__ == "__main__":
    try:
        # "args" can be a list of strings that will be parsed as command-line arguments ["--cash", "50000"]
        # can add sizer_class that extends bt.sizers or CustomSizer from MyBTclasses.py
        run_strategy(CustomStrategy, args=None)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = traceback.extract_tb(exc_tb)[-1][0]
        line = traceback.extract_tb(exc_tb)[-1][1]
        print(f"Error: {e}")
        print(f"File: {fname}")
        print(f"Line: {line}")
        print("Full traceback:")
        traceback.print_exc()