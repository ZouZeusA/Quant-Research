from colorama import Fore, Style
from tabulate import tabulate

import backtrader as bt
import backtrader.indicators as btind

#! notify_order: self.bar_executed to be removed depending on strategy & self.bit is used for Volume fillers
class MyStrategy(bt.Strategy):
    params = {
        # 'period': 10,
        # 'ind': 'sma',
        # 'printlog': False,
        # 'exitbars': 3
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

    def log(self, txt, dt=True, color="WHITE", bold=False, italic=False, doprint=False):
        """
        Logging function with colorama colors, bold, and italic styles
        """
        if self.params.printlog or doprint:
            
            # Choose color
            color = getattr(Fore, color.upper(), Fore.WHITE)
            
            # Add bold and italic styles
            style = ""
            if bold:
                style += "\033[1m"  # ANSI code for bold
            if italic:
                style += "\033[3m"  # ANSI code for italic (may not work on all terminals)
            
            # Create the formatted log message
            log_message = f"{self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d %H:%M:%S')} | {txt}" if dt else txt
            
            # Print the log message with color and styles, and reset all afterward
            print(style + color + log_message + Style.RESET_ALL)
    
    def format_execution_details(self, order, tabular=True):
        """
        Comprehensive formatter for Backtrader order execution details.
        
        Args:
            order: Backtrader order object
            tabular: If True, formats output in a tabular format
        
        Returns:
            str: Formatted execution details
        """
        
        # Prepare execution details
        exec_details = []         

        if order.executed.exbits[self.bit].opened != 0:
            exbit = order.executed.exbits[self.bit]
            # Opened Position Details
            opened_entry = {
                "Bit": self.bit,
                "Opened Size": exbit.opened,
                "Opened Price": f"{exbit.price:.4f}",
                "Opened Value": f"{exbit.openedvalue:.2f}", # Opened Size*Opened Price
                "Opened Commission": f"{exbit.openedcomm:.4f}",
                "Current Cum Open Position Size": exbit.psize,
                "Current Position Average Price": f"{exbit.pprice:.4f}" # [sum of the (size * price)] / total size
            }
            exec_details.append(opened_entry)
            self.bit +=1

        else:
            exbit = order.executed.exbits[self.bit]
            # Closed Position Details
            closed_entry = {
                "Bit": self.bit,
                "Closed Size": exbit.closed,
                "Closed Price": f"{exbit.price:.4f}",
                "Closed Value": f"{exbit.closed*exbit.price:.2f}", # exbit.closedvalue = Closed Size*Opened Position Average Price
                "Closed Commission": f"{exbit.closedcomm:.4f}",
                "PNL": f"{exbit.pnl:.2f}", # (Closed Price - Opened Position Average Price)*Closed Size for Open Buy orders and vice versa for Open Sell orders
                "Remaining Positions to Close Size": exbit.psize,
                "Opened Position Average Price": f"{exbit.pprice:.4f}"
            }
            exec_details.append(closed_entry)
            self.bit +=1
        
        # Tabular output
        if tabular:
            try:
                return tabulate(
                    [list(d.values()) for d in exec_details], 
                    headers=list(exec_details[0].keys()), 
                    tablefmt="pretty"
                )
            except ImportError:
                # Fallback to formatted string if tabulate is not available
                tabular = False
        
        # Formatted string output if not tabular
        if not tabular:
            output = []
            for detail in exec_details:
                for key, value in detail.items():
                    output.append(f"{key}: {value}")
                output.append("-" * 40)
            return "\n".join(output)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('{} {} {} ORDER {} {} | Price: {:<8.2f} | PriceLimit: {:<8.2f}{}| Size: {:<8.2f} | Margin: {:<8.2f}'.format(
                order.data._name,
                'BUY' * order.isbuy() or 'SELL',   
                order.getordername(),  
                order.ref,  
                order.getstatusname(),
                order.created.price,  # Execution price
                order.created.pricelimit or 0.0,  # Price limit for StopLimit orders
                " | TrailAmount: {:<4.2f} ".format(order.created.trailamount) if order.created.trailamount is not None else (" | TrailPercent: {:<2.3f} ".format(order.created.trailpercent) if order.created.trailpercent is not None else ' '),
                order.created.size,  # Requested/executed size
                order.created.margin or 0.0,  # Margin incurred by the order
            ), color="yellow" * order.issell() or "cyan")
            return

        # Handle partial fills
        if order.status in [bt.Order.Partial]:
            self.log('{} {} ORDER {} {}'.format(
                                    'BUY' * order.isbuy() or 'SELL',
                                    order.getordername(),
                                    order.ref,
                                    order.getstatusname()), color="yellow" * order.issell() or "cyan")
            # Average fill price is the [sum of the (size * price)] / total size
            print(f"Filled {order.executed.size} units until now at an average price of {order.executed.price}.")
            # remaining_size = order.size - order.executed.size
            print(f"Remaining order size: {order.executed.remsize} units.") 
            print(" Execution bit details:")
            print(self.format_execution_details(order, tabular=True) ) 
            return

        # Check if an order has been completed i.e. sent to the exchange
        if order.status in [order.Completed]:

            if self.bit!=0:
                self.log(
                    '{} {} ORDER {} {} | Average Price: {:<8.2f} | Size: {:<7.2f} | Value: {:<9.2f} | Total Comm: {:<8.2f}'.format(
                        'BUY' * order.isbuy() or 'SELL',
                        order.getordername(),
                        order.ref,
                        order.getstatusname(),
                        order.executed.price,
                        order.executed.size,
                        order.executed.size*order.executed.price,
                        order.executed.comm
                    ),
                    color="yellow" * order.issell() or "cyan",  # Highlight sell orders in yellow and buy orders in cyan
                    bold=True
                )
                print("Last execution bit details:")
                print(self.format_execution_details(order, tabular=True) ) 
                
            else:
                order_type = 'SELL' * order.issell() or 'BUY'
                self.log(
                    '{} {} {} ORDER {} {} | Executed Price: {:<8.2f} | Size: {:<7.2f} | Value: {:<9.2f} | Comm: {:<8.2f}'.format(
                        order.data._name,
                        order_type,
                        order.getordername(),
                        order.ref,
                        order.getstatusname(),
                        order.executed.price,
                        order.executed.size,
                        order.executed.price*order.executed.size,
                        order.executed.comm
                    ),
                    color="yellow" * order.issell() or "cyan",  # Highlight sell orders in yellow and buy orders in cyan
                    bold=True
                )

            self.bar_executed = len(self)
            self.bit =0
            

        # Handle rejected orders
        elif order.status == bt.Order.Rejected:
            self.log('{} {} {} ORDER {} {}'.format(
                                                order.data._name,
                                                'BUY' * order.isbuy() or 'SELL',
                                                order.getordername(),
                                                order.ref,
                                                order.getstatusname()), color="yellow" * order.issell() or "cyan", bold=True)
            # The reason for rejection will be sent via the notify_store method

        # Handle margin call
        elif order.status == bt.Order.Margin:
            self.log('{} {} {} ORDER {} {} would imply a margin call (not enough cash to execute the order).'.format(
                                                order.data._name,
                                                'BUY' * order.isbuy() or 'SELL',
                                                order.getordername(),
                                                order.ref,
                                                order.getstatusname()), color="yellow" * order.issell() or "cyan", bold=True)
        
        # Handle order cancellation
        elif order.status in [order.Canceled, bt.Order.Cancelled]:
            self.log('{} {} {} ORDER {} {} by the user.'.format(
                                                order.data._name,
                                                'BUY' * order.isbuy() or 'SELL',
                                                order.getordername(),
                                                order.ref,
                                                order.getstatusname()), color="yellow" * order.issell() or "cyan", bold=True)  

        elif order.status == bt.Order.Expired:
            self.log('{} {} {} ORDER {} has {} time validity'.format(
                                                order.data._name,
                                                'BUY' * order.isbuy() or 'SELL',
                                                order.getordername(),
                                                order.ref,
                                                order.getstatusname()), color="yellow" * order.issell() or "cyan", bold=True)
        
        # returns bool if order is in status Partial or Accepted
        if not order.alive() and order in self.orefs: 
            self.orefs.remove(order)
    
    def notify_store(self, msg):
        print(f"Broker notification: {msg}")
    
    def notify_trade(self, trade):
        status = {
            trade.Created: "Created", 
            trade.Open: "Open", 
            trade.Closed: "Closed"
        }.get(trade.status, "Unknown")
        
        if trade.isclosed:
            self.log(f"Trade Reference: {trade.ref} | ID {trade.tradeid} | Instrument: {trade.data._name if trade.data else 'N/A'} | Status: {status} \nTrade Details:", dt=False, bold=True, color="red" if trade.pnlcomm < 0 else "green")
            self.log(f"Duration {bt.num2date(trade.dtclose)-bt.num2date(trade.dtopen)} | Held during {trade.barlen} Bars | Gross PNL: {trade.pnl:.4f} | Total Commission: {trade.commission:.4f} | Net PNL: {trade.pnlcomm:.4f}",dt=False, bold=True, color="red" if trade.pnlcomm < 0 else "green")
            self.trades.append(trade)
            
        elif trade.isopen:
            self.log(f"Trade Reference: {trade.ref} | ID {trade.tradeid} | Instrument: {trade.data._name if trade.data else 'N/A'} | Status: {status}", dt=False, italic=True)
 
    def notify_fund(self, cash, value, fundvalue, shares):
        # Value - it's the sum of your cash and the current market value of all your open positions at the close of each bar.
        self.log('Open: {:<6.4f} | Close {:<6.4f} | Cash: {:<9.2f} | Value: {:<9.4f} | Fund Value: {:<9f} | Shares: {:<9.2f}'
                 .format(self.dataopen[0], self.dataclose[0], cash, value, fundvalue, shares))
    
    def notify_timer(self, timer, when):
        # This function is called when the timer is triggered
        self.log(f"Timer triggered at {when}", doprint=True)
    
    def order_target_xxx(self,data=None, target=0, **kwargs):
        """ 
        Perfect for portfolio rebalancing strategies

        xxx can be replaced by:
        1. size -> amount of shares, contracts in the portfolio of a specific asset

        2. value -> value in monetary units of the asset in the portfolio:
            current price of the asset multiplied by the current position size. 
            data_value = self.broker.get_value([self.datas[0]])

        3. percent -> percentage (from current portfolio) value of the asset in the current portfolio
        
        Specify the final target and the method decides if an operation will be a buy or a sell. 

        1. If the target is greater than the position_size: a buy is issued, with the difference target - position_size and vice versa

        2. The logic for order_target_percent is the same as that of order_target_value.
        If target > value and size >=0 -> buy
        If target > value and size < 0 -> sell
        If target < value and size >= 0 -> sell
        If target < value and size < 0 -> buy

        EXAMPLES:
        # Example: If your portfolio is $100,000 and you have 10% in this asset but want 5%
        self.order_target_percent(data=self.data, target=0.05)  # Will sell to reduce to $5,000
        # Example: If your position is worth $10,000 and you want to exit
        self.order_target_value(data=self.data, target=0)      # Will sell entire position
        # Example: If you have 50 shares and want to have 100 shares
        self.order_target_size(data=self.data, target=100)  # Will buy 50 more shares

        It returns either:
        The generated order
        or
        None if no order has been issued (target == position.size)
        """
        return super().order_target_xxx(data, target, **kwargs)
   
    def cancel(self, order):
        """Cancels the order in the broker"""
        return super().cancel(order)    
    
    def start(self):
        # Create a readable string of parameters
        params_str = ', '.join([f'{name}={getattr(self.params, name)}' 
                              for name in self.params._getkeys()])
        
        print(f'Backtesting with {params_str} is about to start with Portfolio Value: {self.broker.getvalue():.2f}$')
        # self.broker.add_cash(float(input("Enter amount of cash to Add/Remove i.e. +/-: ")))
    
    def stop(self):
        # Create a readable string of parameters
        params_str = ', '.join([f'{name}={getattr(self.params, name)}' 
                            for name in self.params._getkeys()])
        
        print(f'Backtesting with {params_str} is finished with Portfolio Value: {self.broker.getvalue():.2f}$')
    
