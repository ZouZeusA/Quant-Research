import backtrader as bt
import pandas as pd
import numpy as np
import scipy.stats as stats
from tabulate import tabulate

#* Risk Management
class RiskManagement:
    def __init__(self, trades_netpnl:pd.Series, initial_capital:int):
        self.realized_pnl = trades_netpnl
        self.capital = initial_capital
    
    def optimal_f(self) -> float:
        """
        Calculate Optimal f (fraction) to maximize long-term growth rate based on the Kelly Criterion
        
        Args:
            trades: List of historical trade results (profits/losses)
            
        Returns:
            Optimal fraction of capital to risk on each trade
        """
        worst_loss = abs(self.realized_pnl.min())  # Get absolute value of worst loss

        def TWR(f: float) -> float:
            """Terminal Wealth Relative - measures growth of account"""
            if f == 0: # no risk
                return 1.0 # no growth
            
            twr = 1.0
            for trade in self.realized_pnl:
                
                # Calculate HPR (Holding Period Return)
                hpr = 1 + (f * trade / worst_loss)
                    
                twr *= hpr
            return twr
        
        # Find f that maximizes TWR using numerical optimization
        f_values = np.linspace(0, 1, 1000)
        twr_values = [TWR(f) for f in f_values]
        optimal_f = f_values[np.argmax(twr_values)]
        
        return optimal_f
    
    @staticmethod
    def rv_risk_of_ruin(win_rate, avg_return_per_win, avg_return_per_loss, initial_capital, max_risk) -> float:
        """
        Calculate Ralph Vince Risk of Ruin based on trading statistics
        Taken from https://www.quantifiedstrategies.com/risk-of-ruin-in-trading/#How_to_calculate_the_risk_of_ruin
        
        For example, a result of 0.05 means there's a 5% chance of losing the specified percentage of capital (max_risk) over the long run.
        
        Args:
            max_risk: Fraction of capital you're prepared to lose before declaring ruin. (0.5 for 50% of capital.)
        
        Returns:
            Ralph Vince Risk of Ruin probability of losing a specific portion of trading capital
        """
        # probability of losing a trade
        loss_rate = 1 - win_rate
        
        # Calculate averageWin% and averageLoss%
        average_win_percentage = avg_return_per_win / initial_capital
        average_loss_percentage = avg_return_per_loss / initial_capital
        
        # Calculate Z (Expected Value: expected percentage return per trade)
        Z = (win_rate * average_win_percentage) - (loss_rate * average_loss_percentage)
        
        # Calculate A (Standard Deviation of returns)
        A = (
            win_rate * (average_win_percentage ** 2) - 
            loss_rate * (average_loss_percentage ** 2)
        )**0.5

        # Calculate P
        P = 0.5 * ((1 + Z) / A)
        
        # Calculate Risk of Ruin
        if P <= 0 or P >= 1:
            return 1.0  # Maximum risk of ruin

        # Calculate Risk of Ruin
        rv_risk_of_ruin = ((1 - P) / P) ** (max_risk/A)
        
        return rv_risk_of_ruin
    
    @staticmethod
    def pk_risk_of_ruin(win_rate, max_loss, initial_capital, max_risk) -> float:
        """
        Calculate Perry Kaufman Risk of Ruin based on trading statistics
        Taken from https://docs.google.com/spreadsheets/d/1DhuJnfLlnw4xl48Li8fdhn3bKT0JJNv5iLMJ0PfE_7Y/edit?gid=0#gid=0
        Args:
            max_risk: Fraction of capital you're prepared to lose before declaring ruin. (0.5 for 50% of capital.)
        
        Returns:
            Perry Kaufman Risk of Ruin probability of losing a specific portion of trading capital
        """
        # probability of losing a trade
        loss_rate = 1 - win_rate

        # Calculate Edge: If positive, the system has an advantage.
        edge = win_rate-loss_rate

        if edge<=0:
            return 1.0 # Maximum risk of ruin
        
        # Calculate inner term: odds against the trader.
        inner_term = (1-edge)/(1+edge)

        # The amount of capital that would constitute "ruin" if lost.
        risk_capital = initial_capital*max_risk

        # Ensure max_loss is negative and non-zero
        if max_loss >= 0:
            max_loss = -abs(max_loss)  # Convert to negative if positive

        if max_loss == 0:
            return 1.0  # Maximum risk of ruin if no loss limit is set
        
        # calculate the exponent term: how many consecutive maximum losses would lead to ruin.
        exp = risk_capital/-max_loss
        pk_risk_of_ruin = inner_term ** exp
    
        return pk_risk_of_ruin

    def calculate_equity_curve(self, trades):
        """
        Generate an equity curve based on the given sequence of trades.
        
        Args:
            trades (array-like): Sequence of trade profits/losses
        
        Returns:
            numpy.ndarray: Cumulative sum representing portfolio growth (true dollar values)
        """
        return np.cumsum(trades)+self.capital
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate the maximum drawdown from an equity curve.
        
        Args:
            equity_curve (array-like): Cumulative portfolio values
        
        Returns:
            float: Maximum drawdown percentage
        """
        if len(equity_curve) == 0:
            return 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown)*100

    def analyze_simulation(self, equity_curve):
        """
        Analyze the simulated equity curve for performance metrics.
        
        Args:
            equity_curve (array-like): Cumulative portfolio values
        
        Returns:
            dict: Performance metrics including max drawdown and final return
        """
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        net_profit = equity_curve[-1]-self.capital
        return {
            "max_drawdown": max_drawdown, 
            "net_profit": net_profit
        }
    
    def calculate_confidence_interval(self, data, conf_level):
        """
        Calculate confidence interval for given data and confidence level.
        
        Args:
            data (list): Numerical data to analyze
            conf_level (float): Confidence level percentage
        
        Returns:
            tuple: (lower_bound, mean, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_error = np.std(data, ddof=1) / np.sqrt(n)
        
        # Calculate t-value for given confidence level
        t_value = stats.t.ppf((1 + conf_level) / 2, df=n-1)
        
        # Calculate margin of error
        margin_of_error = t_value * std_error
        
        # Calculate confidence interval
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
        return lower_bound, mean, upper_bound
    
    def aggregate_results(self, results):
        """
        Aggregate results of all simulations into summary statistics.
        
        Args:
            results (list): List of simulation results
        
        Returns:
            dict: Summary statistics with confidence intervals
        """
        # Define confidence levels
        confidence_levels = [
            0.50, 0.60, 0.70, 0.80, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1
        ]

        drawdowns = [res["max_drawdown"] for res in results]
        returns = [res["net_profit"] for res in results]
        
        # Prepare table data
        table_data = []
        for conf_level in confidence_levels:
            # Calculate intervals for returns and drawdowns
            returns_stats = self.calculate_confidence_interval(returns, conf_level)
            drawdown_stats = self.calculate_confidence_interval(drawdowns, conf_level)
            
            # Create row for the table
            table_data.append([
                f"{conf_level*100}%",
                f"[{returns_stats[0]:.4f} - {returns_stats[1]:.2f} - {returns_stats[2]:.4f}]",
                f"[{drawdown_stats[0]:.4f} - {drawdown_stats[1]:.2f} - {drawdown_stats[2]:.4f}]"
            ])
        
        # Create and print the table
        headers = [
            "Confidence Level", 
            "Net Profit CI", 
            "Max Drawdown % CI"
        ]
        
        # Print the table using tabulate
        table = tabulate(
            table_data, 
            headers=headers, 
            tablefmt='pretty'
        )
        
        print(table)

    def simulate_trades(self, n_simulations=1000, fraction=1, replace=False):
        """
        Perform Monte Carlo simulation of trades.
        
        Args:
            n_simulations (int): Number of simulations to run
            replace (bool): Whether to sample with replacement
            fraction (float): Fraction of trades to simulate
        """
        # Validate input
        if self.realized_pnl.empty:
            raise ValueError("No trades have been added. Use add_trade() first.")
        
        results = []
        
        for _ in range(n_simulations):
            # Randomize the sequence of trades
            shuffled_trades = self.realized_pnl.sample(frac=fraction, replace=replace).reset_index(drop=True).values
            
            # Calculate equity curve for the randomized trade sequence
            equity_curve = self.calculate_equity_curve(shuffled_trades)
            
            # Analyze and store the results of this simulation
            results.append(self.analyze_simulation(equity_curve))
        
        # Aggregate results across all simulations
        self.aggregate_results(results)

#* Custom Sizer
class CustomSizer(bt.Sizer):
    params = (('stake', 1),("risk_percentage",0.01))

    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        `comminfo`: The CommissionInfo instance that contains
        information about the commission for the data and allows
        calculation of position value, operation cost, commision for the
        operation
        
        This method returns the desired size (absolute) for the buy/sell operation.
        If 0 is returned nothing will be executed.
        """
        # # can use complete portfolio value through self.broker.getvalue()
        # position = self.strategy.getposition(data) 
        # size = self.p.stake * (1 + (position.size != 0)) # double the stake

        risk_amount = cash * self.params.risk_percentage
        atr_value = self.strategy.atr[0]
        current_price = self.strategy.data.close[0] 
        if atr_value:
            position_size = risk_amount / (atr_value*current_price)
            return int(position_size)
        return self.params.stake
# cerebro.addsizer_byidx(idx, CustomSizer, stake=70)

#* Custom Observers
class BuySell(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='^', markersize=8.0, color='lime', fillstyle='full'),
        sell=dict(marker='v', markersize=8.0, color='red', fillstyle='full')
    )

class Trades(bt.observers.Trades):
    plotlines = dict(
        pnlplus=dict(_name='Positive',
                    marker='o', color='blue',
                    markersize=4.0, fillstyle='full'),
        pnlminus=dict(_name='Negative',
                    marker='o', color='red',
                    markersize=4.0, fillstyle='full')
    )

# cerebro.addobserver(bt.observers.Broker)
# Override BuySell and Trades
# cerebro.addobserver(BuySell)
# cerebro.addobserver(Trades)
# cerebro.run(stdstats=False)


#* Custom Commissions Class
class CustomComm(bt.CommInfoBase):
    params = dict(stocklike=True,
                  commtype=bt.CommInfoBase.COMM_PERC, # Apply % Commission
                  percabs=True, # pass perc as 0.xx
    
                # Custom params for the discount
                discount_volume=5000,  # minimum contracts to achieve discount
                discount_perc=50.0)  # 50.0% discount

    def _getcommission(self, size, price, pseudoexec):
        '''Calculates the commission of an operation at a given price

        pseudoexec: if True the operation has not yet been executed.
        A broker offers a 50% discount on futures round-trip commission once the amount of negotiated contracts has exceeeded 5000 units
        Before an order is executed, Backtrader may call _getcommission multiple times to:
        Estimate available cash after execution.
        Simulate what would happen to ensure constraints are met (e.g., margin requirements, commission affordability) for an order.
        The pseudoexec flag differentiates between actual executions (pseudoexec=False) and these preparatory calculations (pseudoexec=True). 
        Without pseudoexec, state changes like updating negotiated_volume would occur during pre-calculations, potentially causing:
        Overestimated trading volume.
        Premature triggering of the discount.
        '''
        if self.negotiated_volume > self.p.discount_volume:
           actual_discount = self.p.discount_perc / 100.0
        else:
           actual_discount = 0.0

        commission = self.p.commission * (1.0 - actual_discount)
        commvalue = size * price * commission

        if not pseudoexec:
           # keep track of actual real executed size for future discounts
           self.negotiated_volume += size

        return commvalue

    # If the broker doesn’t consider weekends or bank holidays when calculating the interest rate.
    def _get_credit_interest(self, size, price, days, dt0, dt1):
        # days: number of days elapsed since last credit calculation (this is (dt0 - dt1).days)
        return 1.0 * abs(size) * price * (self.p.interest / 365.0)
# cerebro.broker.addcommissioninfo(CustomComm(), name='CustomCommission')


#* Custom Indicator class
class MyIndicator(bt.Indicator):
    lines = ('overunder',)
    params = dict(period=20, movav=bt.indicators.MovAv.Simple) #* Backtrader built-in indicators
                                 # bt.talib.SMA(self.data, timeperiod=self.p.period) #* ta-lib indicators
    #* Object-wide plotting options
    plotinfo = dict(plot=True,
                subplot=False, # true for oscillators
                plotname='MyIndicator+',
                plotabove=False, # whether to plot above the data. Else plot below. Only if subplot is true.
                plotlinelabels=False, # whether to plot the names of the individudal lines along the data in the legend on the chart when subplot=False
                plotlinevalues=True, # the legend for the lines in Indicators and Observers has the last plotted value. per-line basis with _plotvalue for each line
                plotvaluetags=True, # value tag with the last value is plotted on the right hand side of the line. per-line basis with _plotvaluetag for each line
                plotymargin=0.15, # margin to add to the top and bottom of individual subcharts on the graph
                plotyticks=[20.0, 80.0], # value ticks to plot
                plothlines=[20.0, 50.0, 80.0], # horizontal lines to plot like overbought and oversold.
                plotyhlines=[], # This can take over both plothlines and plotyticks but they have precedence over the values present in this option
                plotforce=False, # This is a last resort mechanism to try to enforce plotting.
                plotmaster=None, # an Indicator/Observer has a master which is the data on which is working.
                plotylimited=True, # other lines on the data plot don’t change the scale.
           )
    
    #* Line specific plotting options
    # Plot the line "overunder" (the only one) with dash style
    # ls stands for linestyle and is directly passed to matplotlib
    plotlines = dict(
        # line = 
        overunder=dict(
            _name="OverUnder", #which changes the plot name of a specific line
            _method="plot", # or 'bar' which chooses the plotting method matplotlib will use for the element.
            # _fill_gt('another_line', ('red', 0.50)) # Allow filling between the given line and above: Another line or A numeric value
            ls='--', alpha=0.50, width=1.0 # other matplotlib args
            )
        )

    def _plotlabel(self):
        # This method returns a list of labels that will be displayed in between
        # parentheses after the name of the Indicators or Observer on the plot

        # The period must always be there
        plabels = [self.p.period]

        # Put only the moving average if it's not the default one
        plabels += [self.p.movav] * self.p.notdefault('movav')

        return plabels

    def __init__(self):
        movav = self.p.movav(self.data, period=self.p.period)
        self.l.overunder = bt.Cmp(movav, self.data)
    
class DummyInd(bt.Indicator):
    lines = ('dummyline',)

    params = (('value', 5),)

    def __init__(self):
        self.lines.dummyline = bt.Max(0.0, self.params.value)