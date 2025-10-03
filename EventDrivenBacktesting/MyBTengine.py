from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime

import quantstats as qs
import pandas as pd
import backtrader as bt

from MyBTutility import *

def run_strategy(strategy:bt.Strategy, args=None, sizer_class=None):
    if isinstance(args, list):
        args = parse_args(args)
    else:
        args = parse_args()

    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=True) # Set optreturn=False to get full strategy objects instead of using OptReturn objects (which are more memory-efficient for large optimizations)

    # Broker
    cerebro.broker = bt.brokers.BackBroker(**eval("dict(" + args.broker + ")"))

    # Cash
    cerebro.broker.set_cash(args.cash)

    # Load the Data
    dkwargs = dict()
    if args.fromdate:
        fromdate = datetime.datetime.strptime(args.fromdate, "%Y-%m-%d")
        dkwargs["start"] = fromdate

    if args.todate:
        todate = datetime.datetime.strptime(args.todate, "%Y-%m-%d")
        dkwargs["end"] = todate
    

    data_feed = get_data_feed(args.dataname, **dkwargs)
    cerebro.adddata(data_feed)  # First add the original data - smaller timeframe

    # Add second data feed if specified
    if args.dataname2:
        data2 = get_data_feed(args.dataname2, **dkwargs)
        cerebro.adddata(data2)  # And then the large timeframe

    # Handy dictionary for the argument timeframe conversion
    # Resample the data
    if args.resample:
        tframes = dict(
            daily=bt.TimeFrame.Days, weekly=bt.TimeFrame.Weeks, monthly=bt.TimeFrame.Months
        )
        # Only delivers completed bars of higher timeframe
        cerebro.resampledata(
            data_feed, timeframe=tframes[args.timeframe], compression=args.compression
        )

    # First add the original data - smaller timeframe
    # Better for strategies that need to see how higher timeframe bars develop in real-time
    # cerebro.replaydata(data_feed,
    #                    timeframe=tframes[args.timeframe],
    #                    compression=args.compression)


    # Sizer
    if args.sizer_type == 'fixed':
        cerebro.addsizer(bt.sizers.FixedSize, **eval("dict(" + args.sizer + ")"))
    elif args.sizer_type == 'percentage':
        cerebro.addsizer(bt.sizers.PercentSizer, **eval("dict(" + args.sizer + ")"))
    elif args.sizer_type == 'custom':
        cerebro.addsizer(sizer_class, **eval("dict(" + args.sizer + ")"))

    #* An analyzer analyzes the performance of a single strategy and not the performance of an entire system
    cerebro = AddAnalyzers(cerebro)

        # Add a strategy
    if not args.optimization:
        cerebro.addstrategy(strategy, **eval("dict(" + args.strat + ")"))
    else:
        cerebro.optstrategy(strategy, **eval("dict(" + args.strat + ")"))
        optimized_runs = cerebro.run()
        final_results_list = []
        for run in optimized_runs:
            strategy = run[0]
            sharpe = strategy.analyzers.sharpe_ratio.get_analysis()
            # Get all parameters directly - faster approach
            param_values = []
            # Extract parameter names from args.strat
            param_names = [arg.split('=')[0] for arg in args.strat.split(',') if '=' in arg]
            # Get all parameter values directly
            for param_name in param_names:
                param_values.append(getattr(strategy.params, param_name))
            # Add performance metrics
            param_values.extend([sharpe.get('sharperatio', 0)])
            final_results_list.append(param_values)

        sort_by_sharpe = sorted(final_results_list, key=lambda x: x[-1], 
                                reverse=True)
        print(sort_by_sharpe)
        return None
    
    # Run over everything
    results = cerebro.run(runonce=not args.use_next, **eval("dict(" + args.cerebro + ")"))

    #* Extract SQN Value
    sqn_analysis = results[0].analyzers.sqn.get_analysis()
    sqn_value = sqn_analysis['sqn']
    category = categorize_sqn(sqn_value)
    print(f"System Quality Number (SQN) is: {sqn_value:.2f} --> {category}")
    
    cerebro.broker.setcommission(commission=args.commission,
                             margin=None,
                             mult=1.0,
                             percabs=True,
                             commtype=None,
                             stocklike=False,
                             interest=0.0,
                             interest_long=False,
                             leverage=1.0,
                             automargin=False,
                             name=None)

    cerebro.broker.set_slippage_perc(args.slippage,
                                    slip_open=True,
                                    slip_limit=True,
                                    slip_match=True,
                                    slip_out=False)

    #* Quanstats Report
    if args.quantstats:
        benchmark = get_benchmark(args.benchmark, **dkwargs)
        
        # Align benchmark data with returns data (start from the first date in returns)
        returns = results[0].analyzers.time_return.get_analysis()
        returns = pd.Series(index=returns.keys(), data=returns.values())
        returns.name="returns"
        start_date = returns.index[0]
        benchmark = benchmark[benchmark.index >= start_date]
        
        # Generate quantstats report
        qs.reports.full(returns=returns, benchmark=benchmark["Close"].pct_change())
        print("Quantstats report generated")

    #* Trade Analytics    
    if args.genstats:
        trades_netpnl = pd.Series([trade.pnlcomm for trade in results[0].trades],
                                index=[bt.num2date(trade.dtclose) for trade in results[0].trades])
        trades_netpnl.name="netpnl"
        generate_statistics(results[0].analyzers.Trades.get_analysis(), trades_netpnl, args.cash, args.max_risk)
        print("Trade statistics generated")

    # Plot the result
    if args.plot:  # Basic Usage --plot (no argument): returns args.plot == True
        pkwargs = dict(style="candle")  # default plot
        if args.plot is not True:  # evals to True but is not True
            npkwargs = eval("dict(" + args.plot + ")")  # args were passed
            pkwargs.update(npkwargs)

        cerebro.plot(**pkwargs)


def parse_args(pargs=None):
    """
    It automates the parsing of command-line arguments and provides a clean
    way to define and document what arguments the program accepts.

    HELP: python MyBTengine.py -h

    parser.add_argument('--tags', action='append'): Appends multiple occurrences of the argument
    into a list. python script.py --tags tag1 --tags tag2 | Result: tags is ['tag1', 'tag2'].

    action='count': Counts the number of times an argument is used and stores it as an integer.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Multitimeframe test",
    )

    parser.add_argument(
        "--cerebro",
        required=False,
        default="",
        metavar="kwargs",
        help="Cerebro kwargs in key=value format",
    )

    parser.add_argument(
        "--broker",
        required=False,
        default="",
        metavar="kwargs",
        help="Broker kwargs in key=value format",
    )

    parser.add_argument(
        "--sizer",
        required=False,
        default="percents=1",
        metavar="kwargs",
        help="Sizer kwargs in key=value format",
    )
    
    parser.add_argument(
        "--sizer_type",
        default="percentage",
        required=False,
        choices=["fixed", "percentage", "custom"],
        help=(
            "Type of sizer to use:\n"
            "  fixed: bt.sizers.FixedSize \n"
            "  percentage: bt.sizers.PercentSizer (X percentage of available cash)\n"
            "  custom: sizer_class that extends bt.sizers or CustomSizer from MyBTclasses.py"
        ),
    )

    #* add indicators with ind=nameofindicator,parameter=value...
    # example: ind=sma,period=10
    parser.add_argument(
        "--strat",
        required=False,
        default="",
        metavar="kwargs",
        help="Strategy kwargs in key=value format seperated by a comma with no spaces",
    )

    parser.add_argument(
        "--dataname", default="", required=True, help="Smaller timeframe asset to get from Yfinance or csv file name"
    )

    parser.add_argument(
        "--dataname2", default="", required=False, help="Larger timeframe asset to get from Yfinance or csv file name"
    )

    parser.add_argument(
        "--fromdate",
        required=False,
        default="2000-01-01",
        help="Starting date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--todate",
        required=False,
        default="2024-12-31",
        help="Ending date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--resample",
        default=False,
        action="store_true",
        help="Resample to higher timeframe",
    )

    parser.add_argument(
        "--timeframe",
        default="daily",
        required=False,
        choices=["daily", "weekly", "monthly"],
        help="Timeframe to resample to",
    )

    parser.add_argument(
        "--compression",
        default=1,
        required=False,
        type=int,
        help="Compress n bars into 1",
    )

    parser.add_argument(
        "--optimization",
        action="store_true",  # optimization is True if --optimization is used, otherwise False
        help="Wether to try a Strategy with multiple parameters passed in --strat as \
             parameter=range(1,10)",
    )

    parser.add_argument(
        "--use-next",
        required=False,
        action="store_true",
        help=(
            "Use next (step by step) "
            "instead of once (batch). "
            "Run Indicators in vectorized mode to speed \
             up the entire system. Strategies and Observers \
             will always be run on an event based basis"
        ),
    )

    parser.add_argument(
        "--cash",
        required=False,
        action="store",
        type=float,
        default=100000,
        help=("Cash to start with"),
    )
    
    # Analysis options
    parser.add_argument(
        "--quantstats",
        required=False,
        action="store_true",
        default=False,
        help=("Generate Quantstats performance report"),
    )
    
    parser.add_argument(
        "--genstats",
        required=False,
        action="store_true",
        default=False,
        help=("Generate trades statistics and analysis"),
    )
    
    parser.add_argument(
        "--commission",
        required=False,
        action="store",
        type=float,
        default=0.001,
        help=("Commission percentage for trades (default: 0.001 or 0.1percent)"),
    )
    
    parser.add_argument(
        "--slippage",
        required=False,
        action="store",
        type=float,
        default=0.001,
        help=("Slippage percentage for trades (default: 0.001 or 0.1percent)"),
    )
    
    parser.add_argument(
        "--benchmark",
        required=False,
        action="store",
        type=str,
        default="SPY",
        help=("Benchmark ticker for performance comparison (default: SPY)"),
    )
    
    parser.add_argument(
        "--max_risk",
        required=False,
        action="store",
        type=float,
        default=0.5,
        help=("Fraction of capital you're prepared to lose before declaring ruin. (0.5 for 50 percent of capital.)"),
    )

    # Plot options
    parser.add_argument(
        "--plot",
        "-p",  # shorthand alias for the --plot option
        nargs="?",  # allows for zero or one argument to be provided
        required=False,
        metavar="kwargs",  # name shown in the help message for the argument value
        const=True,  # default value if the argument is provided without a value
        help=(
            "Plot the read data applying any kwargs passed\n"
            "\n"
            "For example:\n"
            "\n"
            '  --plot style="candle" [to plot candles]\n'
        ),
    )

    if pargs is not None:
        return parser.parse_args(
            pargs 
        )  # parse a list of strings as if they were command-line arguments
    return parser.parse_args() # processes the command-line arguments provided by the user and returns them as an object.
