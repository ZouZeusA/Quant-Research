import backtrader as bt
import yfinance as yf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MyBTclasses import RiskManagement

INDS = ["sma", "ema", "stoc", "rsi", "macd", "bollinger", "aroon", "ultimate", "trix", "kama", "adxr", "dema", "ppo", "tema", "roc", "williamsr", ]

#* Custom Analyzers
def AddAnalyzers(cerebro):
    # A trade is defined as opening a position and taking the position back to 0.
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='Trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    # Add Analyzer for SQN
    # SquareRoot(NumberTrades) * Average(TradesProfit) / StdDev(TradesProfit)
    # The sqn value should be deemed reliable when the number of trades >= 30
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    #* Pyfolio
    # cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    # # In Main
    # portfolio_stats = results[0].analyzers.getbyname('PyFolio')
    # returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    
    return cerebro

#* Create a data feed from either a CSV file or directly from yfinance.
def get_data_feed(symbol, **kwargs):
    """
    Create a data feed from either a CSV file or directly from yfinance.
    
    Args:
        symbol (str): Asset name or CSV file path
        **kwargs: Additional arguments for data feed
        
    Returns:
        A Backtrader data feed object
    """
    if symbol.endswith('.csv'):
        return bt.feeds.PandasData(
            dataname=pd.read_csv(symbol,parse_dates=True, index_col="Date"),
            fromdate=kwargs["start"],
            todate=kwargs["end"],
            timeframe=bt.TimeFrame.Days,
            compression=1, # Number of actual bars per bar. each n bars become a bar 
            name=symbol.replace(".csv",""),
            tz='UTC' # Data feed timezone
        )
    else:
        data = yf.download(symbol, **kwargs).xs(symbol, axis=1, level=1)
        data.to_csv(f"{symbol}.csv")
        return bt.feeds.PandasData(
            dataname=data,
            timeframe=bt.TimeFrame.Days,
            compression=1,  # Number of actual bars per bar. each n bars become a bar 
            name=symbol,
            tz='UTC'  # Data feed timezone
            # * Settings for out-of-sample data
            # fromdate=datetime.datetime(2018, 1, 1),
            # todate=datetime.datetime(2019, 12, 25))
        )

def get_benchmark(symbol, start, end):
        # Get or create benchmark data
    bench_file = f'{symbol}.csv'
    
    # Check if benchmark data file exists and is up to date
    if not os.path.exists(bench_file):
        print(f"Downloading {symbol} benchmark data...")
        # Download SPY data for the entire period specified in args
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
        bench_data = yf.download(symbol, start=start_date, end=end_date, multi_level_index=False)
        # Save to CSV for future use
        bench_data.to_csv(bench_file)
    else:
        print(f"Using existing {symbol} benchmark data from {bench_file}")
    
    # Load benchmark data from CSV
    return pd.read_csv(bench_file, index_col=0, parse_dates=True)

#* SQN Categorization Function
def categorize_sqn(sqn_value):
    if sqn_value < 1.6:
        return "Poor"
    elif 1.6 <= sqn_value < 2.0:
        return "Below Average"
    elif 2.0 <= sqn_value < 2.5:
        return "Average"
    elif 2.5 <= sqn_value < 3.0:
        return "Good"
    elif 3.0 <= sqn_value <= 5.0:
        return "Excellent"
    elif 5.1 <= sqn_value <= 6.9:
        return "Superb"
    elif sqn_value >= 7.0:
        return "Holy Grail?"
    else:
        return "Unknown"


#* Trade Analytics
def generate_streaks(analysis):

    streaks = analysis['Streaks']
    streak_data = {
        'Streak Type': ['Current Win', 'Longest Win', 'Current Loss', 'Longest Loss'],
        'Streak Length': [
            streaks['Current Win Streak'],
            streaks['Longest Win Streak'],
            streaks['Current Loss Streak'],
            streaks['Longest Loss Streak']
        ]
    }

    # Horizontal bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['skyblue', 'blue', 'salmon', 'red']

    ax.barh(streak_data['Streak Type'], streak_data['Streak Length'], color=colors)
    ax.set_title('Win/Loss Streaks', fontsize=14)
    ax.set_xlabel('Length', fontsize=12)
    ax.set_ylabel('Streak Type', fontsize=12)

    # Annotate bars
    for index, value in enumerate(streak_data['Streak Length']):
        ax.text(value + 0.2, index, str(value), va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

def generate_won_lost_pie(analysis):
    won_trades = analysis['Won Trades']['Total']
    lost_trades = analysis['Lost Trades']['Total']

    # Calculate percentages
    long_won_percent = analysis['Long Trades']['Won']['Total'] / won_trades * 100
    short_won_percent = analysis['Short Trades']['Won']['Total'] / won_trades * 100
    long_lost_percent = analysis['Long Trades']['Lost']['Total'] / lost_trades * 100
    short_lost_percent = analysis['Short Trades']['Lost']['Total'] / lost_trades * 100

    # Pie chart data
    won_sizes = [long_won_percent, short_won_percent]
    lost_sizes = [long_lost_percent, short_lost_percent]
    labels = ['Long Trades', 'Short Trades']

    # Plot pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Won Trades Pie Chart
    axes[0].pie(won_sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'salmon'])
    axes[0].set_title(f' {won_trades} Won Trades Distribution', fontsize=14)

    # Lost Trades Pie Chart
    axes[1].pie(lost_sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'salmon'])
    axes[1].set_title(f'{lost_trades} Lost Trades Distribution', fontsize=14)

    plt.tight_layout()
    plt.show()

def generate_won_lost_length_chart(analysis):
    # Extract data
    won_lengths = analysis['Trade Lengths']['Won Trades Length']
    lost_lengths = analysis['Trade Lengths']['Lost Trades Length']

    # Prepare data for bar chart
    categories = ['Total', 'Average', 'Max Length', 'Min Length']
    won_values = [round(won_lengths[cat],3) for cat in categories]
    lost_values = [round(lost_lengths[cat],3) for cat in categories]

    # Create the bar chart
    x = np.arange(len(categories))  # x-axis positions
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for won and lost trades
    bars1 = ax.bar(x - width/2, won_values, width, label='Won Trades', color='skyblue')
    bars2 = ax.bar(x + width/2, lost_values, width, label='Lost Trades', color='salmon')

    # Add labels and title
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Bars', fontsize=12)
    ax.set_title('Comparison of Won vs. Lost Trades Lengths', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def generate_long_short_chart(analysis):
    # Extract data for bar chart comparison
    categories = ['Total PnL', 'Average PnL']
    long_values = [round(analysis['Long Trades']['PnL']['Total'], 3), round(analysis['Long Trades']['PnL']['Average'], 3)]
    short_values = [round(analysis['Short Trades']['PnL']['Total'], 3), round(analysis['Short Trades']['PnL']['Average'], 3)]

    # Create the bar chart
    x = np.arange(len(categories))  # x-axis positions
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for long and short trades
    bars1 = ax.bar(x - width/2, long_values, width, label='Long Trades', color='skyblue')
    bars2 = ax.bar(x + width/2, short_values, width, label='Short Trades', color='salmon')

    # Add labels and title
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Comparison of Long vs. Short Trades PnL', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Detailed breakdown for Won and Lost trades comparison
    won_categories = ['Total PnL', 'Average PnL', 'Max PnL']
    long_won_values = [
        round(analysis['Long Trades']['Won']['PnL']['Total'], 3),
        round(analysis['Long Trades']['Won']['PnL']['Average'], 3),
        round(analysis['Long Trades']['Won']['PnL']['Max PnL'], 3)
    ]
    short_won_values = [
        round(analysis['Short Trades']['Won']['PnL']['Total'], 3),
        round(analysis['Short Trades']['Won']['PnL']['Average'], 3),
        round(analysis['Short Trades']['Won']['PnL']['Max PnL'], 3)
    ]

    lost_categories = ['Total PnL', 'Average PnL', 'Max PnL']
    long_lost_values = [
        round(analysis['Long Trades']['Lost']['PnL']['Total'], 3),
        round(analysis['Long Trades']['Lost']['PnL']['Average'], 3),
        round(analysis['Long Trades']['Lost']['PnL']['Max PnL'], 3)
    ]
    short_lost_values = [
        round(analysis['Short Trades']['Lost']['PnL']['Total'], 3),
        round(analysis['Short Trades']['Lost']['PnL']['Average'], 3),
        round(analysis['Short Trades']['Lost']['PnL']['Max PnL'], 3)
    ]

    # Create separate bar charts for won and lost trades
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Won Trades
    x_won = np.arange(len(won_categories))
    bars1_won = axes[0].bar(x_won - width/2, long_won_values, width, label='Long Trades', color='skyblue')
    bars2_won = axes[0].bar(x_won + width/2, short_won_values, width, label='Short Trades', color='salmon')
    axes[0].set_title('Won Trades PnL', fontsize=14)
    axes[0].set_xticks(x_won)
    axes[0].set_xticklabels(won_categories, fontsize=10)
    axes[0].legend(fontsize=10)

    # Annotate Won Trades bars
    for bars in [bars1_won, bars2_won]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # Lost Trades
    x_lost = np.arange(len(lost_categories))
    bars1_lost = axes[1].bar(x_lost - width/2, long_lost_values, width, label='Long Trades', color='skyblue')
    bars2_lost = axes[1].bar(x_lost + width/2, short_lost_values, width, label='Short Trades', color='salmon')
    axes[1].set_title('Lost Trades PnL', fontsize=14)
    axes[1].set_xticks(x_lost)
    axes[1].set_xticklabels(lost_categories, fontsize=10)
    axes[1].legend(fontsize=10)

    # Annotate Lost Trades bars
    for bars in [bars1_lost, bars2_lost]:
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def format_trade_analysis(trade_analyzer):
    analysis = {
        'Total Trades': trade_analyzer['total']['total'],
        'Open Trades': trade_analyzer['total']['open'],
        'Closed Trades': trade_analyzer['total']['closed'],
        
        'Streaks': {
            'Current Win Streak': trade_analyzer['streak']['won']['current'],
            'Longest Win Streak': trade_analyzer['streak']['won']['longest'],
            'Current Loss Streak': trade_analyzer['streak']['lost']['current'],
            'Longest Loss Streak': trade_analyzer['streak']['lost']['longest']
        },
        
        'PnL': {
            'Gross PnL': {
                'Total': trade_analyzer['pnl']['gross']['total'],
                'Average': trade_analyzer['pnl']['gross']['average']
            },
            'Net PnL': {
                'Total': trade_analyzer['pnl']['net']['total'],
                'Average': trade_analyzer['pnl']['net']['average']
            }
        },
        
        'Won Trades': {
            'Total': trade_analyzer['won']['total'],
            'PnL': {
                'Total': trade_analyzer['won']['pnl']['total'],
                'Average': trade_analyzer['won']['pnl']['average'],
                'Max PnL': trade_analyzer['won']['pnl']['max']
            }
        },
        
        'Lost Trades': {
            'Total': trade_analyzer['lost']['total'],
            'PnL': {
                'Total': trade_analyzer['lost']['pnl']['total'],
                'Average': trade_analyzer['lost']['pnl']['average'],
                'Max PnL': trade_analyzer['lost']['pnl']['max']
            }
        },
        
        'Long Trades': {
            'Total': trade_analyzer['long']['total'],
            'PnL': {
                'Total': trade_analyzer['long']['pnl']['total'],
                'Average': trade_analyzer['long']['pnl']['average']
            },
            'Won': {
                'Total': trade_analyzer['long']['won'],
                'PnL': {
                    'Total': trade_analyzer['long']['pnl']['won']['total'],
                    'Average': trade_analyzer['long']['pnl']['won']['average'],
                    'Max PnL': trade_analyzer['long']['pnl']['won']['max']
                }
            },
            'Lost': {
                'Total': trade_analyzer['long']['lost'],
                'PnL': {
                    'Total': trade_analyzer['long']['pnl']['lost']['total'],
                    'Average': trade_analyzer['long']['pnl']['lost']['average'],
                    'Max PnL': trade_analyzer['long']['pnl']['lost']['max']
                }
            }
        },
        
        'Short Trades': {
            'Total': trade_analyzer['short']['total'],
            'PnL': {
                'Total': trade_analyzer['short']['pnl']['total'],
                'Average': trade_analyzer['short']['pnl']['average']
            },
            'Won': {
                'Total': trade_analyzer['short']['won'],
                'PnL': {
                    'Total': trade_analyzer['short']['pnl']['won']['total'],
                    'Average': trade_analyzer['short']['pnl']['won']['average'],
                    'Max PnL': trade_analyzer['short']['pnl']['won']['max']
                }
            },
            'Lost': {
                'Total': trade_analyzer['short']['lost'],
                'PnL': {
                    'Total': trade_analyzer['short']['pnl']['lost']['total'],
                    'Average': trade_analyzer['short']['pnl']['lost']['average'],
                    'Max PnL': trade_analyzer['short']['pnl']['lost']['max']
                }
            }
        },
        
        'Trade Lengths': {
            'Total Bars': trade_analyzer['len']['total'],
            'Average Length': trade_analyzer['len']['average'],
            'Max Length': trade_analyzer['len']['max'],
            'Min Length': trade_analyzer['len']['min'],
            
            'Won Trades Length': {
                'Total': trade_analyzer['len']['won']['total'],
                'Average': trade_analyzer['len']['won']['average'],
                'Max Length': trade_analyzer['len']['won']['max'],
                'Min Length': trade_analyzer['len']['won']['min']
            },
            
            'Lost Trades Length': {
                'Total': trade_analyzer['len']['lost']['total'],
                'Average': trade_analyzer['len']['lost']['average'],
                'Max Length': trade_analyzer['len']['lost']['max'],
                'Min Length': trade_analyzer['len']['lost']['min']
            }
        }
    }
    # Display the formatted analysis
    # for key, value in analysis.items():
    #     if isinstance(value, dict):
    #         print(f"{key}:")
    #         for sub_key, sub_value in value.items():
    #             print(f"  {sub_key}: {sub_value}")
    #     else:
    #         print(f"{key}: {value}")

    return analysis

def generate_statistics(raw_analysis, trades_netpnl, initial_capital, max_risk=0.2):
    """
    Capital Fraction: Fraction of capital at risk, expressed as X% of capital.
    """
    analysis = format_trade_analysis(raw_analysis)
    # Total Trades and Direction
    total_trades = analysis['Total Trades']
    remaining_open_trades = analysis['Open Trades']

    won_trades = analysis['Won Trades']['Total']
    closed_trades = analysis['Closed Trades']
    # Win Rate
    # percentage of winning trades, measures the proportion of trades that generate a profit.
    win_rate = won_trades / closed_trades if closed_trades > 0 else 0
    loss_rate = 1 - win_rate  # Complement of win rate
    
    # Profit Factor
    # total profit generated by winning trades to the total loss incurred by losing trades.
    net_winning_profit = analysis['Won Trades']['PnL']['Total']
    net_losing_loss = -analysis['Lost Trades']['PnL']['Total']  # Losses are negative in most implementations
    profit_factor = net_winning_profit / net_losing_loss if net_losing_loss != 0 else float('inf')
    
    # Average Return per Trade
    avg_return_per_win = analysis['Won Trades']['PnL']['Average'] # total profit made by winning trades / the number of winning trades
    avg_return_per_loss = analysis['Lost Trades']['PnL']['Average'] # total loss made by losing trades / the number of losing trades
    
    # Profit to loss Ratio
    profit_to_loss_ratio = avg_return_per_win / -avg_return_per_loss if avg_return_per_loss != 0 else float('inf')
    
    # Expectancy
    expectancy = (win_rate * avg_return_per_win) - (loss_rate * abs(avg_return_per_loss))

    # Max Win and Max Loss
    max_win = analysis['Won Trades']['PnL']['Max PnL']
    max_loss = analysis['Lost Trades']['PnL']['Max PnL']

    # Risk to Reward Ratio
    risk_to_reward_ratio = abs(max_loss / max_win) if max_win != 0 else float('inf')

    # * Risk Management
    risk_management = RiskManagement(trades_netpnl, initial_capital)
    # Perry Kaufman's Risk of Ruin
    perry_ror = risk_management.pk_risk_of_ruin(win_rate, max_loss, initial_capital, max_risk)   

    # Ralph Vince's Risk of Ruin
    ralph_ror = risk_management.rv_risk_of_ruin(win_rate, avg_return_per_win, avg_return_per_loss, initial_capital, max_risk)

    # Optimal F
    optimal_f = risk_management.optimal_f()
    
    # Summary of statistics
    general_statistics = {
        "Total Trades": total_trades,
        "Remaining Open Trades": remaining_open_trades,
        "Total Bars": analysis['Trade Lengths']['Total Bars'],
        "Average Trade Time": analysis['Trade Lengths']['Average Length'],
        "Total Gross P&L" : analysis["PnL"]["Gross PnL"]['Total'],
        "Total Fees" : analysis["PnL"]["Gross PnL"]['Total']-analysis["PnL"]["Net PnL"]['Total'],
        "Average Net Return per any trade": analysis["PnL"]["Net PnL"]['Average'],
    }

    print("\t\tOverall Trade Analysis")
    print("*"*60)
    for key, value in general_statistics.items():
        print(f"{key}: {value}")

    generate_streaks(analysis)
    generate_won_lost_pie(analysis)
    generate_won_lost_length_chart(analysis)
    print("Max Drawdown Duration without weekends and holidays (in bars):", analysis['Trade Lengths']['Lost Trades Length']['Max Length'])
    generate_long_short_chart(analysis)

    print("\t\tTrade Statistics")
    print("*"*60)
    statistics = {
        "Win Rate": f"{win_rate:.2%}",
        "Profit Factor": profit_factor,
        "Profit to Loss Ratio": profit_to_loss_ratio,
        "Expectancy": expectancy,
        "Risk to Reward Ratio": risk_to_reward_ratio,
        "Optimal fraction to bet" : f"{optimal_f:.3%}",
        "Perry Kaufman's Risk of Ruin": f"{perry_ror:.3%}",
        "Ralph Vince's Risk of Ruin": f"{ralph_ror:.3%}",
    }
    
    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\n\t\tMonte Carlo Simulation")
    print("*"*60)
    risk_management.simulate_trades(n_simulations=1000, fraction=0.8, replace=False)
   
   