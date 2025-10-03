# Event Driven Backtesting Description

The Event Driven Backtesting module is a comprehensive algorithmic trading backtesting framework built on top of the **Backtrader** library. Here's what it contains:

## Core Components
###  Strategy Engine (`MyBTstrategy.py`)
- Extends Backtrader's Strategy class with a very simple trend-following approach (buys when current close < previous close < previous close)
- Comprehensive order management with detailed logging and execution tracking
- Support for market, limit, stop, and bracket orders
- Color-coded logging with detailed trade notifications
- Portfolio rebalancing capabilities with `order_target_xxx` methods

###  Backtesting Engine (`MyBTengine.py`)
- Command-line driven configuration system with extensive parameters
- Support for multiple data feeds and timeframes
- Strategy optimization across parameter ranges
- Integration with QuantStats for performance reporting
- Commission and slippage modeling
- Multiple sizer types (fixed, percentage, custom)
- Comprehensive analysis and plotting capabilities

###  Utility Functions (`MyBTutility.py`)
- Data feed creation from CSV files or Yahoo Finance
- Custom analyzers for performance metrics (Sharpe ratio, SQN)
- SQN (System Quality Number) categorization system
- Comprehensive trade analytics with visualization functions
- Risk management calculations (risk of ruin, optimal f)
- Monte Carlo simulation capabilities for strategy validation

###  Custom Classes (`MyBTclasses.py`)
- **RiskManagement**: Kelly Criterion, risk of ruin calculations, Monte Carlo simulations
- **CustomSizer**: ATR-based position sizing with risk percentage
- **CustomComm**: Volume-based commission discounts
- **MyIndicator**: Custom indicator framework with plotting options
- **BuySell/Trades Observers**: Enhanced visual observers for trades

## Documentation & Examples
###  Tutorial (`Backtrader.ipynb`)
- Comprehensive 3,294-line Jupyter notebook covering:
  - Strategy creation and cerebro engine setup
  - Line concepts and data handling
  - Order execution types and timing
  - Multi-timeframe analysis and resampling

###  Testing (`testing.py`)
- Example strategy implementation extending MyStrategy
- Command-line execution with error handling

## Specialized Strategy
###  TLT Flow Effect Subdirectory
- Strategy implementation for TLT (Treasury bond ETF) flow analysis
- Optimization script and backtesting results
- Performance visualizations and data files

## Key Features
- ✅ Multi-timeframe support with resampling capabilities
- ✅ Comprehensive risk management with Monte Carlo validation
- ✅ Advanced analytics including SQN, Sharpe ratio, and drawdown analysis
- ✅ Flexible order management with detailed execution tracking
- ✅ Performance visualization with charts and statistics
- ✅ Strategy optimization across parameter ranges
- ✅ Production-ready with proper error handling and logging