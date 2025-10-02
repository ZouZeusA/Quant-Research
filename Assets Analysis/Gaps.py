import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class GapAnalyzer:
    
    def __init__(self, data):
        """
        Initialize Gap Analyzer with multi-asset OHLCV data
        
        Parameters:
        data : DataFrame
            Multi-column DataFrame with hierarchical columns:
            - Level 0: Features (Open, High, Low, Close, etc.) 
            - Level 1: Tickers
            - Index: DatetimeIndex
        """
        self.data = data
        self.tickers = self._extract_tickers()
        self.gap_data = {}
        self.gap_stats = {}

    def _extract_tickers(self):
        """Extract ticker symbols from MultiIndex columns"""
        if hasattr(self.data.columns, 'levels'):
            return self.data.columns.get_level_values(1).unique().tolist()
        else:
            return [self.data.columns.name] if self.data.columns.name else ['Asset']
    
    def calculate_gaps(self):
        """Calculate overnight gaps for all assets"""
        print("Calculating overnight gaps...")
        
        for ticker in self.tickers:
            try:
                # Extract OHLC data for this ticker
                if len(self.tickers) > 1:
                    open_prices = self.data['Open'][ticker]
                    high_prices = self.data['High'][ticker]
                    low_prices = self.data['Low'][ticker]
                    close_prices = self.data['Close'][ticker]
                    volume = self.data['Volume'][ticker] if 'Volume' in self.data.columns.get_level_values(0) else None
                else:
                    open_prices = self.data['Open']
                    high_prices = self.data['High']
                    low_prices = self.data['Low']
                    close_prices = self.data['Close']
                    volume = self.data['Volume'] if 'Volume' in self.data.columns else None
                
                # Calculate gaps
                prev_close = close_prices.shift(1)
                gap_percentage = ((open_prices - prev_close) / prev_close) * 100
                
                # Calculate gap day returns
                gap_day_return = ((close_prices - open_prices) / open_prices) * 100
                
                # Calculate intraday range
                intraday_range = ((high_prices - low_prices) / open_prices) * 100
                
                # Create comprehensive gap dataset
                gap_df = pd.DataFrame({
                    'Date': open_prices.index,
                    'Prev_Close': prev_close,
                    'Open': open_prices,
                    'High': high_prices,
                    'Low': low_prices,
                    'Close': close_prices,
                    'Volume': volume if volume is not None else np.nan,
                    'Gap_Percentage': gap_percentage,
                    'Gap_Day_Return': gap_day_return,
                    'Intraday_Range': intraday_range,
                    'Gap_Direction': np.where(gap_percentage > 0, 'Up', 
                                            np.where(gap_percentage < 0, 'Down', 'Flat'))
                }).dropna(subset=['Gap_Percentage'])
                
                # Add gap size categories
                gap_df['Gap_Size_Category'] = pd.cut(
                    abs(gap_df['Gap_Percentage']),
                    bins=[0, 0.5, 1.0, 2.0, 5.0, float('inf')],
                    labels=['Small (0-0.5%)', 'Medium (0.5-1%)', 'Large (1-2%)', 
                           'Very Large (2-5%)', 'Extreme (>5%)']
                )
                
                # Add time features
                gap_df['Year'] = gap_df['Date'].dt.year
                gap_df['Month'] = gap_df['Date'].dt.month
                gap_df['DayOfWeek'] = gap_df['Date'].dt.day_name()
                gap_df['IsMonday'] = gap_df['Date'].dt.dayofweek == 0
                
                self.gap_data[ticker] = gap_df.set_index('Date')
                print(f"✓ Processed {ticker}: {len(gap_df)} observations")
                
            except Exception as e:
                print(f"✗ Error processing {ticker}: {str(e)}")
                continue
    
    def analyze_gap_distributions(self):
        """Analyze statistical properties of gaps"""
        print("\nAnalyzing gap distributions...")
        
        for ticker in self.gap_data.keys():
            df = self.gap_data[ticker]
            
            # Basic statistics
            gap_stats = {
                'total_observations': len(df),
                'gap_up_count': len(df[df['Gap_Direction'] == 'Up']),
                'gap_down_count': len(df[df['Gap_Direction'] == 'Down']),
                'gap_flat_count': len(df[df['Gap_Direction'] == 'Flat']),
                'mean_gap': df['Gap_Percentage'].mean(),
                'std_gap': df['Gap_Percentage'].std(),
                'skewness': stats.skew(df['Gap_Percentage']),
                'kurtosis': stats.kurtosis(df['Gap_Percentage']),
                'min_gap': df['Gap_Percentage'].min(),
                'max_gap': df['Gap_Percentage'].max()
            }
            
            # Percentile analysis
            percentiles = [0.01, 0.1, 1, 5, 50, 95, 99, 99.9 , 99.99]
            for p in percentiles:
                gap_stats[f'p{p}'] = np.percentile(df['Gap_Percentage'], p)
            
            # Gap frequency by size
            gap_stats['gap_size_distribution'] = df['Gap_Size_Category'].value_counts().to_dict()
            
            # Day-of-week analysis
            gap_stats['day_of_week_stats'] = df.groupby('DayOfWeek')['Gap_Percentage'].agg(
                    ['count', 'mean', 'std']
            ).to_dict()
            
            # Monday effect
            monday_gaps = df[df['IsMonday']]['Gap_Percentage']
            other_gaps = df[~df['IsMonday']]['Gap_Percentage']
            gap_stats['monday_effect'] = {
                'monday_mean': monday_gaps.mean(),
                'other_days_mean': other_gaps.mean(),
                'monday_std': monday_gaps.std(),
                'other_days_std': other_gaps.std(),
                't_stat': stats.ttest_ind(monday_gaps, other_gaps)[0],
                'p_value': stats.ttest_ind(monday_gaps, other_gaps)[1]
            }
            
            # Volatility relationship
            gap_stats['volatility_correlation'] = df['Gap_Percentage'].corr(df['Intraday_Range'])
            
            self.gap_stats[ticker] = gap_stats
            print(f"✓ Analyzed {ticker} distribution")
    
    def plot_gap_analysis(self, ticker, figsize=(20, 16)):
        """Create comprehensive gap analysis plots"""
        if ticker not in self.gap_data:
            print(f"No data available for {ticker}")
            return
        
        df = self.gap_data[ticker]
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=figsize)
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Gap distribution histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(df['Gap_Percentage'], bins=50, alpha=0.7, color='cyan', edgecolor='white')
        ax1.axvline(0, color='red', linestyle='--', label='No Gap')
        ax1.set_title(f'{ticker} - Gap Distribution')
        ax1.set_xlabel('Gap Percentage (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Gap vs Gap Day Return scatter
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(df['Gap_Percentage'], df['Gap_Day_Return'], 
                             alpha=0.6, c=df['Gap_Percentage'], cmap='RdYlBu', s=20)
        ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='white', linestyle='--', alpha=0.5)
        ax2.set_title(f'{ticker} - Gap vs Gap Day Return')
        ax2.set_xlabel('Gap Percentage (%)')
        ax2.set_ylabel('Gap Day Return (%)')
        plt.colorbar(scatter, ax=ax2, label='Gap %')
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot by gap direction
        ax3 = fig.add_subplot(gs[0, 2])
        gap_directions = ['Down', 'Flat', 'Up']
        returns_by_direction = [df[df['Gap_Direction'] == direction]['Gap_Day_Return'].dropna() 
                               for direction in gap_directions]
        box_plot = ax3.boxplot(returns_by_direction, labels=gap_directions, patch_artist=True)
        colors = ['red', 'yellow', 'green']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_title(f'{ticker} - Returns by Gap Direction')
        ax3.set_ylabel('Next Day Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Day of week analysis
        ax4 = fig.add_subplot(gs[1, 0])
        day_stats = df.groupby('DayOfWeek')['Gap_Percentage'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
        ])
        bars = ax4.bar(range(len(day_stats)), day_stats.values, color='orange', alpha=0.7)
        ax4.set_title(f'{ticker} - Average Gap by Day of Week')
        ax4.set_xlabel('Day of Week')
        ax4.set_ylabel('Average Gap (%)')
        ax4.set_xticks(range(len(day_stats)))
        ax4.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Gap size distribution
        ax5 = fig.add_subplot(gs[1, 1])
        gap_size_counts = df['Gap_Size_Category'].value_counts()
        wedges, texts, autotexts = ax5.pie(gap_size_counts.values, labels=gap_size_counts.index,
                                          autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        ax5.set_title(f'{ticker} - Gap Size Distribution')
        
        # 6. Monthly gap patterns
        ax6 = fig.add_subplot(gs[1, 2])
        monthly_gaps = df.groupby('Month')['Gap_Percentage'].mean()
        ax6.plot(monthly_gaps.index, monthly_gaps.values, marker='o', color='lime', linewidth=2)
        ax6.set_title(f'{ticker} - Seasonal Gap Patterns')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Average Gap (%)')
        ax6.set_xticks(range(1, 13))
        ax6.grid(True, alpha=0.3)
           
        plt.suptitle(f'{ticker} - Comprehensive Gap Analysis', fontsize=16, y=0.95)
        plt.show()
    
    def generate_report(self, ticker):
        """Generate detailed text report for a ticker"""
        if ticker not in self.gap_data:
            print(f"No data available for {ticker}")
            return
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE GAP ANALYSIS REPORT: {ticker}")
        print(f"{'='*60}")
        
        # Basic Statistics
        print(f"\n📊 BASIC GAP STATISTICS")
        print(f"{'─'*40}")
        stats = self.gap_stats[ticker]
        print(f"Total Observations: {stats['total_observations']:,}")
        print(f"Gap Up Events: {stats['gap_up_count']:,} ({stats['gap_up_count']/stats['total_observations']*100:.1f}%)")
        print(f"Gap Down Events: {stats['gap_down_count']:,} ({stats['gap_down_count']/stats['total_observations']*100:.1f}%)")
        print(f"Flat Opens: {stats['gap_flat_count']:,} ({stats['gap_flat_count']/stats['total_observations']*100:.1f}%)")
        print(f"\nGap Statistics:")
        print(f"  Mean Gap: {stats['mean_gap']:.3f}%")
        print(f"  Std Dev: {stats['std_gap']:.3f}%")
        print(f"  Skewness: {stats['skewness']:.3f}")
        print(f"  Kurtosis: {stats['kurtosis']:.3f}")
        print(f"  Range: {stats['min_gap']:.2f}% to {stats['max_gap']:.2f}%")
        
        # Percentile Analysis
        print(f"\n📈 PERCENTILE ANALYSIS")
        print(f"{'─'*40}")
        print(f"0.01th percentile: {stats['p0.01']:.2f}%")
        print(f"0.1th percentile: {stats['p0.1']:.2f}%")
        print(f"1st percentile: {stats['p1']:.2f}%")
        print(f"5th percentile: {stats['p5']:.2f}%")
        print(f"Median: {stats['p50']:.2f}%")
        print(f"95th percentile: {stats['p95']:.2f}%")
        print(f"99th percentile: {stats['p99']:.2f}%")
        print(f"99.9th percentile: {stats['p99.9']:.2f}%")
        print(f"99.99th percentile: {stats['p99.99']:.2f}%")
        
        # Monday Effect
        print(f"\n📅 MONDAY EFFECT ANALYSIS")
        print(f"{'─'*40}")
        monday_effect = stats['monday_effect']
        print(f"Monday Average Gap: {monday_effect['monday_mean']:.3f}%")
        print(f"Other Days Average: {monday_effect['other_days_mean']:.3f}%")
        print(f"Difference: {monday_effect['monday_mean'] - monday_effect['other_days_mean']:.3f}%")
        print(f"Statistical Significance: {'Yes' if monday_effect['p_value'] < 0.05 else 'No'} (p={monday_effect['p_value']:.4f})")

# Run the analysis
if __name__ == "__main__":
    
    # Download data
    print("Downloading data...")
    tickers = ["SPY", "AAPL", "GLD", "EURUSD=X"]
    data = yf.download(tickers, start="2010-01-01", end="2023-12-31", 
                      auto_adjust=False, actions=True)
    
    # Initialize analyzer
    analyzer = GapAnalyzer(data)
    
    # Run analysis
    analyzer.calculate_gaps()
    analyzer.analyze_gap_distributions()
    
    # Generate reports and plots for each ticker
    for ticker in ['SPY']:
        # analyzer.generate_report(ticker)
        analyzer.plot_gap_analysis(ticker)

# plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Add some padding for the suptitle
