import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats
from typing import List
import pandas as pd
import numpy as np
# Create rolling windows using numpy stride tricks for efficiency
from numpy.lib.stride_tricks import sliding_window_view

#! Runtest for Randomness based on https://online.stat.psu.edu/stat415/book/export/html/837
#! Identifies pattern types (trending vs mean-reverting vs random)

def calculate_runs_vectorized(returns_matrix, window):
    """Calculate runs statistics for a matrix of return windows"""
    # Convert returns to binary signals (1 for positive, 0 for non-positive)
    signs = (returns_matrix > 0).astype(int)
    
    # Count positive and negative observations
    n1 = signs.sum(axis=0)  # positive returns
    n2 = window - n1        # non-positive returns
    
    # Calculate runs for each column (asset)
    # Count sign changes and add 1 for initial run
    sign_changes = (signs[1:] != signs[:-1]).sum(axis=0)
    runs = 1 + sign_changes
    
    # Expected runs under randomness
    mu = 1 + (2 * n1 * n2) / (n1 + n2)
    
    # Standard deviation (handle division by zero)
    numerator = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
    denominator = (n1 + n2)**2 * (n1 + n2 - 1)
    
    # Avoid division by zero
    sigma = np.sqrt(np.where(denominator > 0, numerator / denominator, 0))
    
    # Z-score (handle division by zero)
    Z = np.where(sigma > 0, (runs - mu) / sigma, 0)
    
    # Two-tailed p-value
    p_values = 2 * (1 - stats.norm.cdf(np.abs(Z)))
    
    return runs, mu, sigma, Z, p_values

def vectorized_rolling_runs_test(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Vectorized rolling runs test for multiple assets
    
    Parameters:
    - df: DataFrame with returns for multiple assets (columns = assets, index = dates)
    - window: Rolling window size (default 60)
    
    Returns:
    - DataFrame with runs test results for each asset
    """
    # Prepare results storage
    results = []
  
    for asset in df.columns:
        # Get returns for this asset and drop NAs
        returns = df[asset].dropna()
        
        if len(returns) < window:
            continue
            
        # Create sliding windows
        windows = sliding_window_view(returns.values, window_shape=window)
        
        # Calculate runs statistics for all windows at once
        runs, mu, sigma, Z, p_values = calculate_runs_vectorized(windows.T, window)
        
        # Create dates for each window (using the end date of each window)
        window_dates = returns.index[window-1:]
        
        # Store results
        for i, date in enumerate(window_dates):
            results.append({
                'Date': date,
                'Asset': asset,
                'runs': runs[i],
                'expected_runs': mu[i],
                'std_runs': sigma[i],
                'Z_score': Z[i],
                'p_value': p_values[i],
                'efficient': p_values[i] >= 0.05,
                'pattern_type': 'trending' if runs[i] < mu[i] and p_values[i] < 0.05 else 
                              'mean_reverting' if runs[i] > mu[i] and p_values[i] < 0.05 else 'random'
            })
    
    return pd.DataFrame(results)

def plot_runs_test_results(df_returns: pd.DataFrame, df_prices: pd.DataFrame, 
                          runs_results: pd.DataFrame, assets: List[str] = None):
    """
    Plot runs test results for multiple assets with proper time series alignment.
    
    The visualization  clearly distinguishes between:
    - Red shading: Trending behavior (too few runs)
    - Purple shading: Mean-reverting behavior (too many runs)
    - No shading: Random/efficient periods
    
    Parameters:
    - df_returns: DataFrame with returns
    - df_prices: DataFrame with prices  
    - runs_results: Results from vectorized_rolling_runs_test
    - assets: List of assets to plot (if None, plot all)
    """
    
    if assets is None:
        assets = df_returns.columns.tolist()
    
    assets = assets[:6]  # Limit to 6 assets for readability
    n_assets = len(assets)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(n_assets, 2, figsize=(20, 4*n_assets), 
                            gridspec_kw={'width_ratios': [2, 1]})
    
    if n_assets == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['cyan', 'yellow', 'lime', 'orange', 'magenta', 'lightblue']
    
    for i, asset in enumerate(assets):
        if asset not in runs_results['Asset'].unique():
            continue
            
        # Filter and prepare results for this asset
        asset_results = runs_results[runs_results['Asset'] == asset].copy()
        asset_results = asset_results.set_index('Date').sort_index()
        
        if len(asset_results) == 0:
            continue
        
        # Get price data aligned with results timeframe
        price_data = df_prices[asset].loc[asset_results.index[0]:]

        # Each test result applies to the NEXT period, not the current one
        asset_results['regime_forward'] = asset_results['pattern_type'].shift(-1)
        asset_results['efficient_forward'] = asset_results['efficient'].shift(-1)
        
        # Drop the last row since we can't forward-fill it
        asset_results = asset_results[:-1]
        
        # --- Left subplot: Price with regime shading ---
        ax_price = axes[i, 0]
        ax_price.plot(price_data.index, price_data, 
                     color=colors[i], label=f'{asset} Price', linewidth=1.5)
        
        # Shade periods based on FORWARD-FILLED regime classifications
        for idx in range(len(asset_results) - 1):
            current_date = asset_results.index[idx]
            next_date = asset_results.index[idx + 1]
            
            is_efficient = asset_results['efficient_forward'].iloc[idx]
            regime = asset_results['regime_forward'].iloc[idx]
            
            if pd.notna(regime) and not is_efficient:
                color = 'red' if regime == 'trending' else 'purple'
                ax_price.axvspan(current_date, next_date, color=color, alpha=0.2)
        
        ax_price.set_title(f'{asset} - Price with Market Efficiency Regimes')
        ax_price.set_ylabel('Price')
        ax_price.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        
        # --- Right subplot: p-values with regime shading ---
        ax_pval = axes[i, 1]
        ax_pval.plot(asset_results.index, asset_results['p_value'], 
                    color=colors[i], linewidth=2, marker='o', markersize=3)
        ax_pval.axhline(0.05, color='white', linestyle='--', 
                       label='Significance Threshold (α=0.05)', linewidth=2)
        
        # Shade p-value chart consistently with price chart
        for idx in range(len(asset_results) - 1):
            current_date = asset_results.index[idx]
            next_date = asset_results.index[idx + 1]
            
            is_efficient = asset_results['efficient_forward'].iloc[idx]
            regime = asset_results['regime_forward'].iloc[idx]
            
            if pd.notna(regime):
                if is_efficient:
                    ax_pval.axvspan(current_date, next_date, color='lime', alpha=0.2)
                else:
                    color = 'red' if regime == 'trending' else 'purple'
                    ax_pval.axvspan(current_date, next_date, color=color, alpha=0.2)
        
        ax_pval.set_title(f'{asset} - Runs Test p-values (Rolling Window Results)')
        ax_pval.set_ylabel('p-value')
        ax_pval.set_ylim(0, 1)
        ax_pval.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        ax_pval.legend(loc='upper right', fontsize=8)
        
        # Format x-axis for bottom plots only
        if i == n_assets - 1:
            for ax in [ax_price, ax_pval]:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax_price.set_xticklabels([])
            ax_pval.set_xticklabels([])
        
        # Add comprehensive legend to price chart
        legend_elements = [
            mpatches.Patch(color=colors[i], label=f'{asset} Price'),
            mpatches.Patch(color='lime', alpha=0.3, label='Efficient Market (Random)'),
            mpatches.Patch(color='red', alpha=0.3, label='Trending Regime (Momentum)'),
            mpatches.Patch(color='purple', alpha=0.3, label='Mean-Reverting Regime')
        ]
        ax_price.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.suptitle('Rolling Runs Test Analysis - Market Regime Detection\n' + 
                 'Note: Regime classifications are forward-looking based on rolling window tests', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def summarize_runs_results(runs_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for runs test results by asset
    """
    summary = runs_results.groupby('Asset').agg({
        'p_value': ['mean', 'std', 'min', 'max'],
        'efficient': ['mean', 'sum', 'count'],
        'Z_score': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Add percentage efficient
    summary['pct_efficient'] = (summary['efficient_mean'] * 100).round(2)
    
    # Add interpretation
    summary['avg_behavior'] = runs_results.groupby('Asset')['pattern_type'].apply(
        lambda x: x.value_counts().index[0]
    )
    
    return summary

if __name__ == "__main__":
    # Load data from Yahoo Finance
    tickers = ["SPY", "AAPL", "GLD", "EURUSD=X"]
    start_date = '2010-01-01'
    end_date = '2022-02-26'

    import yfinance as yf
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    prices_df = data.copy()

    # Calculate log returns
    returns_df = np.log(data/data.shift(1)).dropna()

    # Run the vectorized runs test
    runs_results = vectorized_rolling_runs_test(returns_df, window=25)

    # Plot results for selected assets
    plot_runs_test_results(returns_df, prices_df, runs_results, 
                        assets=["SPY", "AAPL", "GLD", "EURUSD=X"])

    # Get summary statistics
    summary = summarize_runs_results(runs_results)
    print(summary)

    # Filter for periods of market inefficiency
    inefficient_periods = runs_results[runs_results['efficient'] == False]
    print(f"Found {len(inefficient_periods)} inefficient periods across all assets")
    
