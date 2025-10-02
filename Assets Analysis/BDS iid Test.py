
#! Using BDS statistics to detect nonlinearity in time series: https://2001.isiproceedings.org/pdf/98.PDF
#* BDS (Brock–Dechert–Scheinkman) test is used to test for independence and identically distributed (i.i.d.) data
# It can identify:
# Deterministic chaos (seemingly random but actually following complex rules)
# Nonlinear dependencies (complex relationships between past and future values)
# BDS statistic >> 1.96 → reject i.i.d. → potential chaotic/deterministic structure
# BDS statistic ~ 0 or < 1.96 → cannot reject i.i.d. → consistent with randomness
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds

def run_bds(data, max_dim=5, epsilon=None, distance=1.5):
    """
    Run BDS test using statsmodels implementation.
    Trade-off: Higher m = more power to detect complex patterns, but fewer observations
    
    Parameters:
    -----------
    data : array-like
        Time series data
    max_dim : int
        Maximum embedding dimension to test
    epsilon : float, optional
        Threshold distance. If None, computed automatically
    distance : float
        Distance multiplier when epsilon is None
        
    Returns:
    --------
    pd.Dataframe
        BDS statistics and p-values with 1 and 5% significance for each dimension
    """
    try:
        # statsmodels.tsa.stattools.bds returns (bds_stat, p_values)
        bds_stat, p_values = bds(data, max_dim=max_dim, epsilon=epsilon, distance=distance)
        if bds_stat is None:
            print("BDS test failed")
            return None
    
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Dimension': range(2, max_dim + 1),
            'BDS_Statistic': bds_stat,
            'P_Value': p_values,
            'Significant_5%': p_values < 0.05,
            'Significant_1%': p_values < 0.01
        })
        
        return results_df
    except Exception as e:
        print(f"Error in statsmodels BDS test: {e}")
        return None 

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test 1: White noise (should NOT reject independence)
    print("\n1. Testing White Noise (should NOT reject H0 of independence):")
    print("-" * 50)
    white_noise = np.random.normal(0,1,500)
    
    results_wn = run_bds(white_noise, max_dim=4)
    if results_wn is not None:
        print(results_wn.to_string(index=False))
        print(f"\nInterpretation: If most p-values > 0.05, we fail to reject independence (good for white noise)")
    
    # Test 2: AR(1) process (should reject independence)
    print("\n\n2. Testing AR(1) Process (should reject H0 of independence):")
    print("-" * 55)
    
    # Generate AR(1): x_t = 0.7 * x_{t-1} + epsilon_t
    n = 500
    ar1_data = np.zeros(n)
    ar1_data[0] = np.random.randn()
    for i in range(1, n):
        ar1_data[i] = 0.7 * ar1_data[i-1] + np.random.randn()
    
    results_ar1 = run_bds(ar1_data, max_dim=4)
    if results_ar1 is not None:
        print(results_ar1.to_string(index=False))
        print(f"\nInterpretation: If most p-values < 0.05, we reject independence (expected for AR process)")
    
    # Test 3: Nonlinear process (should strongly reject independence)
    print("\n\n3. Testing Nonlinear Process (should strongly reject H0):")
    print("-" * 50)
    
    # Generate nonlinear series: x_t = 0.6 * x_{t-1} + 0.3 * x_{t-1}^2 + epsilon_t
    nonlinear_data = np.zeros(500)
    nonlinear_data[0] = np.random.randn()
    for i in range(1, 500):
        nonlinear_data[i] = (0.6 * nonlinear_data[i-1] + 
                            0.3 * nonlinear_data[i-1]**2 + 
                            0.5 * np.random.randn())
    
    results_nl = run_bds(nonlinear_data, max_dim=4)
    if results_nl is not None:
        print(results_nl.to_string(index=False))
        print(f"\nInterpretation: Very small p-values indicate strong evidence against independence")
    
    # Test 4: BDS on Chaotic Data
    print("\n\n4. Testing BDS on Chaotic Data:")
    print("-" * 40)
    # Logistic Map
    def logistic_map(r, x0, n):
        x = [x0]
        for _ in range(n - 1):
            x.append(r * x[-1] * (1 - x[-1]))
        return x

    chaos_data = logistic_map(r=3.8, x0=0.9, n=1000)
    
    results_chaos = run_bds(chaos_data, max_dim=4)
    if results_chaos is not None:
        print(results_chaos.to_string(index=False))
        print(f"\nInterpretation: Very small p-values indicate strong evidence against independence")