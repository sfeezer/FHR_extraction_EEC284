import numpy as np
import pandas as pd

def weighted_median(values, weights):
    """
    Computes the weighted median of a set of values.
    """
    if len(values) == 0:
        return np.nan
    
    # Sort values and weights by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Cumulative weight
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)
    
    # Find the index where cumulative weight reaches half of total weight
    median_idx = np.where(cumulative_weights >= total_weight / 2.0)[0][0]
    
    return sorted_values[median_idx]

def fuse_fhr_tracks(fhr_tracks_df, mad_thresh=3.0):
    """
    Fuses multiple FHR tracks into a single estimate using weighted MAD outlier rejection.
    
    Weights (from 2021 Paper):
    ch2: 1
    ch3: 3
    ch4: 2
    ch5: 2
    """
    print(f"Stage 5: Sensor Fusion (MAD Threshold: {mad_thresh})")
    
    # Define weights per detector
    detector_weights = {
        'ch2': 1,
        'ch3': 3,
        'ch4': 2,
        'ch5': 2
    }
    
    fused_results = []
    
    for i, row in fhr_tracks_df.iterrows():
        values = []
        weights = []
        
        for col in fhr_tracks_df.columns:
            # col name is like 'ch2_WL1'
            det = col.split('_')[0]
            val = row[col]
            
            if not np.isnan(val) and det in detector_weights:
                values.append(val)
                weights.append(detector_weights[det])
        
        if not values:
            fused_results.append(np.nan)
            continue
            
        values = np.array(values)
        weights = np.array(weights)
        
        # 1. Calculate Weighted Median
        w_median = weighted_median(values, weights)
        
        # 2. Calculate Median Absolute Deviation (MAD)
        # Using the weighted median for MAD calculation as well
        ad = np.abs(values - w_median)
        mad = weighted_median(ad, weights)
        
        # Avoid division by zero if MAD is 0
        if mad == 0:
            mad = 1e-6
            
        # 3. Outlier Rejection
        mask = ad <= (mad_thresh * mad)
        
        valid_values = values[mask]
        valid_weights = weights[mask]
        
        if len(valid_values) == 0:
            # If all are outliers, fall back to weighted median
            fused_results.append(w_median)
        else:
            # 4. Weighted Mean of valid estimates
            w_mean = np.sum(valid_values * valid_weights) / np.sum(valid_weights)
            fused_results.append(w_mean)
            
    return pd.Series(fused_results, name="fused_fhr")

def calculate_metrics(estimated_fhr, reference_fhr):
    """
    Calculates MAE and RMSE between estimated and reference FHR.
    """
    # Remove NaN values from both
    mask = ~np.isnan(estimated_fhr) & ~np.isnan(reference_fhr)
    y_est = estimated_fhr[mask]
    y_ref = reference_fhr[mask]
    
    if len(y_est) == 0:
        return np.nan, np.nan
        
    mae = np.mean(np.abs(y_est - y_ref))
    rmse = np.sqrt(np.mean((y_est - y_ref)**2))
    
    return mae, rmse
