import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_prominences, peak_widths

# --- Main script execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Path to the main data file
    INPUT_CSV_PATH = 'reshaped_fault_data_long_corrected_serial.csv' 
    # Path to the file containing pre-defined train/test splits
    SERIALS_INFO_PATH = 'serial_numbers.csv' 
    
    OUTPUT_TRAIN_CSV_PATH = 'PBIT_train_features_updated.csv'
    OUTPUT_TEST_CSV_PATH = 'PBIT_test_features_updated.csv'
    
    # MODIFIED: Severity for F05 changed from 3 to 2 to match the paper
    SEVERITY_MAP = {
        'N01': 0, 'N02': 0, 'F01': 1, 'F02': 2, 'F03': 3,
        'F04': 2, 'F05': 2, 'F06': 3
    }

    # Add F05 to your oversampling targets
    OVERSAMPLE_TARGETS = {
        'F05': 60,  # Boost F05 above F06 since it's more complex to distinguish
        'F06': 45   # Keep your current F06 target
    }

    RANDOM_SEED = 42 
    # OVERSAMPLE_TARGET_CLASS = 'F06'
    # TARGET_COUNT_FOR_F06_IN_TRAIN = 45
    NOISE_LEVEL = 0.01 # 1% noise for oversampling, as per the paper

    np.random.seed(RANDOM_SEED)

# --- Feature Calculation Function (UPDATED) ---
def calculate_cycle_features(group_df):
    """
    Calculates features for a single group (cycle), now aligned with the paper's description.
    """
    values = group_df['value'].dropna()
    n_values = len(values)
    window_size = 15

    # --- Initialize all features to NaN ---
    val_min, val_median, val_std_overall = np.nan, np.nan, np.nan
    rolling_mean_last15_val, rolling_std_last15_val, rolling_slope_last15_val = np.nan, np.nan, np.nan
    # NEW: Added rolling min and max to match the paper
    rolling_min_last15_val, rolling_max_last15_val = np.nan, np.nan
    delta_from_roll_mean_val = np.nan
    diff1_mean, diff1_max, diff1_min, diff1_std = np.nan, np.nan, np.nan, np.nan
    diff2_mean, diff2_max, diff2_min, diff2_std = np.nan, np.nan, np.nan, np.nan
    num_peaks, mean_peak_prominence, mean_peak_width = 0, np.nan, np.nan
    zero_crossing_rate_val = 0
    lag1_autocorr_val = np.nan
    std_of_rolling_mean15_series, range_of_rolling_mean15_series = np.nan, np.nan
    std_of_rolling_std15_series = np.nan
    # NEW: Added a proxy for cumulative risk scoring
    cumulative_risk_score = np.nan

    if n_values > 0:
        val_min = values.min()
        val_median = values.median()
        group_last_value = values.iloc[-1]
    if n_values >= 2:
        val_std_overall = values.std()
        # NEW: Cumulative risk proxy - final value of the cumulative sum of absolute changes
        cumulative_risk_score = values.diff().abs().cumsum().iloc[-1]

    if n_values >= window_size:
        last_window = values.iloc[-window_size:]
        rolling_mean_last15_val = last_window.mean()
        rolling_std_last15_val = last_window.std()
        # NEW: Rolling min and max calculation
        rolling_min_last15_val = last_window.min()
        rolling_max_last15_val = last_window.max()
        
        y_slope = last_window
        x_slope = np.arange(window_size)
        try:
            rolling_slope_last15_val = linregress(x_slope, y_slope).slope
        except ValueError:
            rolling_slope_last15_val = np.nan
            
    if not pd.isna(group_last_value) and not pd.isna(rolling_mean_last15_val):
        delta_from_roll_mean_val = group_last_value - rolling_mean_last15_val

    if n_values >= 2:
        diff1 = values.diff().dropna()
        if not diff1.empty:
            diff1_mean, diff1_max, diff1_min, diff1_std = diff1.mean(), diff1.max(), diff1.min(), diff1.std() if len(diff1) >= 2 else np.nan
    if n_values >= 3:
        diff2 = values.diff().diff().dropna()
        if not diff2.empty:
            diff2_mean, diff2_max, diff2_min, diff2_std = diff2.mean(), diff2.max(), diff2.min(), diff2.std() if len(diff2) >= 2 else np.nan

    if n_values >= 1:
        min_prominence = (values.max() - values.min()) * 0.05 if n_values > 1 else 0.01 
        peaks_indices, _ = find_peaks(values, prominence=min_prominence if min_prominence > 0 else None)
        num_peaks = len(peaks_indices)
        if num_peaks > 0:
            prominences = peak_prominences(values, peaks_indices)[0]
            widths_data = peak_widths(values, peaks_indices, rel_height=0.5) 
            mean_peak_prominence = np.mean(prominences) if len(prominences) > 0 else np.nan
            mean_peak_width = np.mean(widths_data[0]) if len(widths_data[0]) > 0 else np.nan

    if n_values >= 2:
        mean_val = values.mean()
        if not pd.isna(mean_val):
            zero_crossing_rate_val = np.sum(np.diff(np.sign(values - mean_val)) != 0)

    if n_values >= 2:
        lag1_autocorr_val = values.autocorr(lag=1)

    if n_values >= window_size:
        rolling_mean_full_series = values.rolling(window=window_size).mean().dropna()
        if len(rolling_mean_full_series) >= 2:
            std_of_rolling_mean15_series = rolling_mean_full_series.std()
            range_of_rolling_mean15_series = rolling_mean_full_series.max() - rolling_mean_full_series.min()
        rolling_std_full_series = values.rolling(window=window_size).std().dropna()
        if len(rolling_std_full_series) >= 2:
            std_of_rolling_std15_series = rolling_std_full_series.std()

    return pd.Series({
        'value_min': val_min, 'value_median': val_median, 'value_std_overall': val_std_overall,
        'rolling_mean_last15': rolling_mean_last15_val, 'rolling_std_last15': rolling_std_last15_val,
        'rolling_slope_last15': rolling_slope_last15_val,
        'rolling_min_last15': rolling_min_last15_val, # NEW
        'rolling_max_last15': rolling_max_last15_val, # NEW
        'delta_from_roll_mean': delta_from_roll_mean_val,
        'diff1_mean': diff1_mean, 'diff1_max': diff1_max, 'diff1_min': diff1_min, 'diff1_std': diff1_std,
        'diff2_mean': diff2_mean, 'diff2_max': diff2_max, 'diff2_min': diff2_min, 'diff2_std': diff2_std,
        'num_peaks': num_peaks, 'mean_peak_prominence': mean_peak_prominence, 'mean_peak_width': mean_peak_width,
        'zero_crossing_rate': zero_crossing_rate_val, 'lag1_autocorr': lag1_autocorr_val,
        'std_roll_mean15_series': std_of_rolling_mean15_series, 'range_roll_mean15_series': range_of_rolling_mean15_series,
        'std_roll_std15_series': std_of_rolling_std15_series,
        'cumulative_risk_score': cumulative_risk_score # NEW
    })

# --- Oversampling Function (UPDATED) ---
def oversample_with_noise(df_majority, df_minority, target_count, noise_level, random_state):
    """
    Oversamples a minority class by duplicating samples and adding Gaussian noise.
    FIXED: Uses abs() to ensure the noise scale is always non-negative.
    """
    rng = np.random.RandomState(random_state)
    num_to_generate = target_count - len(df_minority)
    if num_to_generate <= 0:
        return pd.concat([df_majority, df_minority], ignore_index=True)

    # Identify numeric feature columns to add noise to
    numeric_cols = df_minority.select_dtypes(include=np.number).columns.tolist()
    # Exclude identifiers and labels from noise injection
    non_feature_cols = ['cycle', 'severity_score'] 
    feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
    
    synthetic_samples = []
    for _ in range(num_to_generate):
        # Select a random sample to duplicate
        original_sample = df_minority.sample(n=1, random_state=rng)
        new_sample = original_sample.copy()
        
        # Add noise to the feature columns
        for col in feature_cols:
            feature_value = new_sample[col].iloc[0]
            if pd.notna(feature_value):
                # FIX IS HERE: Use abs(feature_value) to calculate scale
                scale = noise_level * abs(feature_value)
                # Ensure scale is not zero to avoid issues with static features
                if scale > 0:
                    noise = rng.normal(0, scale)
                    new_sample[col] += noise
        
        synthetic_samples.append(new_sample)

    synthetic_df = pd.concat(synthetic_samples, ignore_index=True)
    return pd.concat([df_majority, df_minority, synthetic_df], ignore_index=True)

# --- Main Execution Block ---

# 1. Read Input Data
print("--- 1. Reading Input Data ---")
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Successfully read '{INPUT_CSV_PATH}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Input file '{INPUT_CSV_PATH}' not found.")
    exit()

# 2. Feature Engineering
print("\n--- 2. Performing Feature Engineering ---")
aggregated_df = df.groupby(['serial', 'type', 'cycle'], as_index=False).apply(calculate_cycle_features)
print(f"Shape of aggregated_df with all features: {aggregated_df.shape}")

# 3. Add Severity Score and Weighted Features
print("\n--- 3. Adding Severity and Weighted Features ---")
aggregated_df['severity_score'] = aggregated_df['type'].map(SEVERITY_MAP)
base_features_for_weighting = {
    'delta_from_roll_mean': 'severity_weighted_delta',
    'rolling_std_last15': 'severity_weighted_std',
    'rolling_slope_last15': 'risk_weighted_trend'
}
multipliers = {'severity_weighted_delta': 1.0, 'severity_weighted_std': 0.5, 'risk_weighted_trend': 1.0}
for base_feature, weighted_feature in base_features_for_weighting.items():
    aggregated_df[weighted_feature] = aggregated_df[base_feature] * (1 + aggregated_df['severity_score'] * multipliers[weighted_feature])
print("Added severity score and weighted features.")

# 4. Train-Test Split using Pre-defined File
print(f"\n--- 4. Splitting Data using '{SERIALS_INFO_PATH}' ---")
try:
    serials_info_df = pd.read_csv(SERIALS_INFO_PATH)
except FileNotFoundError:
    print(f"❌ Error: Serials info file '{SERIALS_INFO_PATH}' not found.")
    exit()

train_serials_list = serials_info_df[serials_info_df['category'] == 'T']['sn'].tolist()
test_serials_list = serials_info_df[serials_info_df['category'] == 'V']['sn'].tolist()
print(f"Found {len(train_serials_list)} serials for training and {len(test_serials_list)} for testing.")

train_df_final = aggregated_df[aggregated_df['serial'].isin(train_serials_list)].copy()
test_df_final = aggregated_df[aggregated_df['serial'].isin(test_serials_list)].copy()
print(f"Training set features shape (before oversampling): {train_df_final.shape}")
print(f"Testing set features shape: {test_df_final.shape}")

# 5. Oversample with Noise Injection
print(f"\n--- 5. Oversampling '{OVERSAMPLE_TARGET_CLASS}' with Noise Injection ---")
f06_train_samples = train_df_final[train_df_final['type'] == OVERSAMPLE_TARGET_CLASS]
other_train_samples = train_df_final[train_df_final['type'] != OVERSAMPLE_TARGET_CLASS]

if not f06_train_samples.empty and len(f06_train_samples) < TARGET_COUNT_FOR_F06_IN_TRAIN:
    print(f"Found {len(f06_train_samples)} samples of '{OVERSAMPLE_TARGET_CLASS}'. Oversampling to {TARGET_COUNT_FOR_F06_IN_TRAIN} with {NOISE_LEVEL*100}% noise.")
    train_df_processed = oversample_with_noise(other_train_samples, f06_train_samples, 
                                                TARGET_COUNT_FOR_F06_IN_TRAIN, NOISE_LEVEL, RANDOM_SEED)
else:
    train_df_processed = train_df_final.copy()
    if f06_train_samples.empty:
        print(f"No samples of '{OVERSAMPLE_TARGET_CLASS}' found to oversample.")
    else:
        print("Sufficient samples of '{OVERSAMPLE_TARGET_CLASS}' exist, no oversampling needed.")

# Shuffle the final training dataframe
train_df_processed = train_df_processed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Final processed training set shape: {train_df_processed.shape}")

# --- ADD THIS LINE FOR DEBUGGING ---
print("\n--- DEBUG: Value counts in the final test set ---")
print(test_df_final['type'].value_counts())
# --- End of debug line ---

# 6. Save Processed Datasets
print("\n--- 6. Saving Processed Datasets ---")
try:
    train_df_processed.to_csv(OUTPUT_TRAIN_CSV_PATH, index=False)
    print(f"✅ Processed training dataset saved to '{OUTPUT_TRAIN_CSV_PATH}'")
    test_df_final.to_csv(OUTPUT_TEST_CSV_PATH, index=False)
    print(f"✅ Testing dataset saved to '{OUTPUT_TEST_CSV_PATH}'")
except Exception as e:
    print(f"❌ Error saving datasets: {e}")

print("\n--- Script Finished ---")