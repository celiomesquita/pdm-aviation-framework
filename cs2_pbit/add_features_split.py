import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_prominences, peak_widths # For peak analysis
from sklearn.utils import resample # For oversampling

# Function to calculate features for each group (serial, type, cycle) - NOW EXPANDED
def calculate_cycle_features(group_df):
    """
    Calculates features for a single group (cycle).
    A group is defined by a unique (serial, type, cycle) combination.
    """
    values = group_df['value'].dropna() # Ensure we work with non-missing values
    n_values = len(values)
    window_size = 15 # For rolling calculations

    # --- Initialize all features to NaN ---
    # Basic
    val_min, val_median, val_std_overall = np.nan, np.nan, np.nan
    # From previous version (last window stats)
    group_last_value, rolling_mean_last15_val, rolling_std_last15_val, rolling_slope_last15_val = np.nan, np.nan, np.nan, np.nan
    delta_from_roll_mean_val = np.nan
    # New dynamic features
    diff1_mean, diff1_max, diff1_min, diff1_std = np.nan, np.nan, np.nan, np.nan
    diff2_mean, diff2_max, diff2_min, diff2_std = np.nan, np.nan, np.nan, np.nan
    num_peaks, mean_peak_prominence, mean_peak_width = 0, np.nan, np.nan # num_peaks can be 0
    zero_crossing_rate_val = 0 # Can be 0
    lag1_autocorr_val = np.nan
    std_of_rolling_mean15_series, range_of_rolling_mean15_series = np.nan, np.nan
    std_of_rolling_std15_series = np.nan

    # --- Basic descriptive statistics for the entire cycle ---
    if n_values > 0:
        val_min = values.min()
        val_median = values.median()
        group_last_value = values.iloc[-1]
    if n_values >= 2:
        val_std_overall = values.std()

    # --- Rolling features based on the last 'window_size' measurements (from previous script) ---
    if n_values >= window_size:
        rolling_mean_last15_val = values.rolling(window=window_size, min_periods=window_size).mean().iloc[-1]
        rolling_std_last15_val = values.rolling(window=window_size, min_periods=window_size).std().iloc[-1]
        y_slope = values.iloc[-window_size:]
        x_slope = np.arange(window_size)
        try:
            slope_result = linregress(x_slope, y_slope)
            rolling_slope_last15_val = slope_result.slope
        except ValueError:
            rolling_slope_last15_val = np.nan
            
    if not pd.isna(group_last_value) and not pd.isna(rolling_mean_last15_val):
        delta_from_roll_mean_val = group_last_value - rolling_mean_last15_val

    # --- New Dynamic Features ---
    # 1. Rate of Change (First Derivative)
    if n_values >= 2:
        diff1 = values.diff().dropna()
        if not diff1.empty:
            diff1_mean = diff1.mean()
            diff1_max = diff1.max()
            diff1_min = diff1.min()
            if len(diff1) >= 2:
                diff1_std = diff1.std()

    # 2. Second Derivative
    if n_values >= 3:
        diff2 = values.diff().diff().dropna() # or diff1.diff().dropna() if diff1 is guaranteed non-empty
        if not diff2.empty:
            diff2_mean = diff2.mean()
            diff2_max = diff2.max()
            diff2_min = diff2.min()
            if len(diff2) >= 2:
                diff2_std = diff2.std()

    # 3. Peak Characteristics
    if n_values >= 1: # find_peaks can run on short series
        # Adjust prominence threshold as needed, or make it data-dependent
        # Using a small fraction of the signal range as a heuristic for minimum prominence
        min_prominence = (values.max() - values.min()) * 0.05 if n_values > 1 else 0.01 
        peaks_indices, _ = find_peaks(values, prominence=min_prominence if min_prominence > 0 else None)
        num_peaks = len(peaks_indices)
        if num_peaks > 0:
            prominences = peak_prominences(values, peaks_indices)[0]
            # rel_height=0.5 means width is measured at half prominence
            widths_data = peak_widths(values, peaks_indices, rel_height=0.5) 
            mean_peak_prominence = np.mean(prominences) if len(prominences) > 0 else np.nan
            mean_peak_width = np.mean(widths_data[0]) if len(widths_data[0]) > 0 else np.nan


    # 4. Zero-Crossing Rate (around the mean)
    if n_values >= 2:
        mean_val = values.mean()
        # Check if mean_val is NaN (can happen if values was all NaN initially, though dropna should prevent)
        if not pd.isna(mean_val):
            centered_values = values - mean_val
            # Ensure no NaNs in centered_values before sign, though 'values' is already dropna'd
            zero_crossing_rate_val = np.sum(np.diff(np.sign(centered_values)) != 0)

    # 5. Autocorrelation (Lag-1)
    if n_values >= 2:
        # Pandas autocorr returns NaN if variance is 0, which is fine.
        lag1_autocorr_val = values.autocorr(lag=1)

    # 6. Variability of Rolling Statistics (calculated over the whole series)
    if n_values >= window_size: # Need enough data to form at least one full window
        rolling_mean_full_series = values.rolling(window=window_size, min_periods=window_size).mean().dropna()
        rolling_std_full_series = values.rolling(window=window_size, min_periods=window_size).std().dropna()

        if len(rolling_mean_full_series) >= 1:
            range_of_rolling_mean15_series = rolling_mean_full_series.max() - rolling_mean_full_series.min()
            if len(rolling_mean_full_series) >= 2:
                std_of_rolling_mean15_series = rolling_mean_full_series.std()
        
        if len(rolling_std_full_series) >= 2: # std of stds needs at least 2 std values
            std_of_rolling_std15_series = rolling_std_full_series.std()


    return pd.Series({
        # Existing features
        'value_min': val_min,
        'value_median': val_median,
        'value_std_overall': val_std_overall,
        'rolling_mean_last15': rolling_mean_last15_val,
        'rolling_std_last15': rolling_std_last15_val,
        'rolling_slope_last15': rolling_slope_last15_val,
        'delta_from_roll_mean': delta_from_roll_mean_val,
        # New dynamic features
        'diff1_mean': diff1_mean, 'diff1_max': diff1_max, 'diff1_min': diff1_min, 'diff1_std': diff1_std,
        'diff2_mean': diff2_mean, 'diff2_max': diff2_max, 'diff2_min': diff2_min, 'diff2_std': diff2_std,
        'num_peaks': num_peaks, 'mean_peak_prominence': mean_peak_prominence, 'mean_peak_width': mean_peak_width,
        'zero_crossing_rate': zero_crossing_rate_val,
        'lag1_autocorr': lag1_autocorr_val,
        'std_roll_mean15_series': std_of_rolling_mean15_series,
        'range_roll_mean15_series': range_of_rolling_mean15_series,
        'std_roll_std15_series': std_of_rolling_std15_series
    })

# --- Main script execution (largely the same structure as before) ---
if __name__ == "__main__":
    # --- Configuration ---
    INPUT_CSV_PATH = 'reshaped_fault_data_long_corrected_serial.csv' 
    OUTPUT_TRAIN_CSV_PATH = 'PBIT_train_features_dynamics_oversampled.csv' # New name
    OUTPUT_TEST_CSV_PATH = 'PBIT_test_features_dynamics.csv'         # New name
    
    SEVERITY_MAP = {
        'N01': 0, 'N02': 0, 'F01': 1, 'F02': 2, 'F03': 3,
        'F04': 2, 'F05': 3, 'F06': 3
    }
    TRAIN_SPLIT_RATIO = 0.75
    RANDOM_SEED = 42 
    OVERSAMPLE_TARGET_CLASS = 'F06'
    TARGET_COUNT_FOR_F06_IN_TRAIN = 45

    np.random.seed(RANDOM_SEED)

    # --- 1. Read Input Data ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully read '{INPUT_CSV_PATH}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Input file '{INPUT_CSV_PATH}' not found.")
        exit()
    except Exception as e:
        print(f"❌ Error reading '{INPUT_CSV_PATH}': {e}")
        exit()

    if df.empty:
        print(f"❌ Error: Input file '{INPUT_CSV_PATH}' is empty.")
        exit()
    required_cols = ['serial', 'type', 'cycle', 'value']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"❌ Error: Input CSV is missing required columns: {missing}")
        exit()

    # --- 2. Feature Engineering ---
    print("Calculating cycle-level features (now with more dynamics)...")
    # Wrap apply in a try-except block if complex calculations might fail unexpectedly
    try:
        aggregated_df = df.groupby(['serial', 'type', 'cycle'], as_index=False).apply(calculate_cycle_features)
    except Exception as e:
        print(f"❌ Error during feature calculation in apply(): {e}")
        # For debugging, you might want to inspect a specific group:
        # for (s, t, c), group in df.groupby(['serial', 'type', 'cycle']):
        #     try:
        #         calculate_cycle_features(group)
        #     except Exception as group_e:
        #         print(f"Error in group ({s}, {t}, {c}): {group_e}")
        #         # break # or continue
        exit()

    if aggregated_df.empty:
        print("ℹ️ Aggregated DataFrame (features) is empty. No data to process further.")
        exit()

    # --- 3. Add Severity Score and Weighted Features ---
    # (This part remains the same as the previous version of the script)
    if 'type' in aggregated_df.columns:
        aggregated_df['severity_score'] = aggregated_df['type'].map(SEVERITY_MAP)
    else:
        aggregated_df['severity_score'] = np.nan
        print("⚠️ Warning: 'type' column not found in aggregated_df. Cannot add 'severity_score'.")

    # Calculate severity-weighted features
    base_features_for_weighting = {
        'delta_from_roll_mean': 'severity_weighted_delta',
        'rolling_std_last15': 'severity_weighted_std',
        'rolling_slope_last15': 'risk_weighted_trend'
    }
    multipliers = { # (1 + severity_score) or (1 + severity_score * 0.5)
        'severity_weighted_delta': 1.0, # for (1 + severity_score * N) where N=1.0
        'severity_weighted_std': 0.5,   # for (1 + severity_score * N) where N=0.5
        'risk_weighted_trend': 1.0    # for (1 + severity_score * N) where N=1.0
    }

    for base_feature, weighted_feature in base_features_for_weighting.items():
        if base_feature in aggregated_df.columns and 'severity_score' in aggregated_df.columns:
            aggregated_df[weighted_feature] = aggregated_df[base_feature] * \
                                              (1 + aggregated_df['severity_score'] * multipliers[weighted_feature])
        else:
            aggregated_df[weighted_feature] = np.nan
            print(f"⚠️ Warning: Could not calculate '{weighted_feature}' due to missing base columns ('{base_feature}' or 'severity_score').")
    
    print("Added severity score and weighted features.")
    print(f"Shape of aggregated_df with all features: {aggregated_df.shape}")
    print(f"Columns in aggregated_df: {aggregated_df.columns.tolist()}")


    # --- 4. Train-Test Split based on Serial Numbers ---
    # (This part remains the same)
    if 'serial' not in aggregated_df.columns:
        print(f"❌ Error: 'serial' column not found in aggregated_df. Cannot perform train/test split.")
        exit()
    unique_serials = aggregated_df['serial'].unique()
    num_unique_serials = len(unique_serials)
    print(f"Found {num_unique_serials} unique serial numbers for splitting.")

    if num_unique_serials < 2:
        print(f"❌ Error: Fewer than 2 unique serials ({num_unique_serials}). Cannot perform a meaningful train/test split.")
        exit()
    shuffled_serials = np.copy(unique_serials)
    np.random.shuffle(shuffled_serials)
    split_idx = int(num_unique_serials * TRAIN_SPLIT_RATIO)
    if split_idx == num_unique_serials: split_idx -= 1
    elif split_idx == 0: split_idx += 1
    train_serials_list = shuffled_serials[:split_idx]
    test_serials_list = shuffled_serials[split_idx:]
    print(f"Splitting unique serials: {len(train_serials_list)} for training, {len(test_serials_list)} for testing.")
    train_df_final = aggregated_df[aggregated_df['serial'].isin(train_serials_list)].copy()
    test_df_final = aggregated_df[aggregated_df['serial'].isin(test_serials_list)].copy()
    print(f"Training set features shape (before oversampling): {train_df_final.shape}")
    print(f"Testing set features shape: {test_df_final.shape}")

    # --- 5. Oversample Target Class ('F06') in the Training Set ---
    # (This part remains the same)
    print(f"\n--- Oversampling for class '{OVERSAMPLE_TARGET_CLASS}' in Training Data ---")
    f06_train_samples = train_df_final[train_df_final['type'] == OVERSAMPLE_TARGET_CLASS]
    other_train_samples = train_df_final[train_df_final['type'] != OVERSAMPLE_TARGET_CLASS]
    num_f06_in_train_before = len(f06_train_samples)
    train_df_processed = train_df_final.copy() 
    if not f06_train_samples.empty:
        if num_f06_in_train_before < TARGET_COUNT_FOR_F06_IN_TRAIN:
            print(f"Found {num_f06_in_train_before} samples of '{OVERSAMPLE_TARGET_CLASS}'. Oversampling to target count: {TARGET_COUNT_FOR_F06_IN_TRAIN}.")
            f06_oversampled = resample(f06_train_samples, replace=True, 
                                       n_samples=TARGET_COUNT_FOR_F06_IN_TRAIN, random_state=RANDOM_SEED)
            train_df_processed = pd.concat([other_train_samples, f06_oversampled], ignore_index=True)
            train_df_processed = train_df_processed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            num_f06_in_train_after = len(train_df_processed[train_df_processed['type'] == OVERSAMPLE_TARGET_CLASS])
            print(f"'{OVERSAMPLE_TARGET_CLASS}' count in processed training set: {num_f06_in_train_after}")
        else:
            print(f"'{OVERSAMPLE_TARGET_CLASS}' in training set ({num_f06_in_train_before}) already meets or exceeds target ({TARGET_COUNT_FOR_F06_IN_TRAIN}). No oversampling performed.")
    else:
        print(f"No samples of type '{OVERSAMPLE_TARGET_CLASS}' found in the initial training set. Oversampling not possible.")
    print(f"Final processed training set shape: {train_df_processed.shape}")

    # --- 6. Save Processed Datasets ---
    # (This part remains the same, but filenames are updated)
    try:
        train_df_processed.to_csv(OUTPUT_TRAIN_CSV_PATH, index=False)
        print(f"\n✅ Processed training dataset (with oversampling) saved to '{OUTPUT_TRAIN_CSV_PATH}'")
    except Exception as e:
        print(f"\n❌ Error saving processed training dataset: {e}")
    try:
        test_df_final.to_csv(OUTPUT_TEST_CSV_PATH, index=False)
        print(f"✅ Testing dataset saved to '{OUTPUT_TEST_CSV_PATH}'")
    except Exception as e:
        print(f"\n❌ Error saving testing dataset: {e}")

    print("\n--- Script Finished ---")
    print(f"\nNote on SMOTE: For more advanced oversampling creating synthetic samples (rather than duplicating),")
    print(f"consider SMOTE from the 'imblearn' library (install with 'pip install imbalanced-learn').")