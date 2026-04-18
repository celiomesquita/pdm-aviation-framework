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
    
    # Severity is a post-classification lookup, not a model predictor.
    SEVERITY_MAP = {
        'N01': 0, 'N02': 0, 'F01': 1, 'F02': 2, 'F03': 3,
        'F04': 2, 'F05': 2, 'F06': 3
    }

    # Multi-class oversampling targets
    OVERSAMPLE_TARGETS = {
        'F04': 60,   # NEW: Boost F04 to reduce F04/F05 confusion
        'F05': 60,   # Keep successful F05 boost
        'F06': 45,   # Keep perfect F06 performance  
        # 'N02': 120   # Keep excellent N02 recovery
    }

    RANDOM_SEED = 42 
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
    # Signal-only cumulative variation feature. It does not use class labels.
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
        'cumulative_risk_score': cumulative_risk_score
    })

# --- Multi-Class Oversampling Function (NEW) ---
def oversample_multiple_classes(df, target_counts, noise_level, random_state):
    """
    Oversamples multiple minority classes by duplicating samples and adding Gaussian noise.
    
    Args:
        df: Original training dataframe
        target_counts: Dictionary mapping class names to target sample counts
        noise_level: Gaussian noise level (as fraction of feature value)
        random_state: Random seed for reproducibility
    
    Returns:
        Balanced dataframe with oversampled classes
    """
    rng = np.random.RandomState(random_state)
    
    # Separate classes to oversample from others
    classes_to_oversample = set(target_counts.keys())
    oversampled_dfs = []
    
    # Handle non-oversampled classes
    non_oversampled = df[~df['type'].isin(classes_to_oversample)]
    if not non_oversampled.empty:
        oversampled_dfs.append(non_oversampled)
    
    # Process each class for oversampling
    for class_name, target_count in target_counts.items():
        class_samples = df[df['type'] == class_name]
        current_count = len(class_samples)
        
        print(f"  Processing {class_name}: Current={current_count}, Target={target_count}")
        
        if current_count == 0:
            print(f"    Warning: No samples found for class {class_name}")
            continue
            
        if current_count >= target_count:
            print(f"    No oversampling needed for {class_name}")
            oversampled_dfs.append(class_samples)
            continue
        
        # Generate synthetic samples
        num_to_generate = target_count - current_count
        print(f"    Generating {num_to_generate} synthetic samples for {class_name}")
        
        # Identify feature columns for noise injection
        numeric_cols = class_samples.select_dtypes(include=np.number).columns.tolist()
        non_feature_cols = ['cycle', 'severity_score', 'serial']  # Added 'serial' to exclusions
        feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
        
        synthetic_samples = []
        for i in range(num_to_generate):
            # Select a random sample to duplicate
            original_sample = class_samples.sample(n=1, random_state=rng)
            new_sample = original_sample.copy()
            
            # Add noise to feature columns
            for col in feature_cols:
                feature_value = new_sample[col].iloc[0]
                if pd.notna(feature_value):
                    scale = noise_level * abs(feature_value)
                    if scale > 0:
                        noise = rng.normal(0, scale)
                        new_sample[col] += noise
            
            synthetic_samples.append(new_sample)
        
        # Combine original and synthetic samples for this class
        synthetic_df = pd.concat(synthetic_samples, ignore_index=True)
        class_combined = pd.concat([class_samples, synthetic_df], ignore_index=True)
        oversampled_dfs.append(class_combined)
    
    # Combine all classes
    final_df = pd.concat(oversampled_dfs, ignore_index=True)
    return final_df

# --- Single-Class Oversampling Function (KEPT FOR BACKWARD COMPATIBILITY) ---
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

# 3. Create Severity Lookup for Post-Classification Use
print("\n--- 3. Creating Severity Lookup (excluded from model predictors) ---")
aggregated_df['severity_score'] = aggregated_df['type'].map(SEVERITY_MAP)
severity_lookup_df = (
    aggregated_df[['type', 'severity_score']]
    .drop_duplicates()
    .sort_values('type')
    .reset_index(drop=True)
)
print("Created severity lookup. The severity_score column will be dropped before saving model datasets.")

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

# Display current class distribution in training set
print("\n--- Current Training Set Distribution ---")
train_class_counts = train_df_final['type'].value_counts().sort_index()
print(train_class_counts)

# 5. Multi-Class Oversampling with Noise Injection
print(f"\n--- 5. Multi-Class Oversampling with Noise Injection ---")
print(f"Target counts: {OVERSAMPLE_TARGETS}")
print(f"Noise level: {NOISE_LEVEL*100}%")

train_df_processed = oversample_multiple_classes(
    train_df_final, 
    OVERSAMPLE_TARGETS, 
    NOISE_LEVEL, 
    RANDOM_SEED
)

# Shuffle the final training dataframe
train_df_processed = train_df_processed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"\nFinal processed training set shape: {train_df_processed.shape}")

# Display final class distribution
print("\n--- Final Training Set Distribution (After Oversampling) ---")
final_class_counts = train_df_processed['type'].value_counts().sort_index()
print(final_class_counts)

# Display changes
print("\n--- Oversampling Summary ---")
for class_name in train_class_counts.index:
    original = train_class_counts.get(class_name, 0)
    final = final_class_counts.get(class_name, 0)
    change = final - original
    if change > 0:
        print(f"{class_name}: {original} → {final} (+{change} synthetic samples)")
    else:
        print(f"{class_name}: {original} → {final} (no change)")

# --- DEBUG: Value counts in the final test set ---
print("\n--- DEBUG: Test Set Distribution ---")
print(test_df_final['type'].value_counts().sort_index())

# 6. Save Processed Datasets
print("\n--- 6. Saving Processed Datasets ---")
try:
    leakage_columns = ['severity_score']
    train_export = train_df_processed.drop(columns=leakage_columns, errors='ignore')
    test_export = test_df_final.drop(columns=leakage_columns, errors='ignore')

    train_export.to_csv(OUTPUT_TRAIN_CSV_PATH, index=False)
    print(f"✅ Processed training dataset saved to '{OUTPUT_TRAIN_CSV_PATH}'")
    test_export.to_csv(OUTPUT_TEST_CSV_PATH, index=False)
    print(f"✅ Testing dataset saved to '{OUTPUT_TEST_CSV_PATH}'")
    severity_lookup_df.to_csv('failure_severity_lookup.csv', index=False)
    print("✅ Severity lookup saved to 'failure_severity_lookup.csv' for post-classification prioritization")
except Exception as e:
    print(f"❌ Error saving datasets: {e}")

print("\n--- Script Finished ---")
