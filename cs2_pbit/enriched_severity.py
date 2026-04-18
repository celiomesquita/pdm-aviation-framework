import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load raw data
try:
    df_raw = pd.read_csv("PBIT_data.csv", header=None)
except FileNotFoundError:
    print("Error: 'PBIT_data.csv' not found. Please make sure the file exists in the correct location.")
    exit()

# Define severity mapping
severity_mapping = {
    'N01': 0,  # Normal
    'N02': 0,  # Normal
    'F01': 1,  # Low - excessive damping
    'F02': 2,  # Medium - foreign body coupling
    'F03': 3,  # High - actuation assembly failure
    'F04': 2,  # Medium - actuation surface damage #1
    'F05': 2,  # Medium - actuation surface damage #2
    'F06': 3   # High - command transmission failure
}

# Extract metadata
types = df_raw.iloc[0].tolist()
cycles = pd.to_numeric(df_raw.iloc[1].tolist(), errors='coerce')

# Extract measurement matrix
values_matrix = df_raw.iloc[2:].transpose()
values_matrix = values_matrix.apply(pd.to_numeric, errors='coerce')

# Filter valid rows
valid_mask = pd.notna(cycles) & pd.notna(types)
types = pd.Series(types)[valid_mask].tolist()
cycles = pd.Series(cycles)[valid_mask].tolist()
values_matrix = values_matrix[valid_mask]

# Build long format with severity features
long_data = []
window_size = 15  # approximately 123/8

for i, (t, c) in enumerate(zip(types, cycles)):
    values = values_matrix.iloc[i].dropna().tolist()
    count_v = len(values)
    
    if count_v == 0:
        continue
        
    val_series = pd.Series(values)
    severity_score = severity_mapping.get(str(t), 0)
    
    # Precompute rolling stats
    roll_mean = val_series.rolling(window=window_size, min_periods=1).mean()
    roll_std = val_series.rolling(window=window_size, min_periods=1).std()
    roll_min = val_series.rolling(window=window_size, min_periods=1).min()
    roll_max = val_series.rolling(window=window_size, min_periods=1).max()
    
    # Calculate rolling slope for trend analysis
    roll_slope = val_series.rolling(window=window_size, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=True
    )
    
    mean_v = val_series.mean()
    std_v = val_series.std()
    min_v = val_series.min()
    max_v = val_series.max()
    median_v = val_series.median()
    last_v = val_series.iloc[-1] if count_v > 0 else np.nan
    delta_v = mean_v - last_v if count_v > 0 else np.nan

    # Calculate interval statistics (8 fixed intervals)
    interval_stats_list = [] 
    split_indices = np.linspace(0, count_v, 9, dtype=int)

    for k_interval in range(8):
        start_idx = split_indices[k_interval]
        end_idx = split_indices[k_interval+1]
        
        if start_idx < end_idx and start_idx < count_v:
            interval_slice = val_series[start_idx:end_idx]
            if not interval_slice.empty:
                interval_stats_list.append({
                    'mean': interval_slice.mean(),
                    'min': interval_slice.min(),
                    'max': interval_slice.max()
                })
            else:
                interval_stats_list.append({'mean': np.nan, 'min': np.nan, 'max': np.nan})
        else: 
            interval_stats_list.append({'mean': np.nan, 'min': np.nan, 'max': np.nan})
    
    # Severity-enhanced features calculation
    severity_tolerance_factor = 1.0 + (severity_score * 0.5)  # Higher severity = tighter tolerance
    cumulative_risk = 0
    
    for j, v in enumerate(values):
        current_interval_idx = -1
        for k_idx in range(len(split_indices) - 1):
            if j >= split_indices[k_idx] and j < split_indices[k_idx+1]:
                current_interval_idx = k_idx
                break
        if current_interval_idx == -1 and j == count_v - 1 and j == split_indices[-1]:
            current_interval_idx = 7
        
        row_interval_mean = np.nan
        row_interval_min = np.nan
        row_interval_max = np.nan

        if 0 <= current_interval_idx < len(interval_stats_list):
            row_interval_mean = interval_stats_list[current_interval_idx]['mean']
            row_interval_min = interval_stats_list[current_interval_idx]['min']
            row_interval_max = interval_stats_list[current_interval_idx]['max']
        
        # Standard features
        delta_prev = v - values[j - 1] if j > 0 else 0
        delta_from_roll_mean = v - (roll_mean.iloc[j] if count_v > 0 and not pd.isna(roll_mean.iloc[j]) else np.nan)
        
        # Severity-enhanced features
        severity_weighted_delta = delta_from_roll_mean * (1 + severity_score)
        severity_weighted_std = roll_std.iloc[j] * (1 + severity_score * 0.5) if count_v > 0 else np.nan
        severity_adjusted_anomaly = delta_from_roll_mean / severity_tolerance_factor if not pd.isna(delta_from_roll_mean) else np.nan
        
        # Risk-adjusted rolling features
        risk_weighted_trend = roll_slope.iloc[j] * (1 + severity_score) if count_v > 0 else np.nan
        abs_delta_severity = abs(delta_prev) * (1 + severity_score)
        cumulative_risk += abs_delta_severity
        
        # Severity progression indicators
        severity_momentum = severity_score  # This cycle's severity
        risk_intensity = abs_delta_severity / (roll_std.iloc[j] + 1e-8) if count_v > 0 and not pd.isna(roll_std.iloc[j]) else 0
        
        # Severity-relative thresholds
        severity_normalized_value = (v - roll_mean.iloc[j]) / (roll_std.iloc[j] + 1e-8) if count_v > 0 and not pd.isna(roll_mean.iloc[j]) and not pd.isna(roll_std.iloc[j]) else 0
        severity_threshold_breach = int(abs(severity_normalized_value) > (2.0 - severity_score * 0.3))  # Lower threshold for higher severity
            
        long_data.append({
            'type': t,
            'cycle': int(c),
            'value': v,
            'cycle_relative_pos': j / (count_v - 1) if count_v > 1 else 0,
            'is_last_in_cycle': int(j == (count_v - 1)),
            'value_mean': mean_v,
            'value_std': std_v,
            'value_min': min_v,
            'value_max': max_v,
            'value_median': median_v,
            'value_last': last_v,
            'delta_mean_last': delta_v,
            'value_count': count_v,
            'rolling_mean': roll_mean.iloc[j] if count_v > 0 else np.nan,
            'rolling_std': roll_std.iloc[j] if count_v > 0 else np.nan,
            'rolling_min': roll_min.iloc[j] if count_v > 0 else np.nan,
            'rolling_max': roll_max.iloc[j] if count_v > 0 else np.nan,
            'rolling_slope': roll_slope.iloc[j] if count_v > 0 else np.nan,
            'delta_prev': delta_prev,
            'delta_from_roll_mean': delta_from_roll_mean,
            'interval_mean': row_interval_mean,
            'interval_min': row_interval_min,
            'interval_max': row_interval_max,
            # New severity-enhanced features
            'severity_score': severity_score,
            'severity_weighted_delta': severity_weighted_delta,
            'severity_weighted_std': severity_weighted_std,
            'severity_adjusted_anomaly': severity_adjusted_anomaly,
            'risk_weighted_trend': risk_weighted_trend,
            'cumulative_risk_score': cumulative_risk,
            'severity_momentum': severity_momentum,
            'risk_intensity': risk_intensity,
            'severity_normalized_value': severity_normalized_value,
            'severity_threshold_breach': severity_threshold_breach
        })

df_enriched = pd.DataFrame(long_data)

# Force numeric columns explicitly
numeric_cols = ['value', 'cycle_relative_pos', 'rolling_mean', 'rolling_std',
                'rolling_min', 'rolling_max', 'rolling_slope', 'delta_prev', 'delta_from_roll_mean',
                'value_mean', 'value_std', 'value_min', 'value_max',
                'value_median', 'value_last', 'delta_mean_last', 'value_count',
                'interval_mean', 'interval_min', 'interval_max',
                'severity_score', 'severity_weighted_delta', 'severity_weighted_std',
                'severity_adjusted_anomaly', 'risk_weighted_trend', 'cumulative_risk_score',
                'severity_momentum', 'risk_intensity', 'severity_normalized_value',
                'severity_threshold_breach'] 

for col in numeric_cols:
    if col in df_enriched.columns:
        df_enriched[col] = pd.to_numeric(df_enriched[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in df_enriched for numeric conversion.")


# Add this section after creating df_enriched and before the train/test split

print("Original F06 samples:", len(df_enriched[df_enriched['type'] == 'F06']))

# Oversample F06 to balance with other classes
def oversample_f06(df, target_ratio=5):
    """
    Oversample F06 by duplicating existing cycles with small noise
    target_ratio: how many times to multiply F06 data
    """
    f06_data = df[df['type'] == 'F06'].copy()
    
    if f06_data.empty:
        print("No F06 data found for oversampling")
        return df
    
    # Get unique F06 cycles
    f06_cycles = f06_data['cycle'].unique()
    
    oversampled_data = []
    
    for multiplier in range(1, target_ratio):  # Skip 0 since original data already exists
        for cycle in f06_cycles:
            cycle_data = f06_data[f06_data['cycle'] == cycle].copy()
            
            # Create new cycle number (add offset to avoid conflicts)
            new_cycle = cycle + (multiplier * 10000)  # Large offset to avoid conflicts
            cycle_data['cycle'] = new_cycle
            
            # Add small noise to numerical features to create variation
            noise_columns = ['value', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
                           'delta_prev', 'delta_from_roll_mean', 'severity_weighted_delta',
                           'severity_weighted_std', 'risk_weighted_trend']
            
            for col in noise_columns:
                if col in cycle_data.columns:
                    # Add 1% random noise
                    noise = np.random.normal(0, 0.01 * cycle_data[col].std(), len(cycle_data))
                    cycle_data[col] = cycle_data[col] + noise
            
            oversampled_data.append(cycle_data)
    
    # Combine original data with oversampled F06 data
    if oversampled_data:
        oversampled_df = pd.concat([df] + oversampled_data, ignore_index=True)
        return oversampled_df
    
    return df

# Apply oversampling
df_enriched = oversample_f06(df_enriched, target_ratio=8)  # 8x more F06 data

print("After oversampling F06:", len(df_enriched[df_enriched['type'] == 'F06']))
print("F06 now represents", 
      round(100 * len(df_enriched[df_enriched['type'] == 'F06']) / len(df_enriched), 2), 
      "% of total data")

# Continue with your existing train/test split code...

# Split data into training and testing sets
train_dfs = []
test_dfs = []

if not df_enriched.empty:
    unique_types = df_enriched['type'].unique()

    for type_val in unique_types:
        type_df = df_enriched[df_enriched['type'] == type_val].copy()
        unique_cycles_for_type = type_df['cycle'].unique()
        num_unique_cycles = len(unique_cycles_for_type)

        if num_unique_cycles == 0:
            print(f"INFO: Type '{type_val}' has no data. Skipping.")
            continue
        train_cycles_for_type, test_cycles_for_type = train_test_split(
            unique_cycles_for_type,
            test_size=0.2,
            random_state=42
        )
        if len(train_cycles_for_type) > 0:
            train_dfs.append(type_df[type_df['cycle'].isin(train_cycles_for_type)])
        if len(test_cycles_for_type) > 0:
            test_dfs.append(type_df[type_df['cycle'].isin(test_cycles_for_type)])

df_train = pd.concat(train_dfs).reset_index(drop=True) if train_dfs else pd.DataFrame()
df_test = pd.concat(test_dfs).reset_index(drop=True) if test_dfs else pd.DataFrame()

# Save enriched datasets
if not df_train.empty:
    df_train.to_csv("train_enriched_severity_PBIT_data.csv", index=False)
    print(f"✅ Training dataset saved as 'train_enriched_severity_PBIT_data.csv' with {len(df_train)} rows.")
    print(f"📊 New severity features added: {len([col for col in df_train.columns if 'severity' in col or 'risk' in col])}")
else:
    print("⚠️ Training dataset is empty.")

if not df_test.empty:
    df_test.to_csv("test_enriched_severity_PBIT_data.csv", index=False)
    print(f"✅ Testing dataset saved as 'test_enriched_severity_PBIT_data.csv' with {len(df_test)} rows.")
else:
    print("⚠️ Testing dataset is empty.")

# Display feature summary
print(f"\n📈 Feature Summary:")
print(f"   Original features: {len([col for col in df_enriched.columns if not ('severity' in col or 'risk' in col)])}")
print(f"   Severity-enhanced features: {len([col for col in df_enriched.columns if 'severity' in col or 'risk' in col])}")
print(f"   Total features: {len(df_enriched.columns)}")

print("\nScript finished.")