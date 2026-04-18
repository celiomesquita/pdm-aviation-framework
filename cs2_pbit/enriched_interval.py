import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load raw data
# Ensure "PBIT_data.csv" is in the same directory as the script,
# or provide the full path to the file.
try:
    df_raw = pd.read_csv("PBIT_data.csv", header=None)
except FileNotFoundError:
    print("Error: 'PBIT_data.csv' not found. Please make sure the file exists in the correct location.")
    exit()


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

# Build long format
long_data = []
window_size = 15  # you can adjust

for i, (t, c) in enumerate(zip(types, cycles)):
    values = values_matrix.iloc[i].dropna().tolist()
    count_v = len(values)
    
    if count_v == 0: # Skip if no values for this type/cycle
        continue
        
    val_series = pd.Series(values)
    
    # Precompute rolling stats
    roll_mean = val_series.rolling(window=window_size, min_periods=1).mean()
    roll_std = val_series.rolling(window=window_size, min_periods=1).std()
    roll_min = val_series.rolling(window=window_size, min_periods=1).min()
    roll_max = val_series.rolling(window=window_size, min_periods=1).max()
    
    mean_v = val_series.mean()
    std_v = val_series.std()
    min_v = val_series.min()
    max_v = val_series.max()
    median_v = val_series.median()
    last_v = val_series.iloc[-1] if count_v > 0 else np.nan
    delta_v = mean_v - last_v if count_v > 0 else np.nan

    # Calculate interval statistics (8 fixed intervals)
    interval_stats_list = [] 
    split_indices = np.linspace(0, count_v, 9, dtype=int) # 9 points define 8 intervals

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
            
    for j, v in enumerate(values):
        current_interval_idx = -1  # Default for safety, or handle last point explicitly
        for k_idx in range(len(split_indices) - 1): # Iterate through 0 to 7 for 8 intervals
            if j >= split_indices[k_idx] and j < split_indices[k_idx+1]:
                current_interval_idx = k_idx
                break
        # Handle the very last point if it falls exactly on the last boundary
        if current_interval_idx == -1 and j == count_v - 1 and j == split_indices[-1]:
            current_interval_idx = 7 # Assign to the last interval
        
        row_interval_mean = np.nan
        row_interval_min = np.nan
        row_interval_max = np.nan

        if 0 <= current_interval_idx < len(interval_stats_list):
            row_interval_mean = interval_stats_list[current_interval_idx]['mean']
            row_interval_min = interval_stats_list[current_interval_idx]['min']
            row_interval_max = interval_stats_list[current_interval_idx]['max']
            
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
            'delta_prev': v - values[j - 1] if j > 0 else 0,
            'delta_from_roll_mean': v - (roll_mean.iloc[j] if count_v > 0 and not pd.isna(roll_mean.iloc[j]) else np.nan),
            'interval_mean': row_interval_mean,
            'interval_min': row_interval_min,
            'interval_max': row_interval_max
        })

df_enriched = pd.DataFrame(long_data)

# Force numeric columns explicitly
numeric_cols = ['value', 'cycle_relative_pos', 'rolling_mean', 'rolling_std',
                'rolling_min', 'rolling_max', 'delta_prev', 'delta_from_roll_mean',
                'value_mean', 'value_std', 'value_min', 'value_max',
                'value_median', 'value_last', 'delta_mean_last', 'value_count',
                'interval_mean', 'interval_min', 'interval_max'] 

for col in numeric_cols:
    if col in df_enriched.columns:
        df_enriched[col] = pd.to_numeric(df_enriched[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in df_enriched for numeric conversion.")


# --- Split data into training and testing sets ---
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
    df_train.to_csv("train_enriched_long_PBIT_data.csv", index=False)
    print(f"✅ Training dataset saved as 'train_enriched_long_PBIT_data.csv' with {len(df_train)} rows.")
else:
    print("⚠️ Training dataset is empty.")

if not df_test.empty:
    df_test.to_csv("test_enriched_long_PBIT_data.csv", index=False)
    print(f"✅ Testing dataset saved as 'test_enriched_long_PBIT_data.csv' with {len(df_test)} rows.")
else:
    print("⚠️ Testing dataset is empty. This might be expected if all types had few cycles.")

print("\nScript finished.")