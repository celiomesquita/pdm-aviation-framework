import pandas as pd

# Load raw data
df_raw = pd.read_csv("PBIT_data.csv", header=None)

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
    
    # Convert to Series
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
    last_v = val_series.iloc[-1]
    delta_v = mean_v - last_v

    for j, v in enumerate(values):
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
            'rolling_mean': roll_mean.iloc[j],
            'rolling_std': roll_std.iloc[j],
            'rolling_min': roll_min.iloc[j],
            'rolling_max': roll_max.iloc[j],
            'delta_prev': v - values[j - 1] if j > 0 else 0,
            'delta_from_roll_mean': v - roll_mean.iloc[j]
        })


# Save enriched dataset
df_enriched = pd.DataFrame(long_data)

# Force numeric columns explicitly
numeric_cols = ['value', 'cycle_relative_pos', 'rolling_mean', 'rolling_std',
                'rolling_min', 'rolling_max', 'delta_prev', 'delta_from_roll_mean',
                'value_mean', 'value_std', 'value_min', 'value_max',
                'value_median', 'value_last', 'delta_mean_last', 'value_count']

# Coerce all to numeric (in-place)
for col in numeric_cols:
    if col in df_enriched.columns:
        df_enriched[col] = pd.to_numeric(df_enriched[col], errors='coerce')

df_enriched.to_csv("enriched_long_PBIT_data.csv", index=False)
print("✅ Enriched long-format CSV saved as 'enriched_long_PBIT_data.csv'")

print(df_enriched.dtypes)

