import pandas as pd

# Load raw data
df_raw = pd.read_csv("PBIT_data.csv", header=None)

# Extract metadata
types = df_raw.iloc[0].tolist()
cycles = pd.to_numeric(df_raw.iloc[1].tolist(), errors='coerce')

# Extract measurement values
values_matrix = df_raw.iloc[2:].transpose()
values_matrix = values_matrix.apply(pd.to_numeric, errors='coerce')

# Drop any rows with invalid type or cycle
valid_mask = pd.notna(cycles) & pd.notna(types)
types = pd.Series(types)[valid_mask].tolist()
cycles = pd.Series(cycles)[valid_mask].tolist()
values_matrix = values_matrix[valid_mask]

# Convert to long format
long_data = []

for i, (t, c) in enumerate(zip(types, cycles)):
    for v in values_matrix.iloc[i]:
        if pd.notna(v):
            long_data.append({'type': t, 'cycle': int(c), 'value': v})

# Create DataFrame and export
df_long = pd.DataFrame(long_data)
df_long.to_csv("long_format_PBIT_data.csv", index=False)

print("✅ Long-format CSV saved as 'long_format_PBIT_data.csv'")
