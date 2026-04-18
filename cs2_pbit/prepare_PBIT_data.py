import pandas as pd

# Load raw file
df_raw = pd.read_csv("PBIT_data.csv", header=None)

# Extract labels and cycles
types = df_raw.iloc[0].tolist()
cycles = df_raw.iloc[1].tolist()

# Extract and transpose the value rows (multi-feature matrix)
values_matrix = df_raw.iloc[2:].transpose()

# Force all values to be numeric (non-convertibles become NaN)
values_matrix = values_matrix.apply(pd.to_numeric, errors='coerce')

# Rename features as v1, v2, ..., vn
values_matrix.columns = [f'v{i+1}' for i in range(values_matrix.shape[1])]

# Combine into full DataFrame
df = pd.DataFrame({
    'type': types,
    'cycle': pd.to_numeric(cycles, errors='coerce')
})
df = pd.concat([df, values_matrix], axis=1)

# Drop rows with any missing values (optional)
df.dropna(inplace=True)

# Save to CSV
df.to_csv("prepared_PBIT_data.csv", index=False)
print("✅ Clean numeric CSV saved as 'prepared_PBIT_data.csv'")
