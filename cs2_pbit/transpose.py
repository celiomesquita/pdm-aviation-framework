import pandas as pd

# Load the CSV
df = pd.read_csv("PBIT_data.csv", header=None)

# Extract 'type' from the first row (column names in original)
types = df.iloc[0]
cycles = df.iloc[1]
data = df.iloc[2:]

# Transpose the data
data_transposed = data.transpose()

# Ensure all values are numeric (coerce errors to NaN, then fill or drop if needed)
data_transposed = data_transposed.apply(pd.to_numeric, errors='coerce')

# Rename the data columns to v1, v2, ..., vn
data_transposed.columns = [f"v{i+1}" for i in range(data_transposed.shape[1])]

# Create the final DataFrame
result = pd.DataFrame({
    "type": types,
    "cycle": cycles
})

result = pd.concat([result, data_transposed], axis=1)

# Save the result
result.to_csv("PBIT_data_transposed.csv", index=False)
