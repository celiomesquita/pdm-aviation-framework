import pandas as pd
import numpy as np

# Load the transposed file
df = pd.read_csv("PBIT_data_transposed.csv")

# Extract only the value columns (those starting with 'v')
value_cols = [col for col in df.columns if col.startswith("v")]
value_data = df[value_cols]

# Compute correlation matrix (absolute value)
corr_matrix = value_data.corr().abs()

# Upper triangle of the correlation matrix (no repeats)
upper = corr_matrix.where(~pd.isnull(corr_matrix)).where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Threshold for strong correlation
threshold = 0.9997

# Identify columns to drop
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print(f"Number of strongly correlated columns to drop: {len(to_drop)}")

# Drop strongly correlated features
df_filtered = df.drop(columns=to_drop)

# Save cleaned dataset
df_filtered.to_csv("PBIT_data_filtered.csv", index=False)
