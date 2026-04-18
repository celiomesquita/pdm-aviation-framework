import pandas as pd
import numpy as np
import re

def get_type_cycle(label):
    """
    Extracts the type and cycle from a label like 'F03.221'.
    
    Parameters:
        label (str): The input string, e.g., 'F03.221'
    
    Returns:
        tuple: (type, cycle) if matched, otherwise (label, 1)
    """
    match = re.match(r'^([A-Za-z]+\d+)(?:\.(\d+))?$', str(label).strip())

    if match:
        type = match.group(1)
        cycle = int(match.group(2)) if match.group(2) else 1
        return type, cycle
    else:
        return str(label), 1  # fallback: return label as type, default cycle=1


# === Load Excel File ===
excel_path = "PBIT_atuador_DESCARAC.xlsx"
xls = pd.ExcelFile(excel_path)

# === Load and Clean DATA Sheet ===
data_df = xls.parse('DATA')
data_df = data_df.dropna(how='all').reset_index(drop=True)

current_columns = list(data_df.columns)

# === Extract types and cycles from column names ===
types = []
cycles = []

for label in current_columns:
    type, cycle = get_type_cycle(label)
    types.append(type)
    cycles.append(cycle)

# === Drop the first two rows (headers or metadata) ===
data_df = data_df.iloc[2:].copy()

# === Convert to long format ===
long_format_rows = []

for i, col in enumerate(current_columns):
    type = types[i]
    cycle = cycles[i]
    try:
        values = data_df[col].astype(float).values
        for value in values:
            long_format_rows.append([type, cycle, value])
    except ValueError:
        print(f"⚠️ Skipping column {col}: could not convert values to float.")

long_df = pd.DataFrame(long_format_rows, columns=['type', 'cycle', 'value'])

# === Pivot to cycle-based sequences ===
reshaped_df = long_df.pivot_table(
    index=['type', 'cycle'],
    values='value',
    aggfunc=lambda x: list(x)
).reset_index()

# === Expand list of values into individual columns ===
value_expanded_df = pd.DataFrame(reshaped_df['value'].tolist(), index=reshaped_df.index)
value_expanded_df.columns = [f'v{i+1}' for i in range(value_expanded_df.shape[1])]

# === Final merged table ===
reshaped_df = pd.concat([reshaped_df[['type', 'cycle']], value_expanded_df], axis=1)
reshaped_df.to_csv("reshaped_data.csv", index=False)
print(f"✅ Reshaped data saved with shape: {reshaped_df.shape}")

# === Dataset A: Only rows with complete data from v1 to v244 ===
full_cols = [f'v{i}' for i in range(1, 245)]
dataset_a = reshaped_df[['type', 'cycle'] + full_cols].dropna()
dataset_a.to_csv("reshaped_data_a.csv", index=False)
print(f"✅ Dataset A saved: {dataset_a.shape[0]} rows with complete sequences to v244.")

# === Dataset B: All rows, but only from v1 to v122 ===
short_cols = [f'v{i}' for i in range(1, 123)]
dataset_b = reshaped_df[['type', 'cycle'] + short_cols]
dataset_b.to_csv("reshaped_data_b.csv", index=False)
print(f"✅ Dataset B saved: {dataset_b.shape[0]} rows with values to v122.")
