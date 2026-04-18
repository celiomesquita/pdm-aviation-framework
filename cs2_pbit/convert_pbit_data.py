import pandas as pd
import numpy as np
import re

"""
This script transforms data from a specific Excel sheet format into a "long" CSV format.
The expected Excel structure has been updated based on user feedback:
- First row (Excel header): Column labels like 'F01', 'F03.221', parsed for type and cycle.
- Second row (Excel data, data_df_raw.iloc[0]): Cycle numbers or test indices. (Ignored for serials now)
- Third row (Excel data, data_df_raw.iloc[1]): Actual Serial numbers ('SN 001', etc.). <--- CORRECTED SOURCE FOR SERIALS
- Fourth row (Excel data, data_df_raw.iloc[2]): Assumed to be an empty row. (Skipped for values)
- Fifth row (Excel data, data_df_raw.iloc[3]) onwards: Measurement values. <--- CORRECTED START FOR VALUES

The output CSV will have columns: serial, type, cycle, value.
"""

def get_type_cycle(label):
    """
    Extracts the type and cycle from a label like 'F03.221'.
    """
    match = re.match(r'^([A-Za-z]+\d+)(?:\.(\d+))?$', str(label).strip())
    if match:
        type_val = match.group(1)
        cycle = int(match.group(2)) if match.group(2) else 1 # Default cycle to 1 if not specified
        return type_val, cycle
    else:
        return str(label), 1

# === Configuration ===
excel_path = "PBIT_atuador_DESCARAC_r1.xlsx" # Input Excel file name
output_csv_path = "reshaped_fault_data_long_corrected_serial.csv" # Updated Output CSV file name

# === Load Excel File ===
try:
    xls = pd.ExcelFile(excel_path)
except FileNotFoundError:
    print(f"❌ Error: Excel file not found at '{excel_path}'. Please check the path.")
    exit()

# === Load and Clean DATA Sheet ===
try:
    data_df_raw = xls.parse('DATA')
except ValueError as e:
    print(f"❌ Error: Sheet 'DATA' not found in '{excel_path}' or other parsing error: {e}")
    exit()

data_df_raw = data_df_raw.dropna(how='all').reset_index(drop=True)

if data_df_raw.empty:
    print("❌ Error: 'DATA' sheet is empty after dropping all-NaN rows.")
    exit()

current_columns = list(data_df_raw.columns)
types_list = []
cycles_list = []
sn_list = []

# === Extract Metadata (Types, Cycles, Serial Numbers) ===

# Serial Numbers are now expected from the third Excel row (data_df_raw.iloc[1])
raw_serial_values_from_row = []
if len(data_df_raw) >= 2: # Need at least 2 rows to access .iloc[1] (Excel row 3)
    raw_serial_values_from_row = data_df_raw.iloc[1].astype(str).values
else:
    print(f"❌ Error: Not enough rows in 'DATA' sheet to extract serial numbers (expected at data_df_raw.iloc[1] corresponding to Excel row 3). Found {len(data_df_raw)} data rows after header.")

for i, label in enumerate(current_columns):
    type_val, cycle_val = get_type_cycle(label)
    types_list.append(type_val)
    cycles_list.append(cycle_val)

    if i < len(raw_serial_values_from_row):
        sn_list.append(raw_serial_values_from_row[i])
    else:
        placeholder_serial = f"MISSING_SERIAL_COL_IDX_{i}"
        sn_list.append(placeholder_serial)
        if len(raw_serial_values_from_row) > 0:
             print(f"⚠️ Warning: Missing serial for column '{label}' (index {i}). Serials row (Excel row 3 / iloc[1]) is shorter than columns. Using placeholder: '{placeholder_serial}'.")
        elif len(data_df_raw) >= 2 : # Serials row (iloc[1]) was present but empty
             print(f"⚠️ Warning: Serials row (Excel row 3 / iloc[1]) was empty. Using placeholder for column '{label}' (index {i}): '{placeholder_serial}'.")

if len(data_df_raw) < 2 and len(current_columns) > 0:
    print(f"⚠️ Warning: Serial numbers row (expected at Excel row 3 / iloc[1]) was missing. All serials will be placeholders.")


# === Extract Data Values ===
# Data values are now expected from the fifth Excel row (data_df_raw.iloc[3]),
# after skipping Excel row 4 (iloc[2]), which is assumed empty.
if len(data_df_raw) < 4: # Need at least 4 rows for data to start at .iloc[3] (Excel row 5)
    print(f"⚠️ Warning: Not enough rows in 'DATA' sheet for data values (expected from data_df_raw.iloc[3:] corresponding to Excel row 5 onwards). Found {len(data_df_raw)} data rows after header. Measurement values may be empty.")
    data_values_df = pd.DataFrame(columns=current_columns)
else:
    data_values_df = data_df_raw.iloc[3:].copy().reset_index(drop=True)

# === Construct Intermediate Data Structure ===
processed_data_for_long_format = []
for i, col_name in enumerate(current_columns):
    if i >= len(types_list) or i >= len(cycles_list) or i >= len(sn_list):
        print(f"⚠️ Critical Warning: Mismatch in metadata list lengths for column '{col_name}' (index {i}). Skipping this column.")
        continue

    serial_val = sn_list[i]
    type_val = types_list[i]
    cycle_val = cycles_list[i]
    value_sequence = []

    try:
        if col_name not in data_values_df.columns:
            if data_values_df.empty and len(data_df_raw) < 4: # Adjusted check
                 print(f"ℹ️ Info: data_values_df is empty for column '{col_name}'. No values to process.")
            else:
                print(f"⚠️ Warning: Column '{col_name}' not found in data_values_df. No values to process for this column.")
        else:
            value_sequence = list(data_values_df[col_name].dropna().astype(float).values)
        
        processed_data_for_long_format.append({
            'serial': serial_val,
            'type': type_val,
            'cycle': cycle_val,
            'value_list': value_sequence
        })
    except ValueError:
        print(f"⚠️ Warning: Skipping column {col_name} for processing. Could not convert its values to float.")
    except KeyError:
        print(f"⚠️ Warning: Column {col_name} not found in data_values_df (KeyError).")

# === Create DataFrame and Transform to Long Format ===
final_long_df = pd.DataFrame(columns=['serial', 'type', 'cycle', 'value'])

if processed_data_for_long_format:
    structured_df_with_lists = pd.DataFrame(processed_data_for_long_format)

    if not structured_df_with_lists.empty and 'value_list' in structured_df_with_lists.columns:
        exploded_df = structured_df_with_lists.explode('value_list')
        
        if not exploded_df.empty:
            exploded_df = exploded_df.rename(columns={'value_list': 'value'})
            exploded_df['value'] = pd.to_numeric(exploded_df['value'])
            
            expected_cols = ['serial', 'type', 'cycle', 'value']
            if all(col in exploded_df.columns for col in expected_cols):
                final_long_df = exploded_df[expected_cols]
            else:
                missing_cols = [col for col in expected_cols if col not in exploded_df.columns]
                print(f"❌ Error: Could not create final long DataFrame. Missing columns after explode/rename: {missing_cols}")
        else:
            print("ℹ️ Info: Data became empty after exploding lists (all measurement lists might have been empty).")
    elif structured_df_with_lists.empty:
        print("ℹ️ Info: Intermediate DataFrame ('structured_df_with_lists') was empty. Cannot create long format.")
    else:
        print("⚠️ Warning: 'value_list' column not found in the intermediate DataFrame. Cannot create long format.")
else:
    print("ℹ️ Info: No data was processed into the intermediate structure. Cannot create long format.")

# === Save the Long Format Data to CSV ===
if not final_long_df.empty:
    try:
        final_long_df.to_csv(output_csv_path, index=False)
        print(f"✅ Successfully saved long format data to '{output_csv_path}' with shape: {final_long_df.shape}")
    except Exception as e:
        print(f"❌ Error: Could not save the long format CSV to '{output_csv_path}': {e}")
else:
    print(f"❌ Final long format DataFrame was empty. No '{output_csv_path}' file was saved.")