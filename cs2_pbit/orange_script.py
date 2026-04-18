import numpy as np
import pandas as pd
from Orange.data import Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable

# Unpack all parts of the Orange Table
attrs_df, class_df, metas_df = in_data.to_pandas_dfs()

# Get class column name from domain
class_column = in_data.domain.class_var.name

# Retrieve original string labels from the Orange Table's Y values
class_labels = [in_data.domain.class_var.str_val(val) for val in in_data.Y]
class_df_filtered = pd.DataFrame({class_column: class_labels})

# Identify value columns (those starting with 'v') only from attributes
value_cols = [col for col in attrs_df.columns if col.startswith("v")]
value_data = attrs_df[value_cols]

# Compute absolute correlation matrix
corr_matrix = value_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Threshold for strong correlation
threshold = 0.99997
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Make sure we don't drop the class or meta columns
reserved_cols = set([class_column]) | set(metas_df.columns)
to_drop = [col for col in to_drop if col not in reserved_cols]

# Log the result
print(f"Dropped {len(to_drop)} strongly correlated features: {to_drop}")

# Drop correlated columns from attrs_df only
filtered_attrs_df = attrs_df.drop(columns=to_drop)

# Remove class_column from attributes if it's there by mistake
if class_column in filtered_attrs_df.columns:
    filtered_attrs_df = filtered_attrs_df.drop(columns=[class_column])

# Define Orange domain
attributes = [ContinuousVariable(col) for col in filtered_attrs_df.columns]

# Now that we have the true class labels, define DiscreteVariable correctly
class_values = sorted(set(class_labels))
class_var = DiscreteVariable(class_column, values=class_values)

# Define meta variables
meta_vars = [StringVariable(col) for col in metas_df.columns]
domain = Domain(attributes, class_var, metas=meta_vars)

# Align DataFrames
filtered_attrs_df = filtered_attrs_df.reset_index(drop=True)
class_df_filtered = class_df_filtered.reset_index(drop=True)
metas_df = metas_df.reset_index(drop=True)

# Reconstruct final DataFrame in correct order
final_df = pd.concat([filtered_attrs_df, class_df_filtered, metas_df], axis=1)

# Rebuild Orange Table using Table.from_list (resolves DiscreteVariable by string)
out_data = Table.from_list(domain, final_df.values.tolist())


# pkill -f orange-canvas
