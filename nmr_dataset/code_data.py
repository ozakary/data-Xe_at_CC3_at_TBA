import pandas as pd
import numpy as np

# Load structures.csv
structures_df = pd.read_csv('./structures.csv')
print(f"Loaded structures data with shape: {structures_df.shape} and columns: {structures_df.columns.tolist()}")

# Load magnetic shielding tensors
tensors_df = pd.read_csv('./magnetic_shielding_tensors.csv')
print(f"Loaded magnetic shielding tensors with shape: {tensors_df.shape} and columns: {tensors_df.columns.tolist()}")

# Extract and process tensor components
tensor_components = tensors_df.iloc[:, 2:].values  # get the 9 tensor values
tensor_components = tensor_components.reshape(-1, 3, 3)
tensor_components = 0.5 * (tensor_components + np.transpose(tensor_components, (0, 2, 1)))  # symmetrize

# Compute eigenvalues and derived parameters
w, _ = np.linalg.eigh(tensor_components)
sigma_iso = np.mean(w, axis=1)

# Build DataFrame for calculated values
params_df = tensors_df[['molecule_name', 'atom_index']].copy()
params_df['sigma_iso'] = sigma_iso

# Merge sigma_iso into structures.csv based on molecule_name and atom_index
merged_df = pd.merge(structures_df, params_df[['molecule_name', 'atom_index', 'sigma_iso']],
                     on=['molecule_name', 'atom_index'], how='left')

# Save new CSV with sigma_iso appended
merged_df.to_csv('structures_with_sigma_iso.csv', index=False)
print("Saved merged data to 'structures_with_sigma_iso.csv'")
