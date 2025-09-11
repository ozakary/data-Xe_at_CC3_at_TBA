import pandas as pd
import numpy as np

def process_shielding_tensors():
    # Read the files
    structures_df = pd.read_csv('./dataset_schnet_atomic_coordinates/structures.csv')
    tensors_df = pd.read_csv('./dataset_schnet_shielding_tensors/magnetic_shielding_tensors.csv')
    
    # Create a new DataFrame for the modified tensors with the same structure
    new_tensors_df = pd.DataFrame(columns=tensors_df.columns)
    
    # Process each molecule separately
    for molecule in structures_df['molecule_name'].unique():
        # Get data for current molecule
        mol_struct = structures_df[structures_df['molecule_name'] == molecule]
        mol_tensors = tensors_df[tensors_df['molecule_name'] == molecule]
        
        # Find indices where atom is Xe in structures file
        xe_mask = mol_struct['atom'] == 'Xe'
        xe_indices = mol_struct[xe_mask].index.tolist()
        
        # Find rows with non-zero tensor values
        non_zero_mask = (
            (mol_tensors['XX'] != 0) | 
            (mol_tensors['YY'] != 0) | 
            (mol_tensors['ZZ'] != 0)
        )
        non_zero_tensors = mol_tensors[non_zero_mask].reset_index(drop=True)
        
        # Create a template of zero tensors for this molecule
        zero_tensor_template = pd.DataFrame({
            'molecule_name': [molecule] * len(mol_struct),
            'atom_index': mol_struct['atom_index'].values,
            'XX': [0.0] * len(mol_struct),
            'YX': [0.0] * len(mol_struct),
            'ZX': [0.0] * len(mol_struct),
            'XY': [0.0] * len(mol_struct),
            'YY': [0.0] * len(mol_struct),
            'ZY': [0.0] * len(mol_struct),
            'XZ': [0.0] * len(mol_struct),
            'YZ': [0.0] * len(mol_struct),
            'ZZ': [0.0] * len(mol_struct)
        })
        
        # Place non-zero tensor values at Xe positions
        for xe_idx, tensor_row in zip(xe_indices, non_zero_tensors.itertuples()):
            zero_tensor_template.loc[xe_idx - mol_struct.index[0], 'XX':'ZZ'] = tensor_row[3:]
        
        # Append to the new DataFrame
        new_tensors_df = pd.concat([new_tensors_df, zero_tensor_template], ignore_index=True)
    
    # Save the modified file
    new_tensors_df.to_csv('./dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv', 
                         index=False)
    
    print("Processing completed. Modified file saved as 'magnetic_shielding_tensors_modified.csv'")

if __name__ == "__main__":
    process_shielding_tensors()
