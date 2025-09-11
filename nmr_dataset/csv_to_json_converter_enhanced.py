#!/usr/bin/env python3
"""
Enhanced CSV to JSON converter for ML training

This script converts multiple CSV files with different lattice parameters to JSON format
and generates XYZ and CSV outputs from the test dataset.

Usage:
    python enhanced_csv_to_json_converter.py config.txt

Where config.txt contains:
structures_with_sigma_iso_1.csv,magnetic_shielding_tensors_1.csv,23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077
structures_with_sigma_iso_2.csv,magnetic_shielding_tensors_2.csv,25.76466077 0.0 0.0 0.0 25.76466077 0.0 0.0 0.0 25.76466077

Or direct usage:
    python enhanced_csv_to_json_converter.py structures_1.csv tensors_1.csv lattice_1 structures_2.csv tensors_2.csv lattice_2 ...

Output:
    - dataset_train.json
    - dataset_val.json
    - dataset_test.json
    - dataset_test_structures.xyz (from test dataset)
    - structures_with_sigma_iso_and_tensors.csv (from test dataset)
"""

import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    # Fallback: create a simple progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc
            self.current = 0
            if desc:
                print(f"{desc}: 0/{self.total}")
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
        
        def update(self, n=1):
            self.current += n
            if self.total > 0 and self.current % max(1, self.total // 20) == 0:
                print(f"{self.desc}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}: {self.total}/{self.total} (100.0%)")
        
        @staticmethod
        def write(s):
            print(s)


def parse_lattice_string(lattice_str):
    """Parse lattice string to matrix format."""
    # Remove 'Lattice="' and '"' from the string if present
    lattice_str = lattice_str.replace('Lattice="', '').replace('"', '')
    values = [float(x) for x in lattice_str.split()]
    
    # Reshape to 3x3 matrix
    matrix = [
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]]
    ]
    
    # Calculate lattice parameters
    a = values[0]  # assuming orthorhombic
    b = values[4]
    c = values[8]
    
    return {
        "matrix": matrix,
        "pbc": [True, True, True],
        "a": a,
        "b": b,
        "c": c,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "volume": a * b * c
    }


def create_structure_dict(molecule_df, lattice_info):
    """Create PyMatGen-style structure dictionary."""
    
    # Sort by atom_index to ensure correct order
    molecule_df = molecule_df.sort_values('atom_index')
    
    sites = []
    for _, row in molecule_df.iterrows():
        # Calculate fractional coordinates
        frac_x = row['x'] / lattice_info['a']
        frac_y = row['y'] / lattice_info['b'] 
        frac_z = row['z'] / lattice_info['c']
        
        site = {
            "species": [{"element": row['atom'], "occu": 1}],
            "abc": [frac_x, frac_y, frac_z],
            "xyz": [row['x'], row['y'], row['z']],
            "properties": {},
            "label": row['atom']
        }
        sites.append(site)
    
    structure = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure", 
        "charge": 0,
        "lattice": lattice_info,
        "sites": sites,
        "@version": None
    }
    
    return structure


def check_tensor_symmetry(tensor_matrix, tolerance=1e-8):
    """
    Check if a 3x3 tensor matrix is symmetric.
    
    Args:
        tensor_matrix: 3x3 list or array
        tolerance: Numerical tolerance for symmetry check
    
    Returns:
        bool: True if symmetric, False otherwise
    """
    tensor = np.array(tensor_matrix)
    
    # Check if tensor[i,j] == tensor[j,i] for all i,j
    diff = tensor - tensor.T
    max_asymmetry = np.max(np.abs(diff))
    
    return max_asymmetry <= tolerance


def verify_sigma_iso_calculation(tensor_matrix):
    """
    Verify that sigma_iso calculation is consistent between trace and eigenvalue methods.
    
    Args:
        tensor_matrix: 3x3 list or array
    
    Returns:
        tuple: (trace_method_result, eigenvalue_method_result, difference)
    """
    tensor = np.array(tensor_matrix)
    
    # Method 1: From trace of the tensor
    sigma_iso_trace = (tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3.0
    
    # Method 2: From eigenvalues
    eigenvalues = np.linalg.eigvals(tensor)
    sigma_iso_eigenvals = np.sum(eigenvalues) / 3.0
    
    difference = abs(sigma_iso_trace - sigma_iso_eigenvals)
    
    return sigma_iso_trace, sigma_iso_eigenvals, difference


def symmetrize_tensor(tensor_matrix):
    """
    Symmetrize a 3x3 tensor matrix by averaging off-diagonal elements.
    
    Args:
        tensor_matrix: 3x3 list or array
    
    Returns:
        tuple: (symmetric_tensor_list, new_sigma_iso)
    """
    tensor = np.array(tensor_matrix)
    
    # Symmetrize by averaging: T_sym = (T + T^T) / 2
    symmetric_tensor = (tensor + tensor.T) / 2.0
    
    # Calculate isotropic shielding from the trace of the symmetric tensor
    # sigma_iso = Tr(tensor) / 3 = (T_xx + T_yy + T_zz) / 3
    sigma_iso = (symmetric_tensor[0, 0] + symmetric_tensor[1, 1] + symmetric_tensor[2, 2]) / 3.0
    
    # Verify our calculation (optional check)
    trace_method, eigenval_method, diff = verify_sigma_iso_calculation(symmetric_tensor)
    if diff > 1e-10:
        print(f"Warning: sigma_iso calculation methods differ by {diff}")
    
    # Convert back to list format
    symmetric_tensor_list = symmetric_tensor.tolist()
    
    return symmetric_tensor_list, sigma_iso


def convert_csv_pair_to_json(structures_csv, tensors_csv, lattice_str, start_index=0):
    """
    Convert a pair of CSV files to JSON format with tensor symmetry checking.
    
    Args:
        structures_csv: Path to structures CSV file
        tensors_csv: Path to tensors CSV file
        lattice_str: Lattice parameters string
        start_index: Starting index for structure numbering
    
    Returns:
        tuple: (json_data_dict, next_start_index)
    """
    
    print(f"Processing CSV pair:")
    print(f"  Structures: {structures_csv}")
    print(f"  Tensors: {tensors_csv}")
    print(f"  Lattice: {lattice_str}")
    
    # Read CSV files
    structures_df = pd.read_csv(structures_csv)
    tensors_df = pd.read_csv(tensors_csv)
    
    print(f"  Loaded {len(structures_df)} structure records")
    print(f"  Loaded {len(tensors_df)} tensor records")
    
    # Parse lattice information
    lattice_info = parse_lattice_string(lattice_str)
    
    # Get unique molecules
    unique_molecules = structures_df['molecule_name'].unique()
    print(f"  Found {len(unique_molecules)} unique molecules")
    
    # Initialize JSON structure for this pair
    json_data = {
        "sigma_iso": {},
        "Qn": {},
        "structure": {},
        "species": {},
        "nmr_tensor": {},
        "atom_selector": {}
    }
    
    current_index = start_index
    asymmetric_count = 0
    total_tensors = 0
    
    # Process each molecule with progress bar
    for molecule_name in tqdm(unique_molecules, desc=f"Processing molecules", leave=False):
        # Get data for this molecule
        mol_structures = structures_df[structures_df['molecule_name'] == molecule_name]
        mol_tensors = tensors_df[tensors_df['molecule_name'] == molecule_name]
        
        # Merge structure and tensor data
        mol_data = mol_structures.merge(mol_tensors, on=['molecule_name', 'atom_index'], how='left')
        mol_data = mol_data.sort_values('atom_index')
        
        # Create structure dictionary
        structure_dict = create_structure_dict(mol_data, lattice_info)
        
        # Extract data for JSON
        sigma_iso_values = []
        qn_values = []
        nmr_tensors = []
        atom_selector = []
        species_list = []
        
        for _, row in mol_data.iterrows():
            element = row['atom']
            
            # Convert element symbol to atomic number
            element_to_atomic_number = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                'Xe': 54
            }
            atomic_number = element_to_atomic_number.get(element, 6)  # Default to C if not found
            species_list.append(atomic_number)
            
            # Only include Xe atoms for NMR training
            if element == 'Xe' and not pd.isna(row['sigma_iso']):
                atom_selector.append(True)
                qn_values.append(0)  # Xe atoms have Qn=0 (no bonding)
                
                # Create tensor matrix from CSV columns
                tensor_matrix = [
                    [row['XX'], row['XY'], row['XZ']],
                    [row['YX'], row['YY'], row['YZ']],
                    [row['ZX'], row['ZY'], row['ZZ']]
                ]
                
                total_tensors += 1
                
                # Check tensor symmetry
                if not check_tensor_symmetry(tensor_matrix):
                    asymmetric_count += 1
                    # Symmetrize the tensor and recalculate sigma_iso
                    symmetric_tensor, new_sigma_iso = symmetrize_tensor(tensor_matrix)
                    nmr_tensors.append(symmetric_tensor)
                    sigma_iso_values.append(new_sigma_iso)
                    
                    # Print warning for significant asymmetry
                    original_sigma_iso = row['sigma_iso']
                    if abs(new_sigma_iso - original_sigma_iso) > 1.0:
                        print(f"    Warning: Large sigma_iso change for {molecule_name} atom {row['atom_index']}: "
                              f"{original_sigma_iso:.3f} -> {new_sigma_iso:.3f}")
                else:
                    # Tensor is already symmetric
                    nmr_tensors.append(tensor_matrix)
                    sigma_iso_values.append(row['sigma_iso'])
            else:
                atom_selector.append(False)
        
        # Store in JSON structure using string index
        str_idx = str(current_index)
        json_data["sigma_iso"][str_idx] = sigma_iso_values
        json_data["Qn"][str_idx] = qn_values
        json_data["structure"][str_idx] = structure_dict
        json_data["species"][str_idx] = species_list
        json_data["nmr_tensor"][str_idx] = nmr_tensors
        json_data["atom_selector"][str_idx] = atom_selector
        
        current_index += 1
    
    print(f"  Processed {current_index - start_index} structures")
    if total_tensors > 0:
        print(f"  Tensor symmetry check: {asymmetric_count}/{total_tensors} tensors were asymmetric and corrected")
        if asymmetric_count > 0:
            print(f"  Asymmetry rate: {asymmetric_count/total_tensors*100:.1f}%")
    
    return json_data, current_index


def merge_json_data(json_data_list):
    """Merge multiple JSON data dictionaries into one."""
    
    merged_data = {
        "sigma_iso": {},
        "Qn": {},
        "structure": {},
        "species": {},
        "nmr_tensor": {},
        "atom_selector": {}
    }
    
    current_index = 0
    
    for json_data in json_data_list:
        for old_key in json_data["structure"].keys():
            new_key = str(current_index)
            
            for field in merged_data.keys():
                merged_data[field][new_key] = json_data[field][old_key]
            
            current_index += 1
    
    return merged_data


def split_dataset(json_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split dataset into train/validation/test sets."""
    
    # Get all structure indices
    all_indices = list(json_data["structure"].keys())
    total_structures = len(all_indices)
    
    print(f"Splitting {total_structures} structures:")
    print(f"  Train: {train_ratio*100:.1f}% ({int(total_structures*train_ratio)} structures)")
    print(f"  Val:   {val_ratio*100:.1f}% ({int(total_structures*val_ratio)} structures)")
    print(f"  Test:  {test_ratio*100:.1f}% ({int(total_structures*test_ratio)} structures)")
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        all_indices, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    # Second split: val vs test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=random_state
    )
    
    def create_subset(indices):
        """Create a subset of the data with the given indices."""
        subset = {
            "sigma_iso": {},
            "Qn": {},
            "structure": {},
            "species": {},
            "nmr_tensor": {},
            "atom_selector": {}
        }
        
        # Reindex from 0
        for new_idx, old_idx in enumerate(indices):
            str_new_idx = str(new_idx)
            subset["sigma_iso"][str_new_idx] = json_data["sigma_iso"][old_idx]
            subset["Qn"][str_new_idx] = json_data["Qn"][old_idx]
            subset["structure"][str_new_idx] = json_data["structure"][old_idx]
            subset["species"][str_new_idx] = json_data["species"][old_idx]
            subset["nmr_tensor"][str_new_idx] = json_data["nmr_tensor"][old_idx]
            subset["atom_selector"][str_new_idx] = json_data["atom_selector"][old_idx]
        
        return subset
    
    train_data = create_subset(train_indices)
    val_data = create_subset(val_indices)
    test_data = create_subset(test_indices)
    
    return train_data, val_data, test_data


def save_json_datasets(train_data, val_data, test_data, output_prefix="dataset"):
    """Save the three datasets as JSON files."""
    
    datasets = {
        f"{output_prefix}_train.json": train_data,
        f"{output_prefix}_val.json": val_data,
        f"{output_prefix}_test.json": test_data
    }
    
    for filename, data in datasets.items():
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Print statistics
        num_structures = len(data["structure"])
        total_xe_atoms = sum(len(data["sigma_iso"][str(i)]) for i in range(num_structures))
        
        print(f"Saved {filename}:")
        print(f"  Structures: {num_structures}")
        print(f"  Xe atoms with NMR data: {total_xe_atoms}")


def extract_lattice_string(lattice_dict):
    """Extract lattice parameters as a string from lattice dictionary."""
    matrix = lattice_dict['matrix']
    lattice_values = []
    
    for row in matrix:
        lattice_values.extend(row)
    
    return " ".join([f"{val}" for val in lattice_values])


def generate_xyz_from_test_data(test_data, output_file="dataset_test_structures.xyz"):
    """Generate XYZ file from test dataset."""
    
    print(f"Generating XYZ file from test data: {output_file}")
    
    structure_ids = list(test_data['structure'].keys())
    
    with open(output_file, 'w') as f:
        for i, struct_id in enumerate(tqdm(structure_ids, desc="Writing XYZ structures")):
            # Get structure data
            structure = test_data['structure'][struct_id]
            sites = structure['sites']
            lattice = structure['lattice']
            
            # Extract lattice string
            lattice_str = extract_lattice_string(lattice)
            
            # Write number of atoms
            f.write(f"{len(sites)}\n")
            
            # Write comment line with lattice and properties
            comment_line = f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'
            f.write(comment_line + "\n")
            
            # Write atomic coordinates
            for site in sites:
                element = site['species'][0]['element']
                x, y, z = site['xyz']
                
                f.write(f"{element:>2} {x:15.8f} {y:15.8f} {z:15.8f}\n")
    
    print(f"XYZ file written: {output_file}")
    return output_file


def generate_csv_from_test_data(test_data, output_file="structures_with_sigma_iso_and_tensors.csv"):
    """Generate CSV file with NMR data from test dataset with symmetric tensors."""
    
    print(f"Generating CSV file from test data: {output_file}")
    
    csv_data = []
    structure_ids = list(test_data['structure'].keys())
    asymmetric_count = 0
    total_tensors = 0
    
    for struct_id in tqdm(structure_ids, desc="Processing structures for CSV"):
        structure = test_data['structure'][struct_id]
        sites = structure['sites']
        atom_selector = test_data['atom_selector'][struct_id]
        sigma_iso_values = test_data['sigma_iso'][struct_id]
        nmr_tensors = test_data['nmr_tensor'][struct_id]
        
        # Create molecule name
        molecule_name = f"xe_cnt_{struct_id}"
        
        # Track NMR data index
        nmr_index = 0
        
        for atom_idx, site in enumerate(sites):
            element = site['species'][0]['element']
            x, y, z = site['xyz']
            
            # Check if this atom has NMR data
            if atom_selector[atom_idx] and nmr_index < len(sigma_iso_values):
                sigma_iso = sigma_iso_values[nmr_index]
                
                # Get tensor components
                if nmr_index < len(nmr_tensors):
                    tensor = nmr_tensors[nmr_index]
                    total_tensors += 1
                    
                    # Check and correct tensor symmetry
                    if not check_tensor_symmetry(tensor):
                        asymmetric_count += 1
                        tqdm.write(f"  Warning: Asymmetric tensor found for {molecule_name} atom {atom_idx}, symmetrizing...")
                        tensor, new_sigma_iso = symmetrize_tensor(tensor)
                        sigma_iso = new_sigma_iso
                    
                    # Extract tensor components in the order: XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ
                    xx, xy, xz = tensor[0][0], tensor[0][1], tensor[0][2]
                    yx, yy, yz = tensor[1][0], tensor[1][1], tensor[1][2]
                    zx, zy, zz = tensor[2][0], tensor[2][1], tensor[2][2]
                    
                    # Verify symmetry after correction
                    if abs(xy - yx) > 1e-10 or abs(xz - zx) > 1e-10 or abs(yz - zy) > 1e-10:
                        tqdm.write(f"  Error: Tensor still asymmetric after correction for {molecule_name} atom {atom_idx}")
                    
                    # Add to CSV data
                    csv_data.append({
                        'molecule_name': molecule_name,
                        'atom_index': atom_idx,
                        'atom': element,
                        'x': x,
                        'y': y,
                        'z': z,
                        'sigma_iso': sigma_iso,
                        'XX': xx,
                        'YX': yx,
                        'ZX': zx,
                        'XY': xy,
                        'YY': yy,
                        'ZY': zy,
                        'XZ': xz,
                        'YZ': yz,
                        'ZZ': zz
                    })
                
                nmr_index += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    print(f"CSV file written: {output_file}")
    print(f"Total NMR entries: {len(csv_data)}")
    
    if total_tensors > 0:
        print(f"Tensor symmetry check: {asymmetric_count}/{total_tensors} tensors were asymmetric and corrected")
        if asymmetric_count > 0:
            print(f"Asymmetry rate: {asymmetric_count/total_tensors*100:.1f}%")
    
    if len(csv_data) > 0:
        element_counts = df['atom'].value_counts()
        print("NMR data by element:")
        for element, count in element_counts.items():
            print(f"  {element}: {count} atoms")
    
    return output_file


def parse_config_file(config_file):
    """Parse configuration file with CSV pairs and lattice parameters."""
    
    csv_pairs = []
    
    with open(config_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) != 3:
                print(f"Warning: Skipping invalid line {line_num}: {line}")
                continue
            
            structures_csv = parts[0].strip()
            tensors_csv = parts[1].strip()
            lattice_str = parts[2].strip()
            
            # Verify files exist
            if not os.path.exists(structures_csv):
                print(f"Warning: File not found: {structures_csv}")
                continue
            if not os.path.exists(tensors_csv):
                print(f"Warning: File not found: {tensors_csv}")
                continue
            
            csv_pairs.append((structures_csv, tensors_csv, lattice_str))
    
    return csv_pairs


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python enhanced_csv_to_json_converter.py config.txt")
        print("   or: python enhanced_csv_to_json_converter.py structures_1.csv tensors_1.csv lattice_1 [structures_2.csv tensors_2.csv lattice_2] ...")
        print("\nConfig file format:")
        print("structures_with_sigma_iso_1.csv,magnetic_shielding_tensors_1.csv,23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077")
        print("structures_with_sigma_iso_2.csv,magnetic_shielding_tensors_2.csv,25.76466077 0.0 0.0 0.0 25.76466077 0.0 0.0 0.0 25.76466077")
        print("\nOutput:")
        print("  - dataset_train.json")
        print("  - dataset_val.json") 
        print("  - dataset_test.json")
        print("  - dataset_test_structures.xyz (from test dataset)")
        print("  - structures_with_sigma_iso_and_tensors.csv (from test dataset)")
        sys.exit(1)
    
    # Check if first argument is a config file
    if len(sys.argv) == 2 and sys.argv[1].endswith('.txt'):
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"Error: Config file '{config_file}' not found.")
            sys.exit(1)
        
        print(f"Reading configuration from: {config_file}")
        csv_pairs = parse_config_file(config_file)
        
    else:
        # Parse command line arguments (triplets of structures, tensors, lattice)
        if (len(sys.argv) - 1) % 3 != 0:
            print("Error: Arguments must be in triplets of (structures.csv, tensors.csv, lattice_string)")
            sys.exit(1)
        
        csv_pairs = []
        for i in range(1, len(sys.argv), 3):
            if i + 2 >= len(sys.argv):
                break
            structures_csv = sys.argv[i]
            tensors_csv = sys.argv[i + 1]
            lattice_str = sys.argv[i + 2]
            csv_pairs.append((structures_csv, tensors_csv, lattice_str))
    
    if not csv_pairs:
        print("Error: No valid CSV pairs found.")
        sys.exit(1)
    
    print(f"Found {len(csv_pairs)} CSV pair(s) to process")
    
    # Process all CSV pairs
    json_data_list = []
    current_index = 0
    
    for i, (structures_csv, tensors_csv, lattice_str) in enumerate(tqdm(csv_pairs, desc="Processing CSV pairs")):
        print(f"\n--- Processing pair {i+1}/{len(csv_pairs)} ---")
        json_data, next_index = convert_csv_pair_to_json(
            structures_csv, tensors_csv, lattice_str, current_index
        )
        json_data_list.append(json_data)
        current_index = next_index
    
    # Merge all JSON data
    print(f"\nMerging data from {len(json_data_list)} sources...")
    merged_json_data = merge_json_data(json_data_list)
    
    print(f"Total structures: {len(merged_json_data['structure'])}")
    
    # Split into train/val/test
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_dataset(merged_json_data)
    
    # Save datasets
    print("\nSaving JSON files...")
    save_json_datasets(train_data, val_data, test_data)
    
    print("\nGenerating XYZ and CSV from test dataset...")
    print("Note: All tensors will be checked for symmetry and corrected if needed.")
    xyz_file = generate_xyz_from_test_data(test_data)
    csv_file = generate_csv_from_test_data(test_data)
    
    print(f"\n✅ Conversion completed successfully!")
    print("Generated files:")
    print("  - dataset_train.json")
    print("  - dataset_val.json") 
    print("  - dataset_test.json")
    print(f"  - {xyz_file}")
    print(f"  - {csv_file}")


if __name__ == "__main__":
    main()
