#!/usr/bin/env python3
"""
Predict NMR tensors for Xe atoms from XYZ files using the matten model.

This script reads an XYZ file containing multiple structures with Xe and C atoms,
converts them to PyMatGen Structure objects, and predicts NMR tensors for Xe atoms.

Usage:
    python xyz_nmr_predictor.py input.xyz [output.csv]
"""

import sys
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice
from matten.predict import predict
import re


def parse_xyz_file(xyz_file):
    """
    Parse XYZ file with multiple structures and extract lattice information.
    
    Args:
        xyz_file (str): Path to XYZ file
        
    Returns:
        list: List of dictionaries containing structure information
    """
    structures = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    structure_id = 0
    
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
            
        # Read number of atoms
        try:
            num_atoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
            
        # Read comment line with lattice information
        comment_line = lines[i + 1].strip()
        
        # Extract lattice parameters from comment line
        lattice_match = re.search(r'Lattice="([^"]+)"', comment_line)
        if lattice_match:
            lattice_str = lattice_match.group(1)
            lattice_values = [float(x) for x in lattice_str.split()]
            lattice_matrix = [
                [lattice_values[0], lattice_values[1], lattice_values[2]],
                [lattice_values[3], lattice_values[4], lattice_values[5]],
                [lattice_values[6], lattice_values[7], lattice_values[8]]
            ]
        else:
            # Default lattice if not found
            print(f"Warning: No lattice found for structure {structure_id}, using default")
            lattice_matrix = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 47.84310123]]
        
        # Read atomic coordinates
        atoms = []
        coordinates = []
        forces = []  # If available
        
        for j in range(num_atoms):
            line = lines[i + 2 + j].strip().split()
            
            if len(line) >= 4:
                atom = line[0]
                x, y, z = float(line[1]), float(line[2]), float(line[3])
                
                atoms.append(atom)
                coordinates.append([x, y, z])
                
                # Extract forces if available (columns 4, 5, 6)
                if len(line) >= 7:
                    fx, fy, fz = float(line[4]), float(line[5]), float(line[6])
                    forces.append([fx, fy, fz])
        
        # Store structure information
        structure_info = {
            'id': structure_id,
            'num_atoms': num_atoms,
            'lattice_matrix': lattice_matrix,
            'atoms': atoms,
            'coordinates': coordinates,
            'forces': forces if forces else None,
            'comment': comment_line
        }
        
        structures.append(structure_info)
        
        # Move to next structure
        i += num_atoms + 2
        structure_id += 1
        
        if structure_id % 10 == 0:
            print(f"Parsed {structure_id} structures...")
    
    print(f"Total structures parsed: {len(structures)}")
    return structures


def create_pymatgen_structure(structure_info):
    """
    Convert structure information to PyMatGen Structure object.
    
    Args:
        structure_info (dict): Structure information from XYZ parsing
        
    Returns:
        Structure: PyMatGen Structure object
    """
    lattice = Lattice(structure_info['lattice_matrix'])
    species = structure_info['atoms']
    coords = structure_info['coordinates']
    
    # Create structure with Cartesian coordinates
    structure = Structure(lattice, species, coords, coords_are_cartesian=True)
    
    return structure


def predict_nmr_tensors(structures, model_path, checkpoint_name, batch_size=1):
    """
    Predict NMR tensors for all Xe atoms in the structures.
    
    Args:
        structures (list): List of structure information dictionaries
        model_path (str): Path to model directory
        checkpoint_name (str): Checkpoint filename
        batch_size (int): Number of structures to process at once
        
    Returns:
        list: List of prediction results
    """
    results = []
    
    print(f"Starting NMR tensor prediction for {len(structures)} structures...")
    
    for i, structure_info in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}...")
        
        # Convert to PyMatGen structure
        structure = create_pymatgen_structure(structure_info)
        
        # Find Xe atom indices
        xe_indices = []
        for j, site in enumerate(structure.sites):
            if str(site.specie) == 'Xe':
                xe_indices.append(j)
        
        if not xe_indices:
            print(f"  No Xe atoms found in structure {i}")
            continue
        
        print(f"  Found {len(xe_indices)} Xe atoms at indices: {xe_indices}")
        
        try:
            # Predict NMR tensors
            tensors = predict(
                structure,
                model_identifier=model_path,
                checkpoint=checkpoint_name,
                is_atomic_tensor=True,
            )
            
            # Extract tensors for Xe atoms only
            for xe_idx in xe_indices:
                if xe_idx < len(tensors):
                    tensor = tensors[xe_idx]
                    
                    # Calculate isotropic shielding (sigma_iso)
                    sigma_iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0
                    
                    result = {
                        'structure_id': structure_info['id'],
                        'atom_index': xe_idx,
                        'element': 'Xe',
                        'x': structure_info['coordinates'][xe_idx][0],
                        'y': structure_info['coordinates'][xe_idx][1],
                        'z': structure_info['coordinates'][xe_idx][2],
                        'sigma_iso': sigma_iso,
                        'tensor_xx': tensor[0][0],
                        'tensor_xy': tensor[0][1],
                        'tensor_xz': tensor[0][2],
                        'tensor_yx': tensor[1][0],
                        'tensor_yy': tensor[1][1],
                        'tensor_yz': tensor[1][2],
                        'tensor_zx': tensor[2][0],
                        'tensor_zy': tensor[2][1],
                        'tensor_zz': tensor[2][2]
                    }
                    
                    results.append(result)
                    print(f"    Xe atom {xe_idx}: sigma_iso = {sigma_iso:.4f}")
                else:
                    print(f"    Warning: Tensor not found for Xe atom {xe_idx}")
        
        except Exception as e:
            print(f"  Error predicting for structure {i}: {e}")
            continue
    
    return results


def save_results(results, output_file):
    """Save prediction results to CSV file."""
    if not results:
        print("No results to save!")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total Xe predictions: {len(results)}")
    print("\nSummary statistics:")
    print(f"  Mean sigma_iso: {df['sigma_iso'].mean():.4f}")
    print(f"  Std sigma_iso:  {df['sigma_iso'].std():.4f}")
    print(f"  Min sigma_iso:  {df['sigma_iso'].min():.4f}")
    print(f"  Max sigma_iso:  {df['sigma_iso'].max():.4f}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python xyz_nmr_predictor.py input.xyz [output.csv]")
        print("\nExample:")
        print("  python xyz_nmr_predictor.py structures.xyz predictions.csv")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "nmr_predictions.csv"
    
    # Model configuration - adjust these paths as needed
    model_path = "/scratch/project_462000738/zakaryou/NMR-ML/MatTen/2xe_tba_cc3_high-th-level_3125_sanity-checked_dataset/training-puhti/matten_logs/matten_proj/ncsgsf99/checkpoints/"
    checkpoint_name = "last.ckpt"
    
    print(f"Input XYZ file: {xyz_file}")
    print(f"Output CSV file: {output_file}")
    print(f"Model path: {model_path}")
    print(f"Checkpoint: {checkpoint_name}")
    print()
    
    # Parse XYZ file
    print("Parsing XYZ file...")
    structures = parse_xyz_file(xyz_file)
    
    if not structures:
        print("No structures found in XYZ file!")
        sys.exit(1)
    
    # Predict NMR tensors
    results = predict_nmr_tensors(structures, model_path, checkpoint_name)
    
    # Save results
    save_results(results, output_file)
    
    print(f"\n✅ NMR tensor prediction completed!")
    print(f"Processed {len(structures)} structures")
    print(f"Generated {len(results)} Xe NMR predictions")


if __name__ == "__main__":
    main()
