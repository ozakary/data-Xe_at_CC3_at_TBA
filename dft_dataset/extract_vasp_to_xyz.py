########################################################################################################################################
#################################< NequIP-Allegro <system>.xyz input by O. Zakary - 2024 >##############################################
########################################################################################################################################
#
#---------------------------------------------------------------------------------------------------------------------------------------
# This script extracts the atomic coordinates (x, y, z), forces (Fx, Fy, Fz), stress tensor (Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, 
# Szz), and the total energy issued from a serie of DFT calculations with PBC using the Vienna Abinitio Simulation Package (VASP).
# These quantities will be later written in the form of a trajectory input file <system>.xyz for subsequent Machine-Learning/Neural-
# Network simulations using the E(3)-equivariant graph neural network algorithmss NequIP (https://doi.org/10.1038/s41467-023-36329-y) 
# and Allegro (https://doi.org/10.1038/s41467-022-29939-5).
# Requirements : -> OUTCAR files in path : vasp_folders_<system>/DFT-NMR_iter-{i}/OUT-{i}, with i = 0, N
#                -> POSCAR files in path : vasp_folders_<system>/DFT-NMR_iter-{i}/OUT-{i}, with i = 0, N
#
# Questions ? => contact zakaryouail@gmail.com (^_^)
#---------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import re

def extract_coordinates_and_forces(outcar_file, poscar_file):
    coordinates = []
    forces = []
    atomic_species = []

    with open(outcar_file, 'r') as f:
        lines = f.readlines()

    start_extraction = False
    for line in lines:
        if "POSITION" in line:
            start_extraction = True
            continue
        elif "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in line:
            break

        if start_extraction and "--" not in line:
            # Extract atomic coordinates and forces
            match = re.match(r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line)
            if match:
                coords = [float(match.group(i)) for i in range(1, 4)]
                coordinates.append(coords)
                force = [float(match.group(i)) for i in range(4, 7)]
                forces.append(force)

    # Extract atomic species and repeat each species according to the number of atoms
    with open(poscar_file, 'r') as f:
        lines = f.readlines()
        atomic_species_line = lines[5].split()
        num_atoms_line = lines[6].split()
        for species, num_atoms in zip(atomic_species_line, num_atoms_line):
            atomic_species.extend([species] * int(num_atoms))

    return np.array(coordinates), np.array(forces), atomic_species


def write_coordinates_and_forces(coordinates, forces, atomic_species, output_file, num_atoms, lattice_params, total_energy, stress_tensor, code_num):
    # Write the lattice parameters, energy, and stress tensor in the same line
    lattice_string = f"Lattice=\"{' '.join(map(str, lattice_params))}\" Properties=species:S:1:pos:R:3:forces:R:3 original_dataset_index={code_num} energy={total_energy} stress=\"{' '.join(map(str, stress_tensor))}\" free_energy={total_energy} pbc=\"T T T\"\n"

    # Write data line by line
    with open(output_file, 'w') as f:
        # Write the total number of atoms
        f.write(f"{num_atoms}\n")
        f.write(lattice_string)
        for i in range(len(coordinates)):
            # Format each line with 8 significant digits
            line = "{}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                atomic_species[i], coordinates[i][0], coordinates[i][1], coordinates[i][2],
                forces[i][0], forces[i][1], forces[i][2]
            )
            f.write(line)

    print("Data written to {}".format(output_file))


def modify_dataset_indexes(input_file, output_file, start_index):
    # Read the input file and get the last value of 'original_dataset_index'
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Write modified data to output file
    with open(output_file, 'w') as f:
        new_index = start_index
        for line in lines:
            # Find and replace 'original_dataset_index=' value
            index_match = re.search(r'original_dataset_index=(\d+)', line)
            if index_match:
                original_index = int(index_match.group(1))
                line = re.sub(r'original_dataset_index=\d+', f'original_dataset_index={new_index}', line)
                new_index += 1
            f.write(line)

def main():
    # Define the output directory path for individual ML files
    output_directory = "ML_files"
    os.makedirs(output_directory, exist_ok=True)

    # Define the output file name for the concatenated ML_total file
    output_file = "ML_total"

    # Initialize an empty array to store the paths of individual ML files
    ML_files = []

    # Iterate over the range of directories corresponding to each calculation
    for i in range(0, 13800, 60):
        # Generate the directory path for the current calculation
        directory_path = f"vasp_folders_2Xe_CC3_TBA/DFT-NMR_iter-{i}/OUT-{i}"
        code_num = i

        # Construct the file paths for OUTCAR and POSCAR files
        outcar_file = f"{directory_path}/OUTCAR"
        poscar_file = f"{directory_path}/POSCAR"

        # Generate the output file path for the ML file of the current calculation
        output_file_path = f"{output_directory}/ML_{i}.txt"

        # Extract data from the current calculation and write it to the ML file
        coordinates, forces, atomic_species = extract_coordinates_and_forces(outcar_file, poscar_file)
        num_atoms = len(coordinates)  # Get the total number of atoms
        lattice_params = []
        with open(poscar_file, 'r') as f:
            lines = f.readlines()
            for j in range(2, 5):  # Lines 2 to 4
                lattice_params.extend([float(param) for param in lines[j].split()])
        # Extract lattice volume from lattice parameters
        cell_volume = None
        with open(outcar_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "volume of cell" in line:
                    cell_volume = float(line.split()[-1])
                    break
            else:
                print("Error: Unable to find 'volume of cell' line in OUTCAR file.")

        # Check if cell volume was successfully extracted
        if cell_volume is None:
            print("Error: Unable to extract cell volume from OUTCAR file.")
            continue  # Skip to the next iteration of the loop

        # Extract total energy from OUTCAR file
        total_energy = None
        with open(outcar_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "  free  energy   TOTEN" in line:
                    total_energy = float(line.split()[-2])  # Extract the second-to-last element
                    break
            else:
                print("Error: Unable to find 'TOTEN' line in OUTCAR file.")

        # Extract stress tensor from OUTCAR file
        stress_tensor = None
        with open(outcar_file, 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                if "Total    " in line:
                    # Extract stress tensor values from the line
                    stress_tensor_values = [float(val) for val in line.split("Total    ")[1].split() if val != '']
                    # Rearrange the values to match the expected order in the stress tensor
                    rearranged_values = [stress_tensor_values[0], stress_tensor_values[3], stress_tensor_values[5],  # xx, yy, zz
                                         stress_tensor_values[3], stress_tensor_values[1], stress_tensor_values[4],  # xy, yz, zx
                                         stress_tensor_values[5], stress_tensor_values[4], stress_tensor_values[2]]  # xz, yx, zy
                    stress_tensor = rearranged_values
                    break
            else:
                print("Error: Unable to find stress tensor values in OUTCAR file.")
        # Multiply stress tensor components by minus the cell volume
        stress_tensor = [component / (-cell_volume) for component in stress_tensor]

        # Write coordinates and forces data to the ML file
        if total_energy is not None and num_atoms > 0 and len(forces) > 0 and stress_tensor is not None:
            print(f"Coordinates, forces, and stress tensor extracted successfully for calculation {i}.")
            write_coordinates_and_forces(coordinates, forces, atomic_species, output_file_path, num_atoms, lattice_params, total_energy, stress_tensor, code_num)
            ML_files.append(output_file_path)  # Append the original file path
        else:
            print(f"Error: Unable to extract coordinates, forces, total energy, or stress tensor from the files of calculation {i}.")

    # Concatenate the individual ML files into a single ML_total file
    concatenate_ML_files(ML_files, output_file)

    # Modify dataset indexes in the concatenated ML_total file
    modify_dataset_indexes(output_file, "ML_total_modified", start_index=197)

    print("All ML files concatenated into ML_total_modified")


def concatenate_ML_files(file_paths, output_file):
    with open(output_file, 'w') as outfile:
        for file_path in file_paths:
            with open(file_path) as infile:
                outfile.write(infile.read())


if __name__ == "__main__":
    main()
