import os
import re
import csv
import pandas as pd

def write_tensors_to_csv(output_path, tensors):
    tensors.to_csv(output_path, index=False)

def write_coordinates_to_csv(output_path, coordinates):
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'])
        csvwriter.writerows(coordinates)

def extract_shielding_tensors(mpshift_path, atomic_coordinates, structure_id):
    with open(mpshift_path, 'r') as file:
        lines = file.readlines()

    tensor_data = []
    molecule_name = f'Xe_TBA_{structure_id}'
    
    in_shielding_section = False
    atom_index = -1
    atom_dict = {coord[1]: coord[0] for coord in atomic_coordinates}  # Mapping index to atom symbol
    missing_tensors_atoms = []  # To track atoms with missing tensors

    line_iterator = iter(lines)
    for line in line_iterator:
        if ">>>>> DFT MAGNETIC SHIELDINGS <<<<<" in line or ">>>>> SCF MAGNETIC SHIELDINGS <<<<<" in line:
            in_shielding_section = True
            continue
        
        if in_shielding_section:
            if line.startswith("ATOM"):
                atom_index += 1  # Increment atom_index to match coordinate index
                tensor_found = False  # Reset flag for each atom
                
            elif "total magnetic shielding:" in line:
                tensor = []
                tensor_found = True  # Mark tensor as found
                
                # Skip to the tensor lines
                next(line_iterator)  # Skip "Tensor :" line
                next(line_iterator)  # Skip one empty line or header line if present

                for _ in range(3):
                    tensor_line = next(line_iterator).strip()
                    tensor_row = [float(x) for x in tensor_line.split()]
                    tensor.append(tensor_row)

                # Flatten the tensor and reorder to match the header format
                flat_tensor = [
                    tensor[0][0], tensor[1][0], tensor[2][0],
                    tensor[0][1], tensor[1][1], tensor[2][1],
                    tensor[0][2], tensor[1][2], tensor[2][2]
                ]
                tensor_data.append([molecule_name, atom_index] + flat_tensor)
            
            # If no tensor was found for this atom
            if line.startswith("ATOM") and not tensor_found:
                atom_symbol = atom_dict.get(atom_index, 'Unknown')
                missing_tensors_atoms.append((atom_symbol, atom_index))
                zero_tensor = [0.0] * 9  # 3x3 zero tensor flattened
                tensor_data.append([molecule_name, atom_index] + zero_tensor)

    # Check if all atoms have tensors by comparing with atomic_coordinates
    for i, coord in enumerate(atomic_coordinates):
        if i >= len(tensor_data):  # If we ran out of tensor data
            atom_symbol = coord[2]  # Atom symbol from coordinates
            missing_tensors_atoms.append((atom_symbol, i))
            zero_tensor = [0.0] * 9  # 3x3 zero tensor flattened
            tensor_data.append([molecule_name, i] + zero_tensor)

    # Print summary of missing tensors
    total_atoms = len(atomic_coordinates)
    missing_tensors_count = len(missing_tensors_atoms)
    print(f"Out of {total_atoms} atoms, there are {missing_tensors_count} atoms where the shielding tensor was not calculated.")

    if missing_tensors_atoms:
        print(f"Warning: Magnetic shielding tensors not found for the following atoms (symbol, index):")
        for atom_symbol, index in missing_tensors_atoms:
            print(f"  Atom Symbol: {atom_symbol}, Index: {index}")

    columns = ["molecule_name", "atom_index", "XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    df = pd.DataFrame(tensor_data, columns=columns)
    return df

def extract_atomic_coordinates(xyz_path, structure_id):
    coordinates = []
    try:
        with open(xyz_path, 'r') as file:
            lines = file.readlines()
        
        # Ignore the first line (number of atoms) and the second line (lattice info)
        atom_lines = lines[2:]

        for atom_line in atom_lines:
            parts = atom_line.split()
            if len(parts) == 4:  # Ensure that there are exactly 4 columns
                atom, x, y, z = parts
                coordinates.append([f'Xe_TBA_{structure_id}', len(coordinates), atom, float(x), float(y), float(z)])

    except FileNotFoundError:
        print(f'Warning: {xyz_path} not found.')
        return None  # Return None to indicate that no coordinates were extracted

    return coordinates

def process_magnetic_shielding_tensors(file_path):
    df = pd.read_csv(file_path)
    # Identify rows where any tensor value is non-zero
    mask = (df[['XX', 'YX', 'ZX', 'XY', 'YY', 'ZY', 'XZ', 'YZ', 'ZZ']] != 0).any(axis=1)
    # Update atom_index to 1 where mask is True
    df.loc[mask, 'atom_index'] = 1
    df.to_csv(file_path, index=False)

def main():
    output_dir_tensors = 'dataset_schnet_shielding_tensors'
    output_dir_coordinates = 'dataset_schnet_atomic_coordinates'
    os.makedirs(output_dir_tensors, exist_ok=True)
    os.makedirs(output_dir_coordinates, exist_ok=True)

    all_tensors = []
    all_coordinates = []
    train_data = []

    # Define cluster_folders before using it
    cluster_folders = sorted([folder for folder in os.listdir() if folder.startswith('cluster_') and os.path.isdir(folder)], key=lambda x: int(re.findall(r'\d+', x)[0]))
    total_clusters = len(cluster_folders)  # Move this line here after defining cluster_folders

    missing_xyz_files_count = 0
    missing_tensors_count = 0

    for idx, folder in enumerate(cluster_folders):
        print(f'Processing folder {idx + 1} of {total_clusters}: {folder}')
        structure_id = re.findall(r'\d+', folder)[0]
        xyz_path = os.path.join(folder, f'coord_{structure_id}.xyz')
        
        if os.path.exists(xyz_path):
            # Extract atomic coordinates from the XYZ file
            atomic_coordinates = extract_atomic_coordinates(xyz_path, structure_id)

            if atomic_coordinates is None:
                missing_xyz_files_count += 1
                continue  # Skip this folder if coordinates could not be extracted

            mpshift_path = os.path.join(folder, 'mpshift.out')
            if os.path.exists(mpshift_path):
                # Extract shielding tensors, ensuring zero tensors for missing ones
                tensors = extract_shielding_tensors(mpshift_path, atomic_coordinates, structure_id)

                if tensors is None or tensors.empty:
                    missing_tensors_count += 1

                output_path_tensors = os.path.join(output_dir_tensors, f'Xe_TBA_{structure_id}.csv')
                output_path_coordinates = os.path.join(output_dir_coordinates, f'Xe_TBA_{structure_id}.csv')

                write_tensors_to_csv(output_path_tensors, tensors)
                write_coordinates_to_csv(output_path_coordinates, atomic_coordinates)

                all_tensors.append(tensors)
                all_coordinates.extend(atomic_coordinates)

                for i in range(len(atomic_coordinates)):
                    train_data.append([len(train_data), f'Xe_TBA_{structure_id}'])
            else:
                print(f'Warning: mpshift.out not found in {folder}')
                missing_tensors_count += 1
        else:
            print(f'Warning: {xyz_path} not found in {folder}')
            missing_xyz_files_count += 1

    # Write the concatenated output for tensors
    final_output_path_tensors = os.path.join(output_dir_tensors, 'magnetic_shielding_tensors.csv')
    print('Writing concatenated output for magnetic shielding tensors to magnetic_shielding_tensors.csv')
    if all_tensors:
        concatenated_tensors = pd.concat(all_tensors, ignore_index=True)
        concatenated_tensors.to_csv(final_output_path_tensors, index=False)
        # Post-process the concatenated file
        process_magnetic_shielding_tensors(final_output_path_tensors)

    # Write the concatenated output for coordinates
    final_output_path_coordinates = os.path.join(output_dir_coordinates, 'structures.csv')
    print('Writing concatenated output for atomic coordinates to structures.csv')
    with open(final_output_path_coordinates, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'])
        csvwriter.writerows(all_coordinates)

    # Write the train.csv file
    train_output_path = 'train.csv'
    print('Writing train.csv')
    with open(train_output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'molecule_name'])
        csvwriter.writerows(train_data)

    # Print summary
    print('\nProcessing complete!')
    print(f"Summary of processing:")
    print(f"  Total cluster folders processed: {total_clusters}")
    print(f"  Number of folders missing coord_<cluster_ID>.xyz files: {missing_xyz_files_count}")
    print(f"  Number of folders where magnetic shielding tensors were not calculated: {missing_tensors_count}")

if __name__ == "__main__":
    main()
