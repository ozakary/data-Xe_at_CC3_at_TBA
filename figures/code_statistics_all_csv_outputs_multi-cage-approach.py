import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ase.io import read
from ase.geometry import find_mic
from collections import deque

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LinearLocator

# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)

# Atomic masses (in amu)
ATOMIC_MASSES = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'Xe': 131.293}


def read_first_frame_data_with_clusters(filename):
    """
    Read the first frame to get atomic information including cluster IDs
    Returns: positions, elements, atom_ids, cluster_ids
    """
    with open(filename, 'r') as f:
        # Read number of atoms
        n_atoms = int(f.readline().strip())
        
        # Skip lattice line
        f.readline()
        
        positions = []
        elements = []
        atom_ids = []
        cluster_ids = []
        
        for _ in range(n_atoms):
            line = f.readline().strip().split()
            element = line[0]
            x, y, z = float(line[1]), float(line[2]), float(line[3])
            atom_id = int(line[4])
            cluster_id = int(line[5])  # Last column is cluster ID
            
            positions.append([x, y, z])
            elements.append(element)
            atom_ids.append(atom_id)
            cluster_ids.append(cluster_id)
    
    return np.array(positions), elements, atom_ids, cluster_ids


def identify_all_cc3_cages_from_clusters(filename):
    """
    Identify ALL CC3 cages using cluster IDs from the XYZ file
    Returns: List of dictionaries with cage info
    """
    print(f"Identifying CC3 cages using cluster IDs from first frame...")
    
    # Read first frame with cluster information
    positions, elements, atom_ids, cluster_ids = read_first_frame_data_with_clusters(filename)
    
    # Get cell information from ASE for PBC calculations
    first_frame = read(filename, index='0')  # Read only first frame
    cell = first_frame.get_cell()
    
    # Group atoms by cluster ID, only considering cage elements (H, C, N)
    cage_elements = ['H', 'C', 'N']
    clusters = {}
    
    for i, (elem, atom_id, cluster_id) in enumerate(zip(elements, atom_ids, cluster_ids)):
        if elem in cage_elements:  # Only consider cage atoms
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'atom_indices': [],
                    'atom_ids': [],
                    'elements': [],
                    'positions': []
                }
            
            clusters[cluster_id]['atom_indices'].append(i)
            clusters[cluster_id]['atom_ids'].append(atom_id)
            clusters[cluster_id]['elements'].append(elem)
            clusters[cluster_id]['positions'].append(positions[i])
    
    print(f"Found {len(clusters)} clusters containing cage atoms")
    
    # Convert to cage info format
    all_cages = []
    
    for cluster_id, cluster_data in clusters.items():
        cage_positions = np.array(cluster_data['positions'])
        cage_elements = cluster_data['elements']
        cage_size = len(cage_positions)
        
        # Calculate cage center of mass
        cage_masses = np.array([ATOMIC_MASSES[elem] for elem in cage_elements])
        cage_com = np.sum(cage_positions * cage_masses.reshape(-1, 1), axis=0) / np.sum(cage_masses)
        
        # Calculate composition
        composition = {}
        for elem in cage_elements:
            composition[elem] = composition.get(elem, 0) + 1
        
        cage_info = {
            'id': len(all_cages),  # Sequential ID for plotting
            'cluster_id': cluster_id,  # Original cluster ID from file
            'atom_ids': cluster_data['atom_ids'],
            'atom_indices': cluster_data['atom_indices'],
            'center_of_mass': cage_com,
            'composition': composition,
            'size': cage_size
        }
        
        all_cages.append(cage_info)
        
        print(f"  Cluster {cluster_id}: {cage_size} atoms - {composition}")
    
    # Sort cages by cluster ID for consistent ordering
    all_cages.sort(key=lambda x: x['cluster_id'])
    
    # Reassign sequential IDs after sorting
    for i, cage in enumerate(all_cages):
        cage['id'] = i
    
    print(f"\nTotal CC3 cages identified: {len(all_cages)}")
    print(f"Cluster IDs: {[cage['cluster_id'] for cage in all_cages]}")
    
    return all_cages


def find_nearest_xenon_atoms_to_cage(cage_com, positions, elements, atom_ids, cell, n_nearest=2):
    """
    Find the n nearest xenon atoms to a cage center of mass
    """
    distances = []
    
    for i, (pos, elem, atom_id) in enumerate(zip(positions, elements, atom_ids)):
        if elem == 'Xe':  # Only consider Xenon atoms
            # Calculate PBC-aware distance
            displacement = pos - cage_com
            corrected_displacement, distance = find_mic(displacement.reshape(1, 3), cell)
            distances.append((distance[0], i, atom_id))
    
    # Sort by distance and return n nearest
    distances.sort(key=lambda x: x[0])
    nearest = distances[:n_nearest]
    
    return [(atom_id, dist) for dist, idx, atom_id in nearest]


def calculate_cc3_center_of_mass(positions, masses):
    """
    Calculate center of mass of CC3 cage atoms
    """
    total_mass = np.sum(masses)
    com = np.sum(positions * masses.reshape(-1, 1), axis=0) / total_mass
    return com


def read_xenon_positions_from_xyz(filename, target_xenon_ids):
    """
    Read XYZ trajectory file and extract specific xenon atom data for target IDs
    Returns: {frame_idx: {atom_id: position_index}}
    """
    xenon_mapping = {}
    target_set = set(target_xenon_ids)
    
    with open(filename, 'r') as f:
        frame_idx = 0
        while True:
            # Read number of atoms
            line = f.readline()
            if not line:
                break
                
            n_atoms = int(line.strip())
            
            # Skip lattice line
            f.readline()
            
            # Read all atoms in this frame
            frame_xenons = {}
            for pos_idx in range(n_atoms):
                atom_line = f.readline().strip().split()
                element = atom_line[0]
                if element == 'Xe':
                    atom_id = int(atom_line[4])  # Last column is atom ID
                    if atom_id in target_set:
                        frame_xenons[atom_id] = pos_idx
            
            # Store mapping for this frame
            if frame_xenons:  # Only store frames that have our target xenons
                xenon_mapping[frame_idx] = frame_xenons
            
            frame_idx += 1
    
    return xenon_mapping


def unwrap_trajectory_pbc(positions, cell):
    """
    Unwrap trajectory by applying minimum image convention frame-by-frame using ASE
    """
    if len(positions) < 2:
        return positions.copy()
    
    unwrapped = [positions[0]]  # Start with first position
    
    for i in range(1, len(positions)):
        # Calculate displacement from previous frame
        displacement = positions[i] - positions[i-1]
        
        # Use ASE's find_mic function for proper minimum image convention
        corrected_displacement, _ = find_mic(displacement.reshape(1, 3), cell)
        corrected_displacement = corrected_displacement[0]
        
        # Add corrected displacement to get unwrapped position
        unwrapped.append(unwrapped[-1] + corrected_displacement)
    
    return np.array(unwrapped)


def calculate_displacement_for_all_cages(xyz_file, all_cages, dt):
    """
    Calculate displacement of nearest Xenon atoms relative to all CC3 cage centers of mass
    """
    print(f"Processing displacement for all {len(all_cages)} cages...")
    
    # Get first frame data to identify nearest xenons
    positions, elements, atom_ids, cluster_ids = read_first_frame_data_with_clusters(xyz_file)
    
    # Get cell information
    first_frame = read(xyz_file, index='0')
    cell = first_frame.get_cell()
    has_pbc = first_frame.get_pbc().any()
    
    # Collect all target xenon IDs and cage information
    all_target_xenon_ids = set()
    cage_xenon_mapping = {}  # {cage_id: {xe_id: cage_info}}
    
    print("  Finding nearest xenon atoms for each cage...")
    for cage_info in all_cages:
        # Find nearest xenon atoms to this cage
        nearest_xe = find_nearest_xenon_atoms_to_cage(
            cage_info['center_of_mass'], 
            positions, 
            elements, 
            atom_ids, 
            cell, 
            n_nearest=2
        )
        
        print(f"    Cage #{cage_info['id']} (cluster {cage_info['cluster_id']}): {[(xe_id, f'{dist:.2f} Å') for xe_id, dist in nearest_xe]}")
        
        # Store mapping of xenon to cage
        for xe_id, dist in nearest_xe:
            all_target_xenon_ids.add(xe_id)
            cage_xenon_mapping[xe_id] = {
                'cage_id': cage_info['id'],
                'cluster_id': cage_info['cluster_id'],
                'initial_distance': dist,
                'cage_atom_ids': cage_info['atom_ids'],
                'cage_atom_indices': cage_info['atom_indices']
            }
    
    print(f"  Total target xenon atoms across all cages: {len(all_target_xenon_ids)}")
    
    # Read ASE trajectory
    frames = read(xyz_file, index=':')
    
    # Read xenon mapping for all target atoms
    xenon_mapping = read_xenon_positions_from_xyz(xyz_file, all_target_xenon_ids)
    
    # Prepare cage data structures
    cage_data = {}
    for cage_info in all_cages:
        # Map CC3 atom IDs to indices and get masses
        cc3_indices = []
        cc3_masses = []
        
        for atom_id in cage_info['atom_ids']:
            try:
                idx = atom_ids.index(atom_id)
                cc3_indices.append(idx)
                cc3_masses.append(ATOMIC_MASSES[elements[idx]])
            except ValueError:
                print(f"    Warning: Could not find CC3 atom ID {atom_id} in first frame")
        
        cage_data[cage_info['id']] = {
            'indices': cc3_indices,
            'masses': np.array(cc3_masses),
            'com_trajectory': []
        }
    
    # Collect trajectories for all xenon atoms
    xe_data = {}
    for xe_id in all_target_xenon_ids:
        xe_data[xe_id] = {
            'positions': [], 
            'times': [], 
            'cage_info': cage_xenon_mapping[xe_id]
        }
    
    print("  Processing trajectory frames...")
    # Process trajectory frames
    for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
        time_ps = frame_idx * dt / 1e3  # Convert to ps
        
        # Calculate COM for each cage at this frame
        for cage_id, cage_data_item in cage_data.items():
            cc3_positions = frame.get_positions()[cage_data_item['indices']]
            cc3_com = calculate_cc3_center_of_mass(cc3_positions, cage_data_item['masses'])
            cage_data_item['com_trajectory'].append(cc3_com)
        
        # Get xenon positions for all target atoms
        if frame_idx in xenon_mapping:
            for xe_id in all_target_xenon_ids:
                if xe_id in xenon_mapping[frame_idx]:
                    pos_idx = xenon_mapping[frame_idx][xe_id]
                    xe_pos = frame.get_positions()[pos_idx]
                    xe_data[xe_id]['positions'].append(xe_pos)
                    xe_data[xe_id]['times'].append(time_ps)
    
    # Convert COM trajectories to numpy arrays and unwrap
    for cage_id, cage_data_item in cage_data.items():
        cage_data_item['com_trajectory'] = np.array(cage_data_item['com_trajectory'])
        if has_pbc:
            cage_data_item['com_unwrapped'] = unwrap_trajectory_pbc(cage_data_item['com_trajectory'], cell)
        else:
            cage_data_item['com_unwrapped'] = cage_data_item['com_trajectory']
    
    # Calculate displacements for each xenon atom relative to its nearest cage
    results = {}
    
    for xe_id, xe_data_item in xe_data.items():
        xe_positions = np.array(xe_data_item['positions'])
        xe_times = np.array(xe_data_item['times'])
        cage_id = xe_data_item['cage_info']['cage_id']
        
        if len(xe_positions) == 0:
            continue
        
        # Unwrap xenon trajectory
        if has_pbc:
            xe_unwrapped = unwrap_trajectory_pbc(xe_positions, cell)
        else:
            xe_unwrapped = xe_positions
        
        # Get the corresponding cage COM trajectory
        cage_com_unwrapped = cage_data[cage_id]['com_unwrapped']
        
        # Calculate PBC-aware displacement from cage COM for each frame
        displacement_magnitude = []
        
        for i in range(len(xe_unwrapped)):
            if i < len(cage_com_unwrapped):
                # Raw displacement
                raw_displacement = xe_unwrapped[i] - cage_com_unwrapped[i]
                
                # Apply minimum image convention to get correct displacement
                corrected_displacement, distance = find_mic(raw_displacement.reshape(1, 3), cell)
                
                # Use the PBC-corrected distance
                displacement_magnitude.append(distance[0])
        
        results[xe_id] = {
            'times': xe_times,
            'displacement_magnitude': np.array(displacement_magnitude),
            'cage_info': xe_data_item['cage_info']
        }
    
    print(f"  Successfully calculated displacements for {len(results)} xenon atoms")
    return results


def read_xyz_file_frames(filename, target_xenon_ids):
    """Read XYZ trajectory file and extract specific xenon atom data for target IDs"""
    xenon_data = {}  # {frame: {xenon_id: (x, y, z)}}
    target_set = set(target_xenon_ids)
    
    with open(filename, 'r') as f:
        frame = 0
        while True:
            # Read number of atoms
            line = f.readline()
            if not line:
                break
            
            n_atoms = int(line.strip())
            
            # Skip lattice line
            f.readline()
            
            frame_xenons = {}
            # Read atoms
            for _ in range(n_atoms):
                atom_line = f.readline().strip().split()
                element = atom_line[0]
                if element == 'Xe':
                    x, y, z = float(atom_line[1]), float(atom_line[2]), float(atom_line[3])
                    atom_id = int(atom_line[4])
                    if atom_id in target_set:
                        frame_xenons[atom_id] = (x, y, z)
            
            if frame_xenons:  # Only store frames that have our target xenons
                xenon_data[frame] = frame_xenons
            
            frame += 1
    
    return xenon_data


def match_coordinates(target_coords, csv_coords, tolerance=0.01):
    """Match specific target coordinates with CSV coordinates"""
    matches = {}
    for xenon_id, (x_xyz, y_xyz, z_xyz) in target_coords.items():
        for i, (x_csv, y_csv, z_csv) in enumerate(csv_coords):
            if (abs(x_xyz - x_csv) < tolerance and 
                abs(y_xyz - y_csv) < tolerance and 
                abs(z_xyz - z_csv) < tolerance):
                matches[xenon_id] = i
                break
    return matches

def save_site_data_to_csv(cc3_data_detailed, door_data_detailed, tba_data_detailed, output_prefix="xenon_site_data"):
    """
    Save detailed site data to separate CSV files for each site.
    
    Parameters:
    - cc3_data_detailed: list of dicts with detailed CC3 data
    - door_data_detailed: list of dicts with detailed Door data  
    - tba_data_detailed: list of dicts with detailed TBA data
    - output_prefix: prefix for output CSV filenames
    """
    print("Saving site data to CSV files...")
    
    def save_site_csv(data_list, site_name):
        if len(data_list) > 0:
            df = pd.DataFrame(data_list)
            filename = f"{output_prefix}_{site_name.lower()}.csv"
            df.to_csv(filename, index=False)
            print(f"  Saved {len(data_list)} {site_name} entries to {filename}")
            
            # Print summary statistics
            if 'delta_iso' in df.columns:
                print(f"    {site_name} δ_iso: {df['delta_iso'].mean():.3f} ± {df['delta_iso'].std():.3f} ppm")
                print(f"    {site_name} range: {df['delta_iso'].min():.3f} - {df['delta_iso'].max():.3f} ppm")
        else:
            print(f"  No {site_name} data to save")
    
    # Save each site's data
    save_site_csv(cc3_data_detailed, "Xe@CC3")
    save_site_csv(door_data_detailed, "Xe@Door") 
    save_site_csv(tba_data_detailed, "Xe@TBA")

def analyze_combined_xenon_statistics(xyz_file, csv_file, dt=1000, save_csv=True, output_prefix="xenon_site_data"):
    """
    Analyze xenon statistics for ALL CC3 cages combined
    
    Classification criteria based on displacement from CC3 center of mass:
    - Xe@CC3: 0 to 4.0 Å
    - Xe@Door: > 4.0 Å and ≤ 7.0 Å  
    - Xe@TBA: > 7.0 Å
    """
    print("=== COMBINED XENON STATISTICS ANALYSIS ===")
    
    # Step 1: Identify all CC3 cages
    all_cages = identify_all_cc3_cages_from_clusters(xyz_file)
    
    if len(all_cages) == 0:
        print("No CC3 cages found!")
        return None
    
    # Step 2: Calculate displacement data for all cages
    displacement_data = calculate_displacement_for_all_cages(xyz_file, all_cages, dt)
    
    if not displacement_data:
        print("No displacement data available")
        return None
    
    # Step 3: Apply smoothing to displacement data
    print("Applying smoothing to displacement data...")
    window_length = 81
    
    for xe_id in displacement_data:
        displacements = displacement_data[xe_id]['displacement_magnitude']
        times = displacement_data[xe_id]['times']
        
        # Adjust window length if needed
        window_length_adj = min(window_length, len(times) if len(times) % 2 == 1 else len(times) - 1)
        
        if len(times) > 4:
            smoothed = savgol_filter(displacements, window_length_adj, 3)
        else:
            smoothed = displacements
        
        displacement_data[xe_id]['displacement_smoothed'] = smoothed
    
    # Step 4: Read CSV data and match coordinates
    print("Reading CSV data and matching coordinates...")
    df = pd.read_csv(csv_file)
    xe_data = df[df['element'] == 'Xe']
    
    # Calculate delta_iso
    sigma_iso_ref = 5847.626
    xe_data = xe_data.copy()
    xe_data['delta_iso'] = sigma_iso_ref - xe_data['sigma_iso']
    
    # Get all target xenon IDs
    target_xenon_ids = list(displacement_data.keys())
    
    # Read XYZ trajectory file for coordinate matching
    xenon_xyz_data = read_xyz_file_frames(xyz_file, target_xenon_ids)
    
    # Create mapping between xenon atoms using XYZ coordinates
    print("Mapping xenon atoms between XYZ and CSV files...")
    xenon_mapping = {}  # {structure_id: {xenon_id: csv_row}}
    
    available_structures = set(xe_data['structure_id'].unique())
    matched_structures = 0
    
    for structure_id in tqdm(sorted(xenon_xyz_data.keys()), desc="Coordinate matching"):
        if structure_id in available_structures:
            # Get XYZ coordinates for target xenons in this frame
            xyz_coords = xenon_xyz_data[structure_id]
            
            # Get CSV data for this structure
            structure_data = xe_data[xe_data['structure_id'] == structure_id]
            csv_coords = [(row['x'], row['y'], row['z']) for _, row in structure_data.iterrows()]
            
            # Match coordinates
            matches = match_coordinates(xyz_coords, csv_coords)
            
            if matches:  # If we matched at least some xenons
                xenon_mapping[structure_id] = {}
                for xenon_id, csv_index in matches.items():
                    xenon_mapping[structure_id][xenon_id] = structure_data.iloc[csv_index]
                matched_structures += 1
    
    print(f"Successfully matched {matched_structures} structures")
    
    # Step 5: Classify xenon atoms and collect statistics
    print("Classifying xenon atoms and collecting statistics...")
    
    # Collect data for CC3, Door, and TBA classifications
    cc3_delta_iso = []
    door_delta_iso = []
    tba_delta_iso = []
    
    # Detailed data for CSV export
    cc3_data_detailed = []
    door_data_detailed = []
    tba_data_detailed = []
    
    classification_log = []
    cage_classifications = {}  # Track classifications per cage
    
    def classify_xenon(displacement):
        """Classify xenon based on displacement from CC3 center of mass"""
        if displacement <= 4.0:
            return "CC3"
        elif displacement <= 7.0:
            return "Door"
        else:
            return "TBA"
    
    # Process each xenon atom's trajectory
    for xe_id in displacement_data:
        xe_data_item = displacement_data[xe_id]
        times = xe_data_item['times']
        displacements = xe_data_item['displacement_smoothed']
        cage_id = xe_data_item['cage_info']['cage_id']
        cluster_id = xe_data_item['cage_info']['cluster_id']
        
        # Initialize cage classification tracking
        if cage_id not in cage_classifications:
            cage_classifications[cage_id] = {'CC3': 0, 'Door': 0, 'TBA': 0, 'cluster_id': cluster_id}
        
        for i, time_ps in enumerate(times):
            frame_idx = int(time_ps * 1e3 / dt)  # Convert time back to frame index
            
            if frame_idx in xenon_mapping and i < len(displacements):
                mapping = xenon_mapping[frame_idx]
                
                if xe_id in mapping:
                    displacement = displacements[i]
                    delta_iso = mapping[xe_id]['delta_iso']
                    
                    # Classify xenon atom
                    site = classify_xenon(displacement)
                    

                    # Common data for all sites
                    common_data = {
                        'frame': frame_idx,
                        'time_ps': time_ps,
                        'xenon_id': xe_id,
                        'cage_id': cage_id,
                        'cluster_id': cluster_id,
                        'displacement_from_cage': displacement,
                        'delta_iso': delta_iso,
                        'sigma_iso': mapping[xe_id]['sigma_iso'],
                        'x': mapping[xe_id]['x'],
                        'y': mapping[xe_id]['y'],
                        'z': mapping[xe_id]['z'],
                        'structure_id': mapping[xe_id]['structure_id']
                    }

                    # Add to appropriate lists
                    if site == "CC3":
                        cc3_delta_iso.append(delta_iso)
                        cc3_data_detailed.append(common_data.copy())
                    elif site == "Door":
                        door_delta_iso.append(delta_iso)
                        door_data_detailed.append(common_data.copy())
                    else:  # TBA
                        tba_delta_iso.append(delta_iso)
                        tba_data_detailed.append(common_data.copy())
                    
                    # Track per cage
                    cage_classifications[cage_id][site] += 1
                    
                    # Log first few classifications
                    if len(classification_log) < 20:
                        classification_log.append(
                            f"Frame {frame_idx}: Xe{xe_id}@{site} ({displacement:.2f}Å) [Cage {cage_id}/Cluster {cluster_id}]"
                        )
    
    # Step 6: Calculate statistics
    print("Calculating combined statistics...")
    
    # Step 6: Save to CSV files if requested
    if save_csv:
        save_site_data_to_csv(cc3_data_detailed, door_data_detailed, tba_data_detailed, output_prefix)
    
    cc3_delta_iso = np.array(cc3_delta_iso)
    door_delta_iso = np.array(door_delta_iso)
    tba_delta_iso = np.array(tba_delta_iso)
    
    def calculate_stats(data, name):
        if len(data) > 0:
            stats = {
                'mean': np.mean(data),
                'min': np.min(data),
                'max': np.max(data),
                'std': np.std(data, ddof=1),  # Sample standard deviation
                'sem': np.std(data, ddof=1) / np.sqrt(len(data)),  # Standard error of mean
                'n_snapshots': len(data)
            }
            print(f"\n{name} Statistics (All Cages Combined):")
            print(f"  Number of snapshots: {stats['n_snapshots']}")
            print(f"  Mean δ_iso: {stats['mean']:.3f} ± {stats['sem']:.3f} ppm")
            print(f"  Min δ_iso: {stats['min']:.3f} ppm")
            print(f"  Max δ_iso: {stats['max']:.3f} ppm")
            print(f"  Std δ_iso: {stats['std']:.3f} ppm")
            return stats
        else:
            print(f"\n{name} Statistics: No data available")
            return None
    
    cc3_stats = calculate_stats(cc3_delta_iso, "Xe@CC3")
    door_stats = calculate_stats(door_delta_iso, "Xe@Door")
    tba_stats = calculate_stats(tba_delta_iso, "Xe@TBA")
    
    return {
        'cc3_stats': cc3_stats,
        'door_stats': door_stats,
        'tba_stats': tba_stats,
        'cc3_data': cc3_delta_iso,
        'door_data': door_delta_iso,
        'tba_data': tba_delta_iso,
        'cc3_data_detailed': cc3_data_detailed,
        'door_data_detailed': door_data_detailed,
        'tba_data_detailed': tba_data_detailed,
        'classification_log': classification_log,
        'cage_classifications': cage_classifications,
        'n_cages': len(all_cages),
        'displacement_data': displacement_data
    }


def plot_combined_statistics(results, filename_1, filename_2):
    """
    Create combined summary plot for all cages
    """
    if not results:
        print("No results to plot")
        return
    
    cc3_data = results['cc3_data']
    door_data = results['door_data']
    tba_data = results['tba_data']
    n_cages = results['n_cages']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    
    # Define colors for the three sites
    colors = {'CC3': '#2A9D8F', 'Door': '#E76F51', 'TBA': '#F4A261'}
    
    # Histogram of delta_iso values
    if len(cc3_data) > 0:
        ax1.hist(cc3_data, bins=30, alpha=0.7, color=colors['CC3'], label='CC3', density=True)
    if len(tba_data) > 0:
        ax1.hist(tba_data, bins=30, alpha=0.7, color=colors['TBA'], label='TBA', density=True)
    if len(door_data) > 0:
        ax1.hist(door_data, bins=30, alpha=0.7, color=colors['Door'], label='Door', density=True)

    
    ax1.set_xlabel(r'$\delta_{\text{iso}}$ / ppm')
    ax1.set_ylabel('Density')
    ax1.set_ylim(0, 0.015)
    ax1.set_xlim(-100, 700)
    ax1.legend(frameon=False)
    
    ax1.xaxis.set_major_locator(LinearLocator(numticks=5))
    
#    ax1.set_title(f'Chemical Shift Distribution\n({n_cages} CC3 Cages Combined)')
    
    # Box plot comparison
    data_for_box = []
    labels_for_box = []
    colors_for_box = []
    
    if len(cc3_data) > 0:
        data_for_box.append(cc3_data)
        labels_for_box.append('CC3')
        colors_for_box.append(colors['CC3'])
      
    if len(tba_data) > 0:
        data_for_box.append(tba_data)
        labels_for_box.append('TBA')
        colors_for_box.append(colors['TBA']) 

    if len(door_data) > 0:
        data_for_box.append(door_data)
        labels_for_box.append('Door')
        colors_for_box.append(colors['Door'])
    
    if data_for_box:
        bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel(r'$\delta_{\text{iso}}$ / ppm')
    ax2.set_ylim(-100, 700)
    
    ax2.yaxis.set_major_locator(LinearLocator(numticks=5))
    
    ax2.tick_params(axis='x', rotation=45)
#    ax2.set_title(f'Statistical Comparison\n({n_cages} CC3 Cages Combined)')
    
    plt.tight_layout()
    plt.savefig(filename_1, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(filename_2, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Combined plot saved as {filename_1} and {filename_2}")

def find_nearest_cage_to_xenon(xe_position, cage_coms, cell):
    """
    Find the nearest cage to a xenon atom position
    Returns: (cage_id, distance)
    """
    min_distance = float('inf')
    nearest_cage_id = None
    
    for cage_id, cage_com in cage_coms.items():
        # Calculate PBC-aware distance
        displacement = xe_position - cage_com
        corrected_displacement, distance = find_mic(displacement.reshape(1, 3), cell)
        distance = distance[0]
        
        if distance < min_distance:
            min_distance = distance
            nearest_cage_id = cage_id
    
    return nearest_cage_id, min_distance


def analyze_combined_xenon_statistics_multicage(xyz_file, csv_file, dt=1000, save_csv=True, output_prefix="xenon_site_data"):
    """
    Modified version that considers ALL cages for each xenon classification
    """
    print("=== COMBINED XENON STATISTICS ANALYSIS (MULTI-CAGE) ===")
    
    # Step 1: Identify all CC3 cages
    all_cages = identify_all_cc3_cages_from_clusters(xyz_file)
    
    if len(all_cages) == 0:
        print("No CC3 cages found!")
        return None
    
    # Step 2: Calculate displacement data for all cages (needed for trajectory data)
    displacement_data = calculate_displacement_for_all_cages(xyz_file, all_cages, dt)
    
    if not displacement_data:
        print("No displacement data available")
        return None
    
    # Step 3: Get cell information and read trajectory
    first_frame = read(xyz_file, index='0')
    cell = first_frame.get_cell()
    frames = read(xyz_file, index=':')
    
    # Step 4: Prepare cage data structures (same as before)
    positions, elements, atom_ids, cluster_ids = read_first_frame_data_with_clusters(xyz_file)
    
    cage_data = {}
    for cage_info in all_cages:
        cc3_indices = []
        cc3_masses = []
        
        for atom_id in cage_info['atom_ids']:
            try:
                idx = atom_ids.index(atom_id)
                cc3_indices.append(idx)
                cc3_masses.append(ATOMIC_MASSES[elements[idx]])
            except ValueError:
                print(f"Warning: Could not find CC3 atom ID {atom_id} in first frame")
        
        cage_data[cage_info['id']] = {
            'indices': cc3_indices,
            'masses': np.array(cc3_masses),
            'com_trajectory': []
        }
    
    # Step 5: Calculate COM trajectories for all cages
    print("Calculating COM trajectories for all cages...")
    for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
        for cage_id, cage_data_item in cage_data.items():
            cc3_positions = frame.get_positions()[cage_data_item['indices']]
            cc3_com = calculate_cc3_center_of_mass(cc3_positions, cage_data_item['masses'])
            cage_data_item['com_trajectory'].append(cc3_com)
    
    # Convert to numpy arrays and unwrap
    has_pbc = first_frame.get_pbc().any()
    for cage_id, cage_data_item in cage_data.items():
        cage_data_item['com_trajectory'] = np.array(cage_data_item['com_trajectory'])
        if has_pbc:
            cage_data_item['com_unwrapped'] = unwrap_trajectory_pbc(cage_data_item['com_trajectory'], cell)
        else:
            cage_data_item['com_unwrapped'] = cage_data_item['com_trajectory']
    
    # Step 6: Get all xenon trajectories
    target_xenon_ids = list(displacement_data.keys())
    xenon_mapping = read_xenon_positions_from_xyz(xyz_file, target_xenon_ids)
    
    # Collect xenon trajectories
    xe_trajectories = {}
    for xe_id in target_xenon_ids:
        xe_trajectories[xe_id] = {'positions': [], 'times': []}
    
    for frame_idx, frame in enumerate(frames):
        time_ps = frame_idx * dt / 1e3
        
        if frame_idx in xenon_mapping:
            for xe_id in target_xenon_ids:
                if xe_id in xenon_mapping[frame_idx]:
                    pos_idx = xenon_mapping[frame_idx][xe_id]
                    xe_pos = frame.get_positions()[pos_idx]
                    xe_trajectories[xe_id]['positions'].append(xe_pos)
                    xe_trajectories[xe_id]['times'].append(time_ps)
    
    # Unwrap xenon trajectories
    for xe_id in xe_trajectories:
        xe_positions = np.array(xe_trajectories[xe_id]['positions'])
        if len(xe_positions) > 0:
            if has_pbc:
                xe_trajectories[xe_id]['unwrapped'] = unwrap_trajectory_pbc(xe_positions, cell)
            else:
                xe_trajectories[xe_id]['unwrapped'] = xe_positions
        else:
            xe_trajectories[xe_id]['unwrapped'] = np.array([])
    
    # Step 7: Read CSV data and match coordinates
    print("Reading CSV data and matching coordinates...")
    df = pd.read_csv(csv_file)
    xe_data = df[df['element'] == 'Xe']
    
    sigma_iso_ref = 5847.626
    xe_data = xe_data.copy()
    xe_data['delta_iso'] = sigma_iso_ref - xe_data['sigma_iso']
    
    xenon_xyz_data = read_xyz_file_frames(xyz_file, target_xenon_ids)
    
    xenon_mapping_csv = {}
    available_structures = set(xe_data['structure_id'].unique())
    matched_structures = 0
    
    for structure_id in tqdm(sorted(xenon_xyz_data.keys()), desc="Coordinate matching"):
        if structure_id in available_structures:
            xyz_coords = xenon_xyz_data[structure_id]
            structure_data = xe_data[xe_data['structure_id'] == structure_id]
            csv_coords = [(row['x'], row['y'], row['z']) for _, row in structure_data.iterrows()]
            
            matches = match_coordinates(xyz_coords, csv_coords)
            
            if matches:
                xenon_mapping_csv[structure_id] = {}
                for xenon_id, csv_index in matches.items():
                    xenon_mapping_csv[structure_id][xenon_id] = structure_data.iloc[csv_index]
                matched_structures += 1
    
    print(f"Successfully matched {matched_structures} structures")
    
    # Step 8: NEW MULTI-CAGE CLASSIFICATION
    print("Classifying xenon atoms using multi-cage approach...")
    
    cc3_data_detailed = []
    door_data_detailed = []
    tba_data_detailed = []
    
    def classify_xenon_multicage(distance_to_nearest_cage):
        """Classify based on distance to nearest cage"""
        if distance_to_nearest_cage <= 4.0:
            return "CC3"
        elif distance_to_nearest_cage <= 7.0:
            return "Door"
        else:
            return "TBA"
    
    # Process each xenon trajectory
    total_processing_count = 0
    for xe_id in target_xenon_ids:
        xe_traj = xe_trajectories[xe_id]
        total_processing_count += len(xe_traj['times'])
    
    print(f"Processing {len(target_xenon_ids)} xenon atoms across {total_processing_count} total frame positions...")
    
    processed_count = 0
    for xe_id in tqdm(target_xenon_ids, desc="Processing xenon atoms"):
        xe_traj = xe_trajectories[xe_id]
        times = xe_traj['times']
        xe_positions = xe_traj['unwrapped']
        
        if len(xe_positions) == 0:
            continue
        
        for i, time_ps in enumerate(times):
            frame_idx = int(time_ps * 1e3 / dt)
            
            if frame_idx in xenon_mapping_csv and i < len(xe_positions):
                mapping = xenon_mapping_csv[frame_idx]
                
                if xe_id in mapping:
                    xe_pos = xe_positions[i]
                    
                    # Get all cage COMs for this frame
                    cage_coms = {}
                    for cage_id in cage_data:
                        if i < len(cage_data[cage_id]['com_unwrapped']):
                            cage_coms[cage_id] = cage_data[cage_id]['com_unwrapped'][i]
                    
                    # Find nearest cage to this xenon
                    nearest_cage_id, distance_to_nearest = find_nearest_cage_to_xenon(
                        xe_pos, cage_coms, cell
                    )
                    
                    # Classify based on distance to nearest cage
                    site = classify_xenon_multicage(distance_to_nearest)
                    delta_iso = mapping[xe_id]['delta_iso']
                    
                    # Get cluster ID of nearest cage
                    nearest_cluster_id = all_cages[nearest_cage_id]['cluster_id']
                    
                    common_data = {
                        'frame': frame_idx,
                        'time_ps': time_ps,
                        'xenon_id': xe_id,
                        'nearest_cage_id': nearest_cage_id,
                        'nearest_cluster_id': nearest_cluster_id,
                        'distance_to_nearest_cage': distance_to_nearest,
                        'delta_iso': delta_iso,
                        'sigma_iso': mapping[xe_id]['sigma_iso'],
                        'x': mapping[xe_id]['x'],
                        'y': mapping[xe_id]['y'],
                        'z': mapping[xe_id]['z'],
                        'structure_id': mapping[xe_id]['structure_id']
                    }
                    
                    if site == "CC3":
                        cc3_data_detailed.append(common_data.copy())
                    elif site == "Door":
                        door_data_detailed.append(common_data.copy())
                    else:  # TBA
                        tba_data_detailed.append(common_data.copy())
            
            processed_count += 1
    
    # Step 9: Calculate statistics and save CSV
    if save_csv:
        save_site_data_to_csv(cc3_data_detailed, door_data_detailed, tba_data_detailed, output_prefix)
    
    # Extract delta_iso arrays for statistics
    cc3_delta_iso = np.array([d['delta_iso'] for d in cc3_data_detailed])
    door_delta_iso = np.array([d['delta_iso'] for d in door_data_detailed])
    tba_delta_iso = np.array([d['delta_iso'] for d in tba_data_detailed])
    
    def calculate_stats(data, name):
        if len(data) > 0:
            stats = {
                'mean': np.mean(data),
                'min': np.min(data),
                'max': np.max(data),
                'std': np.std(data, ddof=1),
                'sem': np.std(data, ddof=1) / np.sqrt(len(data)),
                'n_snapshots': len(data)
            }
            print(f"\n{name} Statistics (Multi-Cage Classification):")
            print(f"  Number of snapshots: {stats['n_snapshots']}")
            print(f"  Mean δ_iso: {stats['mean']:.3f} ± {stats['sem']:.3f} ppm")
            print(f"  Min δ_iso: {stats['min']:.3f} ppm")
            print(f"  Max δ_iso: {stats['max']:.3f} ppm")
            print(f"  Std δ_iso: {stats['std']:.3f} ppm")
            return stats
        else:
            print(f"\n{name} Statistics: No data available")
            return None
    
    cc3_stats = calculate_stats(cc3_delta_iso, "Xe@CC3")
    door_stats = calculate_stats(door_delta_iso, "Xe@Door")
    tba_stats = calculate_stats(tba_delta_iso, "Xe@TBA")
    
    return {
        'cc3_stats': cc3_stats,
        'door_stats': door_stats,
        'tba_stats': tba_stats,
        'cc3_data': cc3_delta_iso,
        'door_data': door_delta_iso,
        'tba_data': tba_delta_iso,
        'cc3_data_detailed': cc3_data_detailed,
        'door_data_detailed': door_data_detailed,
        'tba_data_detailed': tba_data_detailed,
        'n_cages': len(all_cages)
    }


# Modified main execution
if __name__ == "__main__":
    xyz_file = '../displacement/2xe_tba_traj_sampled-100_clusters_without_init_str_out.xyz'
    csv_file = './combined_predictions.csv'
    dt = 1000  # Time step in femtoseconds
    
    try:
        # CSV output settings
        save_to_csv = True
        csv_output_prefix = "xenon_site_data_multicage"
        
        # Run the MULTI-CAGE analysis
        results = analyze_combined_xenon_statistics_multicage(
            xyz_file, 
            csv_file, 
            dt, 
            save_csv=save_to_csv, 
            output_prefix=csv_output_prefix
        )
        
        if results:
            # Create combined plot
            filename_1 = 'xenon_statistics_multicage.svg'
            filename_2 = 'xenon_statistics_multicage.png'
            
            plot_combined_statistics(results, filename_1, filename_2)
            
            # Print comparison summary
            print(f"\n{'='*70}")
            print("MULTI-CAGE ANALYSIS COMPLETED!")
            print(f"{'='*70}")
            print(f"Each xenon classified based on distance to NEAREST cage (out of {results['n_cages']} cages)")
            
            if save_to_csv:
                print(f"\nCSV Files Generated:")
                print(f"  {csv_output_prefix}_xe@cc3.csv - Contains {len(results['cc3_data_detailed'])} CC3 entries")
                print(f"  {csv_output_prefix}_xe@door.csv - Contains {len(results['door_data_detailed'])} Door entries")
                print(f"  {csv_output_prefix}_xe@tba.csv - Contains {len(results['tba_data_detailed'])} TBA entries")
            
            # Print statistics
            print(f"\nMulti-Cage Classification Statistics:")
            if results['cc3_stats']:
                print(f"Xe@CC3:  {results['cc3_stats']['mean']:.2f} ± {results['cc3_stats']['sem']:.2f} ppm (n={results['cc3_stats']['n_snapshots']})")
            if results['door_stats']:
                print(f"Xe@Door: {results['door_stats']['mean']:.2f} ± {results['door_stats']['sem']:.2f} ppm (n={results['door_stats']['n_snapshots']})")
            if results['tba_stats']:
                print(f"Xe@TBA:  {results['tba_stats']['mean']:.2f} ± {results['tba_stats']['sem']:.2f} ppm (n={results['tba_stats']['n_snapshots']})")
        
        else:
            print("Multi-cage analysis failed!")
            
    except Exception as e:
        print(f"Error during multi-cage analysis: {e}")
        import traceback
        traceback.print_exc()
