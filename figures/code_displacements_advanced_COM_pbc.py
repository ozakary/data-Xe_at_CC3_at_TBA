import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ase.io import read
from ase.geometry import find_mic
from collections import deque
from matplotlib import patches
import matplotlib.patches as mpatches


# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)


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
            cluster_id = int(line[5])  # The cluster ID from the last column
            
            positions.append([x, y, z])
            elements.append(element)
            atom_ids.append(atom_id)
            cluster_ids.append(cluster_id)
    
    return np.array(positions), elements, atom_ids, cluster_ids


def identify_all_cc3_cages(filename, expected_cage_size=168, size_tolerance=10):
    """
    Identify ALL CC3 cages using cluster IDs from the XYZ file
    Returns: List of dictionaries with cage info
    """
    print(f"Identifying all CC3 cages using cluster IDs from first frame...")
    
    # Read first frame with cluster information
    positions, elements, atom_ids, cluster_ids = read_first_frame_data_with_clusters(filename)
    
    # Get cell information from ASE for PBC calculations
    first_frame = read(filename, index='0')
    cell = first_frame.get_cell()
    
    # Group atoms by cluster ID
    clusters = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    
    print(f"Found {len(clusters)} unique clusters in the system")
    
    # Find CC3 cages by filtering clusters with appropriate size and composition
    all_cages = []
    cage_elements = {'H', 'C', 'N'}  # Expected elements in CC3 cages
    
    for cluster_id, atom_indices in clusters.items():
        cluster_size = len(atom_indices)
        
        # Check if cluster size matches expected CC3 cage size
        if abs(cluster_size - expected_cage_size) <= size_tolerance:
            
            # Check if cluster contains only cage elements (H, C, N)
            cluster_elements = set(elements[i] for i in atom_indices)
            if cluster_elements.issubset(cage_elements):
                
                # Calculate cluster composition
                composition = {}
                for i in atom_indices:
                    elem = elements[i]
                    composition[elem] = composition.get(elem, 0) + 1
                
                # Calculate center of mass
                atomic_masses = {'H': 1.008, 'C': 12.011, 'N': 14.007}
                cluster_positions = positions[atom_indices]
                cluster_masses = np.array([atomic_masses[elements[i]] for i in atom_indices])
                cluster_com = np.sum(cluster_positions * cluster_masses.reshape(-1, 1), axis=0) / np.sum(cluster_masses)
                
                # Convert to atom IDs
                cage_atom_ids = [atom_ids[i] for i in atom_indices]
                
                cage_info = {
                    'id': len(all_cages),
                    'cluster_id': cluster_id,
                    'atom_ids': cage_atom_ids,
                    'atom_indices': atom_indices,
                    'center_of_mass': cluster_com,
                    'composition': composition,
                    'size': cluster_size
                }
                
                all_cages.append(cage_info)
                
                print(f"Found CC3 cage #{len(all_cages)} (Cluster ID: {cluster_id}) with {cluster_size} atoms")
                print(f"  Composition: {composition}")
                print(f"  Center of mass: {cluster_com}")
    
    print(f"\nTotal CC3 cages identified: {len(all_cages)}")
    return all_cages


def find_nearest_xenon_atoms(cage_com, xenon_positions, xenon_ids, cell, n_nearest=2):
    """
    Find the n nearest xenon atoms to a cage center of mass
    """
    distances = []
    
    for i, (xe_pos, xe_id) in enumerate(zip(xenon_positions, xenon_ids)):
        # Calculate PBC-aware distance
        displacement = xe_pos - cage_com
        corrected_displacement, distance = find_mic(displacement.reshape(1, 3), cell)
        distances.append((distance[0], i, xe_id))
    
    # Sort by distance and return n nearest
    distances.sort(key=lambda x: x[0])
    nearest = distances[:n_nearest]
    
    return [(xe_id, dist) for dist, idx, xe_id in nearest]


def calculate_cc3_center_of_mass(positions, masses):
    """
    Calculate center of mass of CC3 cage atoms
    """
    total_mass = np.sum(masses)
    com = np.sum(positions * masses.reshape(-1, 1), axis=0) / total_mass
    return com


def read_xenon_positions_from_xyz(filename):
    """
    Read XYZ trajectory file to map xenon atom IDs to their position indices in each frame
    Returns: {frame_idx: {atom_id: position_index}}
    """
    xenon_mapping = {}
    
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
                    atom_id = int(atom_line[4])  # Fifth column is atom ID
                    frame_xenons[atom_id] = pos_idx
            
            # Store mapping for this frame
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


def calculate_displacement_for_cage(xyz_file, cage_info, dt):
    """
    Calculate displacement of nearest Xenon atoms relative to a specific CC3 cage center of mass
    """
    print(f"Processing cage #{cage_info['id']} (Cluster ID: {cage_info['cluster_id']}) with {cage_info['size']} atoms...")
    
    # Read XYZ trajectory file with ASE
    frames = read(xyz_file, index=':')
    
    # Read xenon positions from XYZ file
    xenon_mapping = read_xenon_positions_from_xyz(xyz_file)
    
    # Get all xenon atom IDs and positions from first frame
    positions_0, elements_0, atom_ids_0, cluster_ids_0 = read_first_frame_data_with_clusters(xyz_file)
    xenon_positions = []
    xenon_ids = []
    for i, (elem, atom_id) in enumerate(zip(elements_0, atom_ids_0)):
        if elem == 'Xe':
            xenon_positions.append(positions_0[i])
            xenon_ids.append(atom_id)
    
    # Get cell and atomic masses
    cell = frames[0].get_cell()
    has_pbc = frames[0].get_pbc().any()
    atomic_masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'Xe': 131.293}
    
    # Find nearest xenon atoms to this cage
    nearest_xe = find_nearest_xenon_atoms(
        cage_info['center_of_mass'], 
        xenon_positions, 
        xenon_ids, 
        cell, 
        n_nearest=2
    )
    
    print(f"  Nearest Xenon atoms: {[(xe_id, f'{dist:.2f} Å') for xe_id, dist in nearest_xe]}")
    
    # Map CC3 atom IDs to indices and get masses
    cc3_indices = []
    cc3_masses = []
    
    for atom_id in cage_info['atom_ids']:
        try:
            idx = atom_ids_0.index(atom_id)
            cc3_indices.append(idx)
            cc3_masses.append(atomic_masses[elements_0[idx]])
        except ValueError:
            print(f"Warning: Could not find CC3 atom ID {atom_id} in first frame")
    
    cc3_masses = np.array(cc3_masses)
    
    # Collect trajectories
    xe_data = {}
    cc3_com_trajectory = []
    times = []
    
    # Initialize data for nearest xenon atoms
    for i, (xe_id, dist) in enumerate(nearest_xe):
        xe_data[f'xe{i+1}'] = {'positions': [], 'times': [], 'id': xe_id, 'initial_distance': dist}
    
    print("  Processing trajectory frames...")
    for frame_idx, frame in enumerate(tqdm(frames, desc=f"Cage {cage_info['id']}")):
        time_ps = frame_idx * dt / 1e3  # Convert to ps
        times.append(time_ps)
        
        # Get CC3 center of mass for this frame
        cc3_positions = frame.get_positions()[cc3_indices]
        cc3_com = calculate_cc3_center_of_mass(cc3_positions, cc3_masses)
        cc3_com_trajectory.append(cc3_com)
        
        # Get xenon positions for nearest atoms
        if frame_idx in xenon_mapping:
            for i, (xe_id, _) in enumerate(nearest_xe):
                if xe_id in xenon_mapping[frame_idx]:
                    pos_idx = xenon_mapping[frame_idx][xe_id]
                    xe_pos = frame.get_positions()[pos_idx]
                    xe_data[f'xe{i+1}']['positions'].append(xe_pos)
                    xe_data[f'xe{i+1}']['times'].append(time_ps)
    
    cc3_com_trajectory = np.array(cc3_com_trajectory)
    times = np.array(times)
    
    # Unwrap CC3 COM trajectory
    if has_pbc:
        cc3_com_unwrapped = unwrap_trajectory_pbc(cc3_com_trajectory, cell)
    else:
        cc3_com_unwrapped = cc3_com_trajectory
    
    # Calculate displacements for each xenon atom
    results = {}
    
    for key in xe_data:
        xe_positions = np.array(xe_data[key]['positions'])
        xe_times = np.array(xe_data[key]['times'])
        
        if len(xe_positions) == 0:
            continue
        
        # Unwrap xenon trajectory
        if has_pbc:
            xe_unwrapped = unwrap_trajectory_pbc(xe_positions, cell)
        else:
            xe_unwrapped = xe_positions
        
        # Calculate PBC-aware displacement from CC3 COM for each frame
        displacement_magnitude = []
        
        for i in range(len(xe_unwrapped)):
            if i < len(cc3_com_unwrapped):
                # Raw displacement
                raw_displacement = xe_unwrapped[i] - cc3_com_unwrapped[i]
                
                # Apply minimum image convention to get correct displacement
                corrected_displacement, distance = find_mic(raw_displacement.reshape(1, 3), cell)
                
                # Use the PBC-corrected distance
                displacement_magnitude.append(distance[0])
        
        results[key] = {
            'time': xe_times,
            'displacement_magnitude': np.array(displacement_magnitude),
            'id': xe_data[key]['id'],
            'initial_distance': xe_data[key]['initial_distance']
        }
    
    return results


def plot_displacement_data_for_cage(data, cage_info, filename_1, filename_2):
    """
    Plot displacement data for a specific cage with Savitzky-Golay smoothing
    """
    plt.figure(figsize=(9, 4.5))
    
    colors = ['#2A9D8F', '#E76F51', '#F4A261', '#264653']  # Colors for up to 4 xenon atoms
    
    # Define regions with colored patterns only
    regions = [
        {'range': (0, 4), 'pattern_color': 'grey', 'pattern': '...', 'label': 'CC3'},
        {'range': (4, 7), 'pattern_color': 'grey', 'pattern': '///', 'label': 'Door'},
        {'range': (7, 30), 'pattern_color': 'grey', 'pattern': '|||', 'label': 'TBA'}
    ]

    # Add pattern-only regions
    for region in regions:
        plt.axhspan(region['range'][0], region['range'][1], 
                    facecolor='none',  # No background color
                    edgecolor=region['pattern_color'],  # Colored pattern
                    hatch=region['pattern'], 
                    linewidth=0, 
                    alpha=0.6,  # Controls pattern opacity
                    zorder=0)
                    
    # Create custom legend patches for regions
    region_patches = []
    for region in regions:
        patch = mpatches.Patch(facecolor='white',  # White background for legend visibility
                              edgecolor=region['pattern_color'],
                              alpha=0.8,  # Higher alpha for legend visibility
                              hatch=region['pattern'], 
                              label=region['label'])
        region_patches.append(patch)
    
    for i, key in enumerate(sorted(data.keys())):
        if key.startswith('xe'):
            xe_data = data[key]['displacement_magnitude']
            time_points = data[key]['time']
            xe_id = data[key]['id']
            initial_dist = data[key]['initial_distance']
            
            color = colors[i % len(colors)]
            
            # Apply Savitzky-Golay filter
            window_length = 81
            window_length = min(window_length, len(time_points) if len(time_points) % 2 == 1 else len(time_points) - 1)
            
            if len(time_points) > 4:
                xe_smooth = savgol_filter(xe_data, window_length, 3)
                
                # Raw data with high transparency
                plt.plot(time_points, xe_data, '-', alpha=0.2, color=color)
                
                # Smoothed data
                plt.plot(time_points, xe_smooth, '-', color=color, linewidth=2, 
                        label=f'Xe@CC3@t$_{0}$' if key == 'xe1' else f'Xe@Door@t$_{0}$')
            else:
                plt.plot(time_points, xe_data, '-', color=color, linewidth=2, 
                        label=f'Xe@CC3@t$_{0}$' if key == 'xe1' else f'Xe@Door@t$_{0}$')
    
    plt.xlabel('$t$ / ps')
    plt.ylabel(r'$\Delta_{\text{Xe-CM\{CC3\}}}$ / Å')
    plt.ylim(0, 30)
    
    # Get the current axes
    ax = plt.gca()

    # Create main legend for xenon trajectories first
    main_legend = ax.legend(frameon=False, loc='upper left')

    # Create separate legend for regions at the bottom
    region_legend = ax.legend(handles=region_patches, 
                             labels=[patch.get_label() for patch in region_patches],
                             frameon=False,  # Use frame to make it visible
                             loc='lower center', 
                             bbox_to_anchor=(0.5, -0.45),
                             ncol=3)

    # Add the main legend back to the axes (this is the key step)
    ax.add_artist(main_legend)
    
    # Save as SVG
    plt.savefig(filename_1, format='svg', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(filename_2, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    print(f"Plot saved as {filename_1} and {filename_2}")


# Main execution
if __name__ == "__main__":
    xyz_file = "./2xe_tba_traj_sampled-100_clusters_without_init_str_out.xyz"
    dt = 100  # Time step in femtoseconds
    
    try:
        # Identify all CC3 cages using cluster IDs
        all_cages = identify_all_cc3_cages(xyz_file, expected_cage_size=168, size_tolerance=10)
        
        if len(all_cages) == 0:
            print("No CC3 cages found!")
            exit(1)
        
        print(f"\nProcessing {len(all_cages)} CC3 cages...")
        
        # Process each cage separately
        for cage_info in all_cages:
            print(f"\n{'='*60}")
            print(f"Processing CC3 Cage #{cage_info['id']} (Cluster ID: {cage_info['cluster_id']})")
            print(f"{'='*60}")
            
            # Calculate displacement data for this cage
            displacement_data = calculate_displacement_for_cage(xyz_file, cage_info, dt)
            
            if displacement_data:
                # Create plot filename
                filename_1 = f'xenon_displacement_cage_{cage_info["id"]}_cluster_{cage_info["cluster_id"]}.svg'
                filename_2 = f'xenon_displacement_cage_{cage_info["id"]}_cluster_{cage_info["cluster_id"]}.png'
                
                # Plot data for this cage
                plot_displacement_data_for_cage(displacement_data, cage_info, filename_1, filename_2)
                
                print(f"Completed analysis for cage #{cage_info['id']}")
            else:
                print(f"No displacement data available for cage #{cage_info['id']}")
        
        print(f"\nAnalysis completed! Generated plots for {len(all_cages)} CC3 cages.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
