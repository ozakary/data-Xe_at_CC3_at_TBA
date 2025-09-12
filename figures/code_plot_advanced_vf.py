import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from tqdm import tqdm
from ase.io import read
from ase.geometry import find_mic
from matplotlib import patches
import matplotlib.patches as mpatches

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LinearLocator

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
    atomic_masses = {'H': 1.008, 'C': 12.011, 'N': 14.007}
    
    for cluster_id, cluster_data in clusters.items():
        cage_positions = np.array(cluster_data['positions'])
        cage_elements = cluster_data['elements']
        cage_size = len(cage_positions)
        
        # Calculate cage center of mass
        cage_masses = np.array([atomic_masses[elem] for elem in cage_elements])
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


def analyze_chemical_shifts_for_cage(xyz_filename, csv_filename, cage_info, dt, sigma_iso_ref=5847.626):
    """
    Analyze chemical shifts for xenon atoms nearest to a specific CC3 cage
    """
    print(f"Processing chemical shifts for cage #{cage_info['id']} (cluster {cage_info['cluster_id']})...")
    
    # Get first frame data to identify nearest xenons
    positions, elements, atom_ids, cluster_ids = read_first_frame_data_with_clusters(xyz_filename)
    
    # Get cell information
    first_frame = read(xyz_filename, index='0')
    cell = first_frame.get_cell()
    
    # Find nearest xenon atoms to this cage
    nearest_xe = find_nearest_xenon_atoms_to_cage(
        cage_info['center_of_mass'], 
        positions, 
        elements, 
        atom_ids, 
        cell, 
        n_nearest=2
    )
    
    target_xenon_ids = [xe_id for xe_id, dist in nearest_xe]
    print(f"  Target Xenon atoms: {[(xe_id, f'{dist:.2f} Å') for xe_id, dist in nearest_xe]}")
    
    if len(target_xenon_ids) == 0:
        print(f"  No xenon atoms found near cage #{cage_info['id']}")
        return None
    
    # Read XYZ trajectory for these specific xenons
    print("  Reading XYZ trajectory for target xenons...")
    xenon_xyz_data = read_xyz_file_frames(xyz_filename, target_xenon_ids)
    print(f"  Read {len(xenon_xyz_data)} frames from XYZ file")
    
    # Read CSV file
    print("  Reading CSV file...")
    try:
        df = pd.read_csv(csv_filename)
        xe_data = df[df['element'] == 'Xe']
        xe_data = xe_data.copy()
        xe_data['delta_iso'] = sigma_iso_ref - xe_data['sigma_iso']
    except FileNotFoundError:
        print(f"  Error: CSV file '{csv_filename}' not found")
        return None
    
    # Get available structures in CSV
    available_structures = set(xe_data['structure_id'].unique())
    
    # Create mapping between xenon atoms using XYZ coordinates
    print("  Mapping xenon atoms between XYZ and CSV files...")
    xenon_mapping = {}  # {structure_id: {xenon_id: csv_row}}
    
    matched_structures = 0
    for structure_id in tqdm(sorted(xenon_xyz_data.keys()), desc=f"Cage {cage_info['id']}"):
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
    
    print(f"  Successfully matched {matched_structures} structures")
    
    if not xenon_mapping:
        print(f"  No coordinate matching could be performed for cage #{cage_info['id']}")
        return None
    
    # Extract data for the target xenon atoms
    result_data = {}
    
    for xenon_id in target_xenon_ids:
        xe_data_points = []
        times = []
        
        for structure_id in sorted(xenon_mapping.keys()):
            mapping = xenon_mapping[structure_id]
            if xenon_id in mapping:
                # Convert frame number to time in picoseconds
                time_ps = structure_id * dt / 1e3  # Convert fs to ps
                times.append(time_ps)
                xe_data_points.append(mapping[xenon_id]['delta_iso'])
        
        if len(xe_data_points) > 0:
            # Find initial distance for this xenon
            initial_dist = None
            for xe_id, dist in nearest_xe:
                if xe_id == xenon_id:
                    initial_dist = dist
                    break
            
            result_data[xenon_id] = {
                'times': times,
                'delta_iso': xe_data_points,
                'initial_distance': initial_dist
            }
    
    print(f"  Extracted data for {len(result_data)} xenon atoms")
    return result_data


def plot_chemical_shifts_for_cage(data, cage_info, filename_1, filename_2):
    """
    Plot chemical shift data for a specific cage with Savitzky-Golay smoothing
    FIXED: Sort by initial distance to ensure consistent color assignment
    """
    if not data:
        print(f"No data to plot for cage #{cage_info['id']}")
        return
    
    plt.figure(figsize=(9, 4.5))
    
    # Define colors: #2A9D8F for closest (inside cage), #E76F51 for second closest (at door)
    colors = ['#2A9D8F', '#E76F51', '#F4A261', '#264653']

    # Define regions with colored patterns only
    regions = [
        {'range': (-100, 150), 'pattern_color': 'grey', 'pattern': '...', 'label': 'CC3'},
        {'range': (250, 700), 'pattern_color': 'grey', 'pattern': '///', 'label': 'Door'},
        {'range': (150, 250), 'pattern_color': 'grey', 'pattern': '|||', 'label': 'TBA'}
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
    
    # CRITICAL FIX: Sort by initial distance instead of xenon ID
    # This ensures closest xenon (inside cage) always gets first color (#2A9D8F)
    # and second closest (at door) always gets second color (#E76F51)
    sorted_data = sorted(data.items(), key=lambda x: x[1]['initial_distance'])
    
    for i, (xenon_id, xenon_data) in enumerate(sorted_data):
        times = xenon_data['times']
        delta_iso = xenon_data['delta_iso']
        initial_dist = xenon_data['initial_distance']
        
        color = colors[i % len(colors)]
        
        # Determine label based on distance
        if initial_dist < 3.0:
            location_label = "inside"
        elif initial_dist < 6.0:
            location_label = "door"
        else:
            location_label = "outside"
        
        if len(times) > 4:  # Need enough points for smoothing
            # Apply Savitzky-Golay filter
            window_length = 81  # Must be odd number
            window_length = min(window_length, len(times) if len(times) % 2 == 1 else len(times) - 1)
            
            delta_iso_smooth = savgol_filter(delta_iso, window_length, 3)
            
            # Raw data with high transparency
            plt.plot(times, delta_iso, '-', alpha=0.2, color=color)
            
            # Smoothed data
            plt.plot(times, delta_iso_smooth, '-', color=color, linewidth=2, 
                    label=f'Xe@CC3@t$_{0}$' if i == 0 else f'Xe@Door@t$_{0}$')
        else:
            # Not enough points for smoothing
            plt.plot(times, delta_iso, '-', color=color, linewidth=2, 
                    label=f'Xe@CC3@t$_{0}$' if i == 0 else f'Xe@Door@t$_{0}$')
    
    plt.xlabel('$t$ / ps')
    plt.ylabel(r'$\delta_{\text{iso}}$ / ppm')
    plt.ylim(-100, 700)
    
#    plt.legend(frameon=False, loc='upper left')

    # Get the current axes
    ax = plt.gca()

    # Create main legend for xenon trajectories first
#    main_legend = ax.legend(frameon=False, loc='upper left')

    # Create separate legend for regions at the bottom
    region_legend = ax.legend(handles=region_patches, 
                             labels=[patch.get_label() for patch in region_patches],
                             frameon=False,  # Use frame to make it visible
                             loc='lower center', 
                             bbox_to_anchor=(0.5, -0.45),
                             ncol=3)

    # Add the main legend back to the axes (this is the key step)
#    ax.add_artist(main_legend)
    
    
    ax.yaxis.set_major_locator(LinearLocator(numticks=5))
    
    # Save as SVG
    plt.savefig(filename_1, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(filename_2, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {filename_1} and {filename_2}")
    
    # Print distance info for verification
    print(f"  Color assignment verification for cage #{cage_info['id']}:")
    for i, (xenon_id, xenon_data) in enumerate(sorted_data):
        color = colors[i % len(colors)]
        initial_dist = xenon_data['initial_distance']
        print(f"    Xe{xenon_id}: {initial_dist:.2f} Å → {color}")


# Main execution
if __name__ == "__main__":
    xyz_filename = '../displacement/2xe_tba_traj_sampled-100_clusters_without_init_str_out.xyz'
    csv_filename = './combined_predictions.csv'
    dt = 100  # Time step in femtoseconds
    sigma_iso_ref = 5847.626  # Reference value for chemical shift calculation
    
    try:
        # Identify all CC3 cages using cluster IDs
        all_cages = identify_all_cc3_cages_from_clusters(xyz_filename)
        
        if len(all_cages) == 0:
            print("No CC3 cages found!")
            exit(1)
        
        print(f"\nProcessing chemical shifts for {len(all_cages)} CC3 cages...")
        
        # Process each cage separately
        for cage_info in all_cages:
            print(f"\n{'='*70}")
            print(f"Processing Chemical Shifts for CC3 Cage #{cage_info['id']} (Cluster {cage_info['cluster_id']})")
            print(f"{'='*70}")
            
            # Analyze chemical shifts for this cage
            shift_data = analyze_chemical_shifts_for_cage(
                xyz_filename, 
                csv_filename, 
                cage_info, 
                dt,  # Pass timestep parameter
                sigma_iso_ref
            )
            
            if shift_data:
                # Create plot filename including cluster ID
                filename_1 = f'xenon_chemical_shift_cage_{cage_info["id"]}_cluster_{cage_info["cluster_id"]}.svg'
                filename_2 = f'xenon_chemical_shift_cage_{cage_info["id"]}_cluster_{cage_info["cluster_id"]}.png'
                
                # Plot data for this cage
                plot_chemical_shifts_for_cage(shift_data, cage_info, filename_1, filename_2)
                
                print(f"Completed chemical shift analysis for cage #{cage_info['id']} (cluster {cage_info['cluster_id']})")
            else:
                print(f"No chemical shift data available for cage #{cage_info['id']} (cluster {cage_info['cluster_id']})")
        
        print(f"\nChemical shift analysis completed! Generated plots for {len(all_cages)} CC3 cages.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
