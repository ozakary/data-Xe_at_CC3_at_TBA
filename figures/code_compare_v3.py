import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter

# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, Dict, List, Any

def parse_xyz_file(filename, is_ml_file=False):
    """Parse XYZ file and extract properties.
    
    Args:
        filename: Path to the XYZ file
        is_ml_file: Boolean flag indicating if this is an ML prediction file with original_dataset_index
    
    Returns:
        List of structures with properties
    """
    structures = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        structure_index = 0
        while i < len(lines):
            n_atoms = int(lines[i].strip())
            header = lines[i + 1].strip()
            
            # Parse header information
            properties = {}
            
            # Handle the special case where we want to track original indices
            if is_ml_file:
                properties['original_index'] = None
            else:
                properties['original_index'] = structure_index
                
            # Split header into parts, properly handling quoted strings
            parts = []
            in_quotes = False
            current_part = ''
            
            for char in header:
                if char == '"':
                    in_quotes = not in_quotes
                    current_part += char
                elif char == ' ' and not in_quotes:
                    if current_part:
                        parts.append(current_part)
                        current_part = ''
                else:
                    current_part += char
            if current_part:
                parts.append(current_part)
            
            # Process parts to extract key-value pairs
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    properties[key] = value.strip('"')
            
            # Extract and process numeric properties
            if 'stress' in properties:
                stress_str = properties['stress'].strip('"')
                properties['stress'] = [float(x) for x in stress_str.split()]
            
            # Parse other numeric properties
            for key in ['energy', 'free_energy']:
                if key in properties:
                    try:
                        properties[key] = float(properties[key])
                    except ValueError:
                        pass
            
            # Extract original_dataset_index for ML files
            if is_ml_file and 'original_dataset_index' in properties:
                try:
                    properties['original_index'] = int(properties['original_dataset_index'])
                except ValueError:
                    pass
            
            # Extract atomic data
            atoms = []
            forces = []
            for j in range(n_atoms):
                atom_data = lines[i + 2 + j].strip().split()
                
                # Base atom data includes species and position
                atom = {
                    'species': atom_data[0],
                    'position': [float(x) for x in atom_data[1:4]],
                }
                
                # If forces are present (expected format)
                if len(atom_data) >= 7:
                    atom['forces'] = [float(x) for x in atom_data[4:7]]
                    forces.append([float(x) for x in atom_data[4:7]])
                
                atoms.append(atom)
            
            properties['atoms'] = atoms
            properties['forces'] = np.array(forces)
            structures.append(properties)
            
            # Move to the next structure
            i += n_atoms + 2
            structure_index += 1
            
    return structures

def create_joint_plot(x_data, y_data, xlabel, ylabel, filename, scale_factor=1.0, 
                     tick_format='{x:.3f}', stats_format=None):
    """Create and save a joint plot with marginal distributions.
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        scale_factor: Factor by which data has been scaled for display (default=1.0)
        tick_format: Format string for tick labels (default='{x:.3f}')
        stats_format: Dictionary with format strings for R2, RMSE, and MAE 
                     (default={'r2': 6, 'rmse': 6, 'mae': 6})
    """
    # Set default stats formatting if not provided
    if stats_format is None:
        stats_format = {'r2': 6, 'rmse': 6, 'mae': 6}
    
    plt.figure(figsize=(6, 6))
    
    # Create the joint plot
    g = sns.JointGrid(data=None, x=x_data, y=y_data, height=6)
    
    # Add the scatter plot with specific color and size
    g.plot_joint(plt.scatter, alpha=0.5, color='#2A9D8F', s=100, edgecolor='#2A9D8F')
    
    # Add marginal distributions with matching color
    n_bins = 50
    g.plot_marginals(sns.histplot, color='#2A9D8F', kde=False, bins=n_bins)
    
    # Update KDE line color in marginal plots
    for ax in [g.ax_marg_x, g.ax_marg_y]:
        for line in ax.lines:
            line.set_color('#2A9D8F')
            
    # Add labels and title
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)

    # Control number of ticks
    x_min, x_max = g.ax_joint.get_xlim()
    y_min, y_max = g.ax_joint.get_ylim()
    
    overall_min = min(x_min, y_min)
    overall_max = max(x_max, y_max)
    
    # Create 4 evenly spaced ticks
    x_ticks = np.linspace(overall_min, overall_max, 4)
    y_ticks = np.linspace(overall_min, overall_max, 4)
    
    g.ax_joint.set_xticks(x_ticks)
    g.ax_joint.set_yticks(y_ticks)
 
    # Format tick labels using the custom format
    g.ax_joint.xaxis.set_major_formatter(StrMethodFormatter(tick_format))
    g.ax_joint.yaxis.set_major_formatter(StrMethodFormatter(tick_format))
    # Rotate x-axis tick labels by 45 degrees
    plt.setp(g.ax_joint.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_joint.get_yticklabels(), rotation=45, ha='right')

    # Calculate statistics on the raw (unscaled) data
    # Rescale the data if necessary
    x_raw = x_data / scale_factor
    y_raw = y_data / scale_factor
    
    # Add regression statistics on raw data
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_raw, y_raw)
    rmse = np.sqrt(np.mean((x_raw - y_raw)**2))
    mae = np.mean(np.abs(x_raw - y_raw))
    
    # Display the statistics (raw data) with custom formatting
    stats_text = f'R² = {r_value**2:.{stats_format["r2"]}f}\nRMSE = {rmse:.{stats_format["rmse"]}f}\nMAE = {mae:.{stats_format["mae"]}f}'
    g.ax_joint.text(0.05, 0.75, stats_text,
                   transform=g.ax_joint.transAxes)
 
    # Add regression line (for display data)
    slope_display, intercept_display, _, _, _ = stats.linregress(x_data, y_data)
    line_x = np.array([min(x_data), max(x_data)])
    line_y = slope_display * line_x + intercept_display
    g.ax_joint.plot(line_x, line_y, alpha=1, label=f'y = {slope_display:.4f}x + {intercept_display:.4f}', color='#264653')

    # Plot identity line (x=y) for reference
    min_val = min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0])
    max_val = max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    # Save the plot
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()

def export_to_csv(data, filename):
    """Export data to a CSV file."""
    with open(filename, 'w') as f:
        # Write header
        header = ','.join(str(key) for key in data.keys())
        f.write(header + '\n')
        
        # Write data rows
        for i in range(len(next(iter(data.values())))):
            row = ','.join(str(data[key][i]) if i < len(data[key]) else '' for key in data.keys())
            f.write(row + '\n')
    
    print(f"Data exported to {filename}")

def align_structures_by_coordinates(dft_data, ml_data, tolerance=1e-6):
    """Align DFT and ML structures based on atomic coordinates.
    
    Args:
        dft_data: List of DFT structures
        ml_data: List of ML structures  
        tolerance: Tolerance for coordinate matching (default: 1e-6 Angstrom)
    
    Returns:
        Tuple of aligned (dft_structures, ml_structures)
    """
    
    def get_coordinate_signature(structure):
        """Create a coordinate signature for structure matching."""
        positions = []
        for atom in structure['atoms']:
            positions.extend(atom['position'])
        return np.array(positions)
    
    def structures_match(struct1, struct2, tol=tolerance):
        """Check if two structures have matching coordinates."""
        # First check if they have the same number of atoms
        if len(struct1['atoms']) != len(struct2['atoms']):
            return False
            
        coords1 = get_coordinate_signature(struct1)
        coords2 = get_coordinate_signature(struct2)
        
        # Check if coordinate arrays have the same length
        if len(coords1) != len(coords2):
            return False
            
        # Check if coordinates match within tolerance
        return np.allclose(coords1, coords2, atol=tol)
    
    aligned_dft = []
    aligned_ml = []
    
    print(f"Aligning {len(dft_data)} DFT structures with {len(ml_data)} ML structures...")
    
    # For each DFT structure, find the corresponding ML structure
    for i, dft_structure in enumerate(dft_data):
        if i % 100 == 0:  # Progress indicator for large datasets
            print(f"  Processing structure {i+1}/{len(dft_data)}")
            
        match_found = False
        for j, ml_structure in enumerate(ml_data):
            if structures_match(dft_structure, ml_structure):
                aligned_dft.append(dft_structure)
                aligned_ml.append(ml_structure)
                match_found = True
                break
        
        if not match_found:
            print(f"  Warning: No matching ML structure found for DFT structure {i}")
    
    print(f"Successfully aligned {len(aligned_dft)} structure pairs")
    return aligned_dft, aligned_ml

def identify_outliers(dft_data, ml_data, criteria=None):
    """Identify outlier structures where ML predictions are way off.
    
    Args:
        dft_data: List of DFT structures
        ml_data: List of ML structures
        criteria: Dictionary with outlier criteria. Default uses multiple methods:
                 {
                     'energy_abs_threshold': 0.1,  # eV absolute difference
                     'energy_std_multiplier': 3.0,  # std deviations
                     'force_rmse_threshold': 2.0,   # eV/Å
                     'percentile_worst': 5          # worst 5% of predictions
                 }
    
    Returns:
        Dictionary with outlier indices and reasons
    """
    
    if criteria is None:
        criteria = {
            'energy_abs_threshold': 0.1,    # eV
            'energy_std_multiplier': 3.0,   # standard deviations
            'force_rmse_threshold': 2.0,    # eV/Å
            'percentile_worst': 5            # worst 5%
        }
    
    n_structures = len(dft_data)
    outliers = {}
    
    # Calculate energy differences
    dft_energies = np.array([s['energy'] for s in dft_data])
    ml_energies = np.array([s['energy'] for s in ml_data])
    energy_diffs = np.abs(dft_energies - ml_energies)
    
    # Calculate force RMSE for each structure
    force_rmse_per_structure = []
    for i in range(n_structures):
        dft_forces = dft_data[i]['forces'].flatten()
        ml_forces = ml_data[i]['forces'].flatten()
        rmse = np.sqrt(np.mean((dft_forces - ml_forces)**2))
        force_rmse_per_structure.append(rmse)
    force_rmse_per_structure = np.array(force_rmse_per_structure)
    
    print(f"\nOutlier Detection Summary:")
    print(f"Energy difference - Mean: {np.mean(energy_diffs):.4f} eV, Std: {np.std(energy_diffs):.4f} eV")
    print(f"Force RMSE - Mean: {np.mean(force_rmse_per_structure):.4f} eV/Å, Std: {np.std(force_rmse_per_structure):.4f} eV/Å")
    
    # Method 1: Absolute energy threshold
    if 'energy_abs_threshold' in criteria:
        energy_outliers = np.where(energy_diffs > criteria['energy_abs_threshold'])[0]
        for idx in energy_outliers:
            if idx not in outliers:
                outliers[idx] = []
            outliers[idx].append(f"Energy diff: {energy_diffs[idx]:.4f} eV > {criteria['energy_abs_threshold']} eV")
    
    # Method 2: Statistical outliers (standard deviation)
    if 'energy_std_multiplier' in criteria:
        energy_mean = np.mean(energy_diffs)
        energy_std = np.std(energy_diffs)
        energy_threshold = energy_mean + criteria['energy_std_multiplier'] * energy_std
        statistical_outliers = np.where(energy_diffs > energy_threshold)[0]
        for idx in statistical_outliers:
            if idx not in outliers:
                outliers[idx] = []
            outliers[idx].append(f"Energy diff: {energy_diffs[idx]:.4f} eV > {energy_threshold:.4f} eV ({criteria['energy_std_multiplier']}σ)")
    
    # Method 3: Force RMSE threshold
    if 'force_rmse_threshold' in criteria:
        force_outliers = np.where(force_rmse_per_structure > criteria['force_rmse_threshold'])[0]
        for idx in force_outliers:
            if idx not in outliers:
                outliers[idx] = []
            outliers[idx].append(f"Force RMSE: {force_rmse_per_structure[idx]:.4f} eV/Å > {criteria['force_rmse_threshold']} eV/Å")
    
    # Method 4: Percentile-based (worst predictions)
    if 'percentile_worst' in criteria:
        percentile_threshold = 100 - criteria['percentile_worst']
        energy_percentile = np.percentile(energy_diffs, percentile_threshold)
        percentile_outliers = np.where(energy_diffs > energy_percentile)[0]
        for idx in percentile_outliers:
            if idx not in outliers:
                outliers[idx] = []
            outliers[idx].append(f"Worst {criteria['percentile_worst']}% energy predictions (>{energy_percentile:.4f} eV)")
    
    print(f"\nFound {len(outliers)} outlier structures:")
    for idx, reasons in outliers.items():
        print(f"  Structure {idx}: Energy diff = {energy_diffs[idx]:.4f} eV, Force RMSE = {force_rmse_per_structure[idx]:.4f} eV/Å")
        for reason in reasons:
            print(f"    - {reason}")
    
    return outliers, energy_diffs, force_rmse_per_structure

def write_xyz_file(structures, filename, source_label=""):
    """Write structures to XYZ file format.
    
    Args:
        structures: List of structure dictionaries
        filename: Output filename
        source_label: Label for the source (e.g., "DFT", "ML")
    """
    
    with open(filename, 'w') as f:
        for struct_idx, structure in enumerate(structures):
            n_atoms = len(structure['atoms'])
            
            # Write number of atoms
            f.write(f"{n_atoms}\n")
            
            # Construct header with properties
            header_parts = []
            
            # Add energy if present
            if 'energy' in structure:
                header_parts.append(f'energy={structure["energy"]}')
            
            # Add stress if present
            if 'stress' in structure and structure['stress'] is not None:
                stress_str = ' '.join([f"{x}" for x in structure['stress']])
                header_parts.append(f'stress="{stress_str}"')
            
            # Add other properties if present
            for key, value in structure.items():
                if key not in ['atoms', 'forces', 'energy', 'stress', 'original_index']:
                    header_parts.append(f'{key}={value}')
            
            # Add source label if provided
            if source_label:
                header_parts.append(f'source={source_label}')
                
            # Add outlier index for reference
            header_parts.append(f'outlier_structure_index={struct_idx}')
            
            header = ' '.join(header_parts)
            f.write(f"{header}\n")
            
            # Write atomic data
            for atom in structure['atoms']:
                species = atom['species']
                pos = atom['position']
                
                # Write coordinates
                line = f"{species} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
                
                # Add forces if present
                if 'forces' in atom:
                    forces = atom['forces']
                    line += f" {forces[0]:.6f} {forces[1]:.6f} {forces[2]:.6f}"
                
                f.write(line + "\n")
    
    print(f"Saved {len(structures)} structures to {filename}")

def save_outlier_structures(dft_data, ml_data, outlier_indices, output_prefix="outlier"):
    """Save outlier structures to separate XYZ files.
    
    Args:
        dft_data: List of DFT structures
        ml_data: List of ML structures
        outlier_indices: List of indices of outlier structures
        output_prefix: Prefix for output filenames
    """
    
    if len(outlier_indices) == 0:
        print("No outlier structures to save.")
        return
    
    # Extract outlier structures
    outlier_dft = [dft_data[i] for i in outlier_indices]
    outlier_ml = [ml_data[i] for i in outlier_indices]
    
    # Save to separate files
    dft_filename = f"{output_prefix}_dft.xyz"
    ml_filename = f"{output_prefix}_ml.xyz"
    
    write_xyz_file(outlier_dft, dft_filename, "DFT")
    write_xyz_file(outlier_ml, ml_filename, "ML")
    
    # Also create a summary file
    summary_filename = f"{output_prefix}_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Outlier Structures Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Total outlier structures found: {len(outlier_indices)}\n\n")
        
        for i, idx in enumerate(outlier_indices):
            dft_energy = dft_data[idx]['energy']
            ml_energy = ml_data[idx]['energy']
            energy_diff = abs(dft_energy - ml_energy)
            
            # Calculate force RMSE
            dft_forces = dft_data[idx]['forces'].flatten()
            ml_forces = ml_data[idx]['forces'].flatten()
            force_rmse = np.sqrt(np.mean((dft_forces - ml_forces)**2))
            
            f.write(f"Outlier {i+1} (Original index: {idx}):\n")
            f.write(f"  DFT Energy: {dft_energy:.6f} eV\n")
            f.write(f"  ML Energy:  {ml_energy:.6f} eV\n")
            f.write(f"  Energy Difference: {energy_diff:.6f} eV\n")
            f.write(f"  Force RMSE: {force_rmse:.6f} eV/Å\n")
            f.write(f"  Number of atoms: {len(dft_data[idx]['atoms'])}\n\n")
    
    print(f"Outlier analysis complete:")
    print(f"  DFT structures: {dft_filename}")
    print(f"  ML structures: {ml_filename}")
    print(f"  Summary: {summary_filename}")

def plot_comparisons(dft_data, ml_data):
    """Create individual comparison plots for all properties and export CSV data."""
    
    # Define custom formatters for different quantities
    energy_format = '{x:.3f}'      # 1 decimal place for energy (scaled by 1e-3)
    force_format = '{x:.0f}'       # 4 decimal places for forces
    stress_format = '{x:.0f}'      # 2 decimal places for stress (scaled by 1e3)
    
    # Define custom statistics formatting for different quantities
    energy_stats_format = {'r2': 2, 'rmse': 1, 'mae': 1}      # R²: 6 decimals, RMSE/MAE: 3 decimals
    force_stats_format = {'r2': 3, 'rmse': 2, 'mae': 2}       # All: 4 decimal places
    stress_stats_format = {'r2': 4, 'rmse': 5, 'mae': 5}      # R²: 4 decimals, RMSE/MAE: 6 decimals
    
    # Energy comparison
    dft_energies = np.array([s['energy'] for s in dft_data])
    ml_energies = np.array([s['energy'] for s in ml_data])
    
    # Export energy data (raw values)
    energy_data = {
        'structure_index': np.arange(len(dft_energies)),
        'dft_energy': dft_energies,
        'ml_energy': ml_energies
    }
    export_to_csv(energy_data, 'energy_comparison.csv')
    
    # For display, scale by 1e-3 but compute stats on raw data
    display_scale = 1e-3
    create_joint_plot(
        dft_energies*display_scale, ml_energies*display_scale,
        r'$E_{\text{DFT}}$ $\times10^{-3}$ / eV', r'$E_{\text{ML}}$ $\times10^{-3}$ / eV',
        'energy_comparison.svg',
        scale_factor=display_scale,
        tick_format=energy_format,
        stats_format=energy_stats_format
    )
    
    # Forces comparisons
    dft_forces = np.concatenate([s['forces'] for s in dft_data])
    ml_forces = np.concatenate([s['forces'] for s in ml_data])
    
    # Export force data
    force_data = {
        'atom_index': np.arange(len(dft_forces.flatten())//3),
        'dft_force_x': dft_forces[:,0],
        'ml_force_x': ml_forces[:,0],
        'dft_force_y': dft_forces[:,1],
        'ml_force_y': ml_forces[:,1],
        'dft_force_z': dft_forces[:,2],
        'ml_force_z': ml_forces[:,2]
    }
    export_to_csv(force_data, 'forces_comparison.csv')
    
    # Total forces - no scaling needed
    create_joint_plot(
        dft_forces.flatten(), ml_forces.flatten(),
        r'$\vec{f}_{\text{DFT}}$ / eV.Å$^{-1}$', r'$\vec{f}_{\text{ML}}$ / eV.Å$^{-1}$',
        'forces_total_comparison.png',
        scale_factor=1.0,
        tick_format=force_format,
        stats_format=force_stats_format
    )
    
    # Individual force components - no scaling needed
    components = [r'$f_{x}$', r'$f_{y}$', r'$f_{z}$']
    components_2 = ['fx', 'fy', 'fz']
    for i in range(3):
        create_joint_plot(
            dft_forces[:,i], ml_forces[:,i],
            components[i] + r'$^{{\text{DFT}}}$ / eV.Å$^{{-1}}$',
            components[i] + r'$^{{\text{ML}}}$ / eV.Å$^{{-1}}$',
            f'force_{components_2[i]}_comparison.png',
            scale_factor=1.0,
            tick_format=force_format,
            stats_format=force_stats_format
        )
    
    # Stress comparisons
    try:
        # Store raw values for CSV
        raw_dft_stress = np.array([s['stress'] for s in dft_data])
        raw_ml_stress = np.array([s['stress'] for s in ml_data])
        
        # Apply scaling for display purposes (1e3)
        dft_stress = raw_dft_stress * 1e3
        ml_stress = raw_ml_stress * 1e3
        
        # Stress components for reference
        stress_components = [
            ('xx', 0), ('xy', 1), ('xz', 2),
            ('yx', 1), ('yy', 4), ('yz', 5),
            ('zx', 2), ('zy', 5), ('zz', 8)
        ]
        
        # Export stress data with raw values
        stress_data = {'structure_index': np.arange(len(raw_dft_stress))}
        for label, idx in stress_components:
            stress_data[f'dft_stress_{label}'] = raw_dft_stress[:,idx]
            stress_data[f'ml_stress_{label}'] = raw_ml_stress[:,idx]
        
        export_to_csv(stress_data, 'stress_comparison.csv')
        
        # Scale for display but compute stats on raw data
        display_scale = 1e3
        
        # Total stress (using flattened arrays for all components)
        create_joint_plot(
            dft_stress.flatten(), ml_stress.flatten(),
            r'$\boldsymbol{s}_{\text{DFT}}$ $\times10^{3}$ / eV.Å$^{-3}$', 
            r'$\boldsymbol{s}_{\text{ML}}$ $\times10^{3}$ / eV.Å$^{-3}$',
            'stress_total_comparison.svg',
            scale_factor=display_scale,
            tick_format=stress_format,
            stats_format=stress_stats_format
        )

        # Define LaTeX templates outside f-strings
        dft_stress_template = r'$s_{{{}}}^{{\text{{DFT}}}} \times10^{{3}}$ / eV.Å$^{{-3}}$'
        ml_stress_template = r'$s_{{{}}}^{{\text{{ML}}}} \times10^{{3}}$ / eV.Å$^{{-3}}$'
        
        # Individual stress components
        for label, idx in stress_components:
            try:
                create_joint_plot(
                    dft_stress[:,idx], ml_stress[:,idx],
                    dft_stress_template.format(label),
                    ml_stress_template.format(label),
                    f'stress_{label}_comparison.svg',
                    scale_factor=display_scale,
                    tick_format=stress_format,
                    stats_format=stress_stats_format
                )
            except IndexError as e:
                print(f"Error creating stress {label} plot: {e}")
    except (KeyError, ValueError) as e:
        print(f"Error processing stress data: {e}")

def main():
    # Define input filenames
    dft_file = '../../dataset_2xe_cc3_tba.xyz'
    ml_file = '../../dataset_2xe_cc3_tba_out_prediction.xyz'
    
    # Parse the files
    print("Reading DFT structures...")
    dft_structures = parse_xyz_file(dft_file, is_ml_file=False)
    print(f"Found {len(dft_structures)} structures in DFT file")
    
    print("\nReading ML structures...")
    ml_structures = parse_xyz_file(ml_file, is_ml_file=True)
    print(f"Found {len(ml_structures)} structures in ML file")
    
    # Print first structure's properties for debugging
    if dft_structures:
        print("\nFirst DFT structure properties:")
        print(f"  Energy: {dft_structures[0].get('energy')}")
        print(f"  Number of atoms: {len(dft_structures[0]['atoms'])}")
        print(f"  First atom position: {dft_structures[0]['atoms'][0]['position']}")
        if 'stress' in dft_structures[0]:
            print(f"  Stress tensor shape: {len(dft_structures[0]['stress'])}")
    
    if ml_structures:
        print("\nFirst ML structure properties:")
        print(f"  Energy: {ml_structures[0].get('energy')}")
        print(f"  Number of atoms: {len(ml_structures[0]['atoms'])}")
        print(f"  First atom position: {ml_structures[0]['atoms'][0]['position']}")
        if 'stress' in ml_structures[0]:
            print(f"  Stress tensor shape: {len(ml_structures[0]['stress'])}")
    
    # Align structures based on atomic coordinates
    print("\nAligning DFT and ML structures based on atomic coordinates...")
    aligned_dft, aligned_ml = align_structures_by_coordinates(dft_structures, ml_structures)
    
    # Verify alignment worked
    if len(aligned_dft) == 0:
        print("ERROR: No structures could be aligned! Check coordinate tolerance or file formats.")
        return
    
    # Quick verification: check first aligned pair
    print(f"\nVerification of first aligned pair:")
    print(f"  DFT energy: {aligned_dft[0]['energy']}")
    print(f"  ML energy: {aligned_ml[0]['energy']}")
    dft_first_atom = aligned_dft[0]['atoms'][0]['position']
    ml_first_atom = aligned_ml[0]['atoms'][0]['position']
    print(f"  DFT first atom: {dft_first_atom}")
    print(f"  ML first atom: {ml_first_atom}")
    print(f"  Coordinate difference: {np.array(dft_first_atom) - np.array(ml_first_atom)}")
    
    # Detect outlier structures
    print("\nDetecting outlier structures...")
    
    # Customize these criteria based on your needs
    outlier_criteria = {
        'energy_abs_threshold': 10,   # 10 eV absolute difference
        'energy_std_multiplier': 5,   # 2.5 standard deviations  
        'force_rmse_threshold': 1.5,    # 1.5 eV/Å force RMSE
        'percentile_worst': 1           # worst 10% of predictions
    }
    
    outliers, energy_diffs, force_rmse = identify_outliers(
        aligned_dft, aligned_ml, criteria=outlier_criteria
    )
    
    # Save outlier structures if any found
    if outliers:
        outlier_indices = list(outliers.keys())
        save_outlier_structures(
            aligned_dft, aligned_ml, outlier_indices, 
            output_prefix="energy_force_outliers"
        )
    else:
        print("No outlier structures found with current criteria.")
    
    # Create all comparison plots and export CSV data
    print("\nCreating comparison plots and exporting CSV data...")
    plot_comparisons(aligned_dft, aligned_ml)
    
    print("\nAnalysis complete. Individual plots have been saved and data exported to CSV files.")

if __name__ == "__main__":
    main()
