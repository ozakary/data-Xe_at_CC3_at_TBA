import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from scipy.signal import savgol_filter

# Import the figure_formatting module (assuming you have it)
try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
except ImportError:
    print("Warning: figure_formatting_v2 not found, using default matplotlib settings")


def plot_delta_iso_evolution(csv_file, sigma_iso_ref=5847.626, dt_ps=1.0, 
                             apply_smoothing=True, window_length=81, poly_order=3):
    """
    Plot time evolution of delta_iso for Xenon atoms from CSV file with Savitzky-Golay filtering
    
    Parameters:
    - csv_file: path to CSV file
    - sigma_iso_ref: reference sigma_iso value (default: 5847.626 ppm)
    - dt_ps: time step in ps (default: 1.0 ps per row)
    - apply_smoothing: whether to apply Savitzky-Golay filter (default: True)
    - window_length: window length for Savitzky-Golay filter (default: 81)
    - poly_order: polynomial order for Savitzky-Golay filter (default: 3)
    """
    
    print(f"Reading data from: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Filter only Xenon atoms
    xe_data = df[df['element'] == 'Xe'].copy()
    
    if len(xe_data) == 0:
        print("Error: No Xenon atoms found in the CSV file!")
        return None
    
    print(f"Found {len(xe_data)} Xenon data points")
    
    # Calculate delta_iso
    xe_data['delta_iso'] = sigma_iso_ref - xe_data['sigma_iso']
    
    # Create time array (assuming each row represents dt_ps time step)
    xe_data['time_ps'] = xe_data['structure_id'] * dt_ps
    
    # Sort by time to ensure proper plotting
    xe_data = xe_data.sort_values('time_ps')
    
    # Extract data for plotting
    times = xe_data['time_ps'].values
    delta_iso_values = xe_data['delta_iso'].values
    
    # Apply Savitzky-Golay smoothing if requested
    delta_iso_smoothed = None
    if apply_smoothing and len(times) > 4:
        # Adjust window length if needed
        window_length_adj = min(window_length, len(times) if len(times) % 2 == 1 else len(times) - 1)
        if window_length_adj >= poly_order + 1:
            delta_iso_smoothed = savgol_filter(delta_iso_values, window_length_adj, poly_order)
            print(f"Applied Savitzky-Golay filter (window={window_length_adj}, poly_order={poly_order})")
        else:
            print("Warning: Not enough data points for smoothing")
            apply_smoothing = False
    else:
        apply_smoothing = False
    
    # Calculate statistics (use smoothed data if available, otherwise raw data)
    data_for_stats = delta_iso_smoothed if apply_smoothing else delta_iso_values
    stats = {
        'mean': np.mean(data_for_stats),
        'min': np.min(data_for_stats),
        'max': np.max(data_for_stats),
        'std': np.std(data_for_stats, ddof=1),  # Sample standard deviation
        'sem': np.std(data_for_stats, ddof=1) / np.sqrt(len(data_for_stats)),  # Standard error of mean
        'n_points': len(data_for_stats),
        'smoothed': apply_smoothing
    }
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot raw data (transparent)
    ax.plot(times, delta_iso_values, linewidth=0.8, alpha=0.3, color='gray')#, label='Raw data' if apply_smoothing else 'Data')
    
    # Plot smoothed data if available (opaque)
    if apply_smoothing:
        ax.plot(times, delta_iso_smoothed, linewidth=1.5, alpha=1.0, color='#2A9D8F')#, label='Smoothed data')
        
        # Add horizontal line for mean value (always from raw data)
        ax.axhline(y=stats['mean'], color='red', linestyle='--', alpha=0.7, 
                   label=r"$\langle \delta_{\text{iso}} \rangle=$" + f"{stats['mean']:.1f} ± {stats['sem']:.1f} ppm")
    else:
        # Make raw data more visible if no smoothing
        ax.lines[0].set_alpha(0.8)
        ax.lines[0].set_color('#2A9D8F')
        
        # Add horizontal line for raw mean value
        ax.axhline(y=stats['mean'], color='red', linestyle='--', alpha=0.7, 
                   label=f"Mean: {stats['mean']:.2f} ± {stats['sem']:.2f} ppm")
    
    # Formatting
    ax.set_xlabel(r'$t$ / ps')
    ax.set_ylabel(r'$\delta_{\text{iso}}$ / ppm')
    
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 400)
    
    ax.legend(frameon=False)
    
    # Set reasonable tick locators
    ax.xaxis.set_major_locator(LinearLocator(numticks=6))
    ax.yaxis.set_major_locator(LinearLocator(numticks=6))
    
    # Title with basic info
    filter_info = " (Smoothed)" if apply_smoothing else ""
    
    plt.tight_layout()
    
    # Save plots
    output_base = csv_file.replace('.csv', '_delta_iso_evolution')
    if apply_smoothing:
        output_base += '_smoothed'
    plt.savefig(f'{output_base}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DELTA_ISO STATISTICS" + (" (SMOOTHED)" if apply_smoothing else ""))
    print(f"{'='*60}")
    print(f"Number of data points:  {stats['n_points']}")
    print(f"Time range:             {times[0]:.1f} - {times[-1]:.1f} ps")
    print(f"Mean δ_iso:            {stats['mean']:.3f} ± {stats['sem']:.3f} ppm")
    print(f"Standard deviation:     {stats['std']:.3f} ppm")
    print(f"Standard error of mean: {stats['sem']:.3f} ppm")
    print(f"Minimum δ_iso:         {stats['min']:.3f} ppm")
    print(f"Maximum δ_iso:         {stats['max']:.3f} ppm")
    print(f"Range:                  {stats['max'] - stats['min']:.3f} ppm")
    if apply_smoothing:
        print(f"Filter window length:   {window_length_adj}")
        print(f"Polynomial order:       {poly_order}")
    print(f"{'='*60}")
    
    print(f"\nPlots saved as:")
    print(f"  {output_base}.svg")
    print(f"  {output_base}.png")
    
    return {
        'data': xe_data,
        'statistics': stats,
        'times': times,
        'delta_iso_raw': delta_iso_values,
        'delta_iso_smoothed': delta_iso_smoothed if apply_smoothing else None
    }


def plot_multiple_xenon_evolution(csv_file, sigma_iso_ref=5847.626, dt_ps=1.0, max_atoms=None, 
                                 apply_smoothing=True, window_length=81, poly_order=3):
    """
    Plot time evolution of delta_iso for multiple Xenon atoms with Savitzky-Golay filtering
    
    Parameters:
    - csv_file: path to CSV file
    - sigma_iso_ref: reference sigma_iso value (default: 5847.626 ppm)
    - dt_ps: time step in ps (default: 1.0 ps per row)
    - max_atoms: maximum number of atoms to plot (None for all)
    - apply_smoothing: whether to apply Savitzky-Golay filter (default: True)
    - window_length: window length for Savitzky-Golay filter (default: 81)
    - poly_order: polynomial order for Savitzky-Golay filter (default: 3)
    """
    
    print(f"Reading data from: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Filter only Xenon atoms
    xe_data = df[df['element'] == 'Xe'].copy()
    
    if len(xe_data) == 0:
        print("Error: No Xenon atoms found in the CSV file!")
        return None
    
    # Calculate delta_iso
    xe_data['delta_iso'] = sigma_iso_ref - xe_data['sigma_iso']
    
    # Create time array
    xe_data['time_ps'] = xe_data['structure_id'] * dt_ps
    
    # Check if we have multiple atoms per frame
    atoms_per_frame = xe_data.groupby('structure_id')['atom_index'].nunique()
    
    if atoms_per_frame.max() == 1:
        print("Single xenon atom per frame detected, using simple evolution plot...")
        return plot_delta_iso_evolution(csv_file, sigma_iso_ref, dt_ps, apply_smoothing, window_length, poly_order)
    
    print(f"Multiple xenon atoms detected: {atoms_per_frame.max()} atoms per frame")
    
    # Group by atom_index to get individual atom trajectories
    unique_atoms = xe_data['atom_index'].unique()
    
    if max_atoms and len(unique_atoms) > max_atoms:
        unique_atoms = unique_atoms[:max_atoms]
        print(f"Limiting plot to first {max_atoms} atoms")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Colors for different atoms
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_atoms)))
    
    all_delta_iso = []
    all_times = []
    
    # Plot individual atom trajectories
    print("Processing individual atom trajectories...")
    for i, atom_idx in enumerate(unique_atoms):
        atom_data = xe_data[xe_data['atom_index'] == atom_idx].sort_values('structure_id')
        
        if len(atom_data) > 0:
            times = atom_data['time_ps'].values
            delta_iso_vals = atom_data['delta_iso'].values
            
            # Apply smoothing to individual trajectories if requested
            if apply_smoothing and len(times) > 4:
                window_length_adj = min(window_length, len(times) if len(times) % 2 == 1 else len(times) - 1)
                if window_length_adj >= poly_order + 1:
                    delta_iso_smoothed = savgol_filter(delta_iso_vals, window_length_adj, poly_order)
                    
                    # Plot raw data (transparent)
                    ax1.plot(times, delta_iso_vals, linewidth=0.8, alpha=0.2, color=colors[i])
                    
                    # Plot smoothed data (opaque)
                    ax1.plot(times, delta_iso_smoothed, linewidth=1.5, alpha=0.8, 
                            color=colors[i], label=f'Atom {atom_idx}')
                    
                    all_delta_iso.extend(delta_iso_smoothed)
                else:
                    # Not enough points for smoothing this trajectory
                    ax1.plot(times, delta_iso_vals, linewidth=1.0, alpha=0.7, 
                            color=colors[i], label=f'Atom {atom_idx}')
                    all_delta_iso.extend(delta_iso_vals)
            else:
                # No smoothing
                ax1.plot(times, delta_iso_vals, linewidth=1.0, alpha=0.7, 
                        color=colors[i], label=f'Atom {atom_idx}')
                all_delta_iso.extend(delta_iso_vals)
            
            all_times.extend(times)
    
    # Format first subplot
    ax1.set_xlabel(r'$t$ / ps')
    ax1.set_ylabel(r'$\delta_{\text{iso}}$ / ppm')
    
    ax1.set_xlim(0, 5000)
    
    filter_info = " (Smoothed)" if apply_smoothing else ""
    
    if len(unique_atoms) <= 10:  # Only show legend if not too many atoms
        ax1.legend(frameon=False, ncol=2)
    
    # Calculate overall statistics
    all_delta_iso = np.array(all_delta_iso)
    stats = {
        'mean': np.mean(all_delta_iso),
        'min': np.min(all_delta_iso),
        'max': np.max(all_delta_iso),
        'std': np.std(all_delta_iso, ddof=1),
        'sem': np.std(all_delta_iso, ddof=1) / np.sqrt(len(all_delta_iso)),  # Standard error of mean
        'n_points': len(all_delta_iso),
        'n_atoms': len(unique_atoms),
        'smoothed': apply_smoothing
    }
    
    # Plot overall average evolution
    print("Processing average evolution...")
    if len(set(xe_data['structure_id'])) > 1:  # Multiple time points
        avg_by_time = xe_data.groupby('structure_id')['delta_iso'].agg(['mean', 'std']).reset_index()
        avg_by_time['time_ps'] = avg_by_time['structure_id'] * dt_ps
        
        times_avg = avg_by_time['time_ps'].values
        mean_vals = avg_by_time['mean'].values
        std_vals = avg_by_time['std'].values
        
        if apply_smoothing and len(times_avg) > 4:
            window_length_adj = min(window_length, len(times_avg) if len(times_avg) % 2 == 1 else len(times_avg) - 1)
            if window_length_adj >= poly_order + 1:
                mean_smoothed = savgol_filter(mean_vals, window_length_adj, poly_order)
                std_smoothed = savgol_filter(std_vals, window_length_adj, poly_order)
                
                # Plot raw average (transparent)
                ax2.plot(times_avg, mean_vals, linewidth=0.8, alpha=0.3, color='red', label='Raw average')
                ax2.fill_between(times_avg, mean_vals - std_vals, mean_vals + std_vals, 
                                alpha=0.1, color='red')
                
                # Plot smoothed average (opaque)
                ax2.plot(times_avg, mean_smoothed, linewidth=2.0, alpha=1.0, color='red', label='Smoothed average')
                ax2.fill_between(times_avg, mean_smoothed - std_smoothed, mean_smoothed + std_smoothed,
                                alpha=0.3, color='red', label='±1σ (smoothed)')
            else:
                # Not enough points for smoothing
                ax2.plot(times_avg, mean_vals, linewidth=2.0, color='red', label='Average')
                ax2.fill_between(times_avg, mean_vals - std_vals, mean_vals + std_vals,
                                alpha=0.3, color='red', label='±1σ')
        else:
            # No smoothing
            ax2.plot(times_avg, mean_vals, linewidth=2.0, color='red', label='Average')
            ax2.fill_between(times_avg, mean_vals - std_vals, mean_vals + std_vals,
                            alpha=0.3, color='red', label='±1σ')
    
    ax2.set_xlabel(r'$t$ / ps')
    ax2.set_ylabel(r'$\delta_{\text{iso}}$ / ppm')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False)
    filter_avg_info = " (Smoothed)" if apply_smoothing else ""
    ax2.set_title(f'Average Evolution{filter_avg_info} with Standard Deviation')
    
    plt.tight_layout()
    
    # Save plots
    output_base = csv_file.replace('.csv', '_multi_xenon_evolution')
    if apply_smoothing:
        output_base += '_smoothed'
    plt.savefig(f'{output_base}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("MULTI-XENON DELTA_ISO STATISTICS" + (" (SMOOTHED)" if apply_smoothing else ""))
    print(f"{'='*60}")
    print(f"Number of atoms:        {stats['n_atoms']}")
    print(f"Total data points:      {stats['n_points']}")
    print(f"Time range:             {min(all_times):.1f} - {max(all_times):.1f} ps")
    print(f"Overall mean δ_iso:    {stats['mean']:.3f} ± {stats['sem']:.3f} ppm")
    print(f"Overall std deviation:  {stats['std']:.3f} ppm")
    print(f"Overall SEM:            {stats['sem']:.3f} ppm")
    print(f"Overall minimum:        {stats['min']:.3f} ppm")
    print(f"Overall maximum:        {stats['max']:.3f} ppm")
    print(f"Overall range:          {stats['max'] - stats['min']:.3f} ppm")
    if apply_smoothing:
        print(f"Filter window length:   {window_length}")
        print(f"Polynomial order:       {poly_order}")
    print(f"{'='*60}")
    
    return {
        'data': xe_data,
        'statistics': stats,
        'unique_atoms': unique_atoms
    }


# Main execution
if __name__ == "__main__":
    csv_file = './output_prediction.csv'
    
    try:
        # Try to detect if we have single or multiple atoms
        df_test = pd.read_csv(csv_file)
        xe_test = df_test[df_test['element'] == 'Xe']
        
        if len(xe_test) == 0:
            print("No Xenon atoms found in CSV file!")
        else:
            atoms_per_frame = xe_test.groupby('structure_id')['atom_index'].nunique()
            max_atoms_per_frame = atoms_per_frame.max()
            
            if max_atoms_per_frame == 1:
                print("Single xenon atom trajectory detected")
                results = plot_delta_iso_evolution(csv_file, apply_smoothing=True)
            else:
                print(f"Multiple xenon atoms detected ({max_atoms_per_frame} max per frame)")
                results = plot_multiple_xenon_evolution(csv_file, max_atoms=16, apply_smoothing=True)
            
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
