import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from glob import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Import the figure_formatting module for consistent plotting style
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)


def read_rdf_file(filename):
    """Read RDF data from a file and extract the distance and g(r) values."""
    distances = []
    g_r_values = []
    
    with open(filename, 'r') as f:
        # Skip the header lines
        for i, line in enumerate(f):
            if i < 2:  # Skip the first two lines (headers)
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:  # Ensure we have at least 2 columns
                distances.append(float(parts[0]))
                g_r_values.append(float(parts[1]))  # g(r) is the 2nd column
    
    return np.array(distances), np.array(g_r_values)


def process_rdf_files(file_pattern, max_files=1500):
    """Process multiple RDF files and average their values."""
    all_files = sorted(glob(file_pattern))[:max_files]
    
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    # Read the first file to get distances and initialize the sum
    distances, first_rdf = read_rdf_file(all_files[0])
    sum_rdf = first_rdf
    
    # Process the rest of the files
    for i, filename in enumerate(all_files[1:], 1):
        _, rdf_values = read_rdf_file(filename)
        sum_rdf += rdf_values
        
        # Optional: print progress
        if i % 100 == 0:
            print(f"Processed {i+1} files out of {len(all_files)}")
    
    # Calculate the average
    avg_rdf = sum_rdf / len(all_files)
    
    return distances, avg_rdf


def smooth_rdf(distances, rdf_values, window_size=5, poly_order=1):
    """Apply Savitzky-Golay filter to smooth the RDF data."""
    # Make sure window_size is odd and smaller than the data length
    if window_size >= len(rdf_values):
        window_size = min(len(rdf_values) - 1, 5)
        if window_size % 2 == 0:
            window_size -= 1
    
    # Apply the Savitzky-Golay filter for smoothing
    smoothed_rdf = savgol_filter(rdf_values, window_size, poly_order)
    
    # Ensure no negative values in the smoothed RDF
    smoothed_rdf = np.maximum(smoothed_rdf, 0)
    
    return smoothed_rdf


def calculate_reliability_factor(data1, data2):
    """
    Calculate reliability factor between two curves.
    R-factor = 100% * (1 - Σ|data1 - data2|/Σ|data1 + data2|)
    
    Values close to 100% indicate better agreement between curves.
    """
    # Ensure no division by zero
    denominator = np.sum(np.abs(data1 + data2))
    if denominator == 0:
        return 0.0
    
    numerator = np.sum(np.abs(data1 - data2))
    r_factor = 100.0 * (1.0 - numerator / denominator)
    
    return r_factor


def plot_comparison_rdf(aimd_data, mlip_data, output_file='rdf_mlmd_vs_aimd_with_insets.svg'):
    """Plot smoothed RDF data from AIMD and MLIP for comparison with difference below zero and inset plot."""
    aimd_distances, aimd_smoothed = aimd_data
    mlip_distances, mlip_smoothed = mlip_data
    
    # Handle data with different x-values
    if np.array_equal(aimd_distances, mlip_distances):
        # Same x-values
        difference = aimd_smoothed - mlip_smoothed
        difference_x = aimd_distances
        r_factor = calculate_reliability_factor(aimd_smoothed, mlip_smoothed)
    else:
        # Different x-values - need to interpolate
        from scipy.interpolate import interp1d
        mlip_interp = interp1d(mlip_distances, mlip_smoothed, 
                              bounds_error=False, fill_value=0)
        mlip_resampled = mlip_interp(aimd_distances)
        difference = aimd_smoothed - mlip_resampled
        difference_x = aimd_distances
        r_factor = calculate_reliability_factor(aimd_smoothed, mlip_resampled)
    
    # Print the reliability factor
    print(f"Reliability factor: {r_factor:.2f}%")
    
    # Now create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   sharex=True)
    
    # Plot AIMD data on top subplot
    ax1.plot(aimd_distances, aimd_smoothed, color='#264653', linewidth=1, label='AIMD')
    ax1.fill_between(aimd_distances, 0, aimd_smoothed, color='#264653', alpha=0.3)
    
    # Plot MLIP data on top subplot
    ax1.plot(mlip_distances, mlip_smoothed, linestyle='--', color='#E76F51', linewidth=1, label='MLMD')
    ax1.fill_between(mlip_distances, 0, mlip_smoothed, color='#E76F51', alpha=0.3)
    
    # Add labels and set limits for top subplot
    ax1.set_ylabel(r'$g$($r$)')
    ax1.set_ylim(0, 8)
    ax1.legend(frameon=False, loc='upper left')
    
    # Create inset plots - two separate insets stacked vertically
    # Top inset for RDF comparison
    axins_top = inset_axes(ax1, width="50%", height="55%", loc='upper right', 
                          bbox_to_anchor=(0.02, 0.15, 0.96, 0.83), bbox_transform=ax1.transAxes)
    
    # Bottom inset for difference plot  
    axins_bottom = inset_axes(ax1, width="50%", height="35%", loc='upper right',
                             bbox_to_anchor=(0.02, 0.08, 0.96, 0.40), bbox_transform=ax1.transAxes)

    # Plot RDF data in the top inset
    axins_top.plot(aimd_distances, aimd_smoothed, color='#264653', linewidth=1)
    axins_top.fill_between(aimd_distances, 0, aimd_smoothed, color='#264653', alpha=0.3)
    axins_top.plot(mlip_distances, mlip_smoothed, linestyle='--', color='#E76F51', linewidth=1)
    axins_top.fill_between(mlip_distances, 0, mlip_smoothed, color='#E76F51', alpha=0.3)
    
    # Plot difference in the bottom inset
    axins_bottom.plot(difference_x, difference, color='#666666', linewidth=1.2, linestyle='-')
    axins_bottom.axhline(y=0, color='#999999', linestyle='--', linewidth=0.8)
    
    # Set the zoom region for both inset subplots (0.5 to 3 Angstroms)
    axins_top.set_xlim(0.8, 1.6)
    axins_bottom.set_xlim(0.8, 1.6)
    
    # Calculate appropriate y-limits for the inset based on data in the zoom region
    zoom_mask_aimd = (aimd_distances >= 0.5) & (aimd_distances <= 3.0)
    zoom_mask_mlip = (mlip_distances >= 0.5) & (mlip_distances <= 3.0)
    zoom_mask_diff = (difference_x >= 0.5) & (difference_x <= 3.0)
    
    # Set y-limits for top inset (RDF)
    if np.any(zoom_mask_aimd) or np.any(zoom_mask_mlip):
        max_val_aimd = np.max(aimd_smoothed[zoom_mask_aimd]) if np.any(zoom_mask_aimd) else 0
        max_val_mlip = np.max(mlip_smoothed[zoom_mask_mlip]) if np.any(zoom_mask_mlip) else 0
        max_val_zoom = max(max_val_aimd, max_val_mlip)
        axins_top.set_ylim(0, max_val_zoom * 1.1)  # Add 10% padding
    else:
        axins_top.set_ylim(0, 8)  # Fallback to main plot limits
    
    # Set y-limits for bottom inset (difference)
    if np.any(zoom_mask_diff):
        diff_zoom = difference[zoom_mask_diff]
        diff_abs_max_zoom = max(abs(np.nanmin(diff_zoom)), abs(np.nanmax(diff_zoom)))
        y_limit_zoom = diff_abs_max_zoom * 1.2
        axins_bottom.set_ylim(-y_limit_zoom, y_limit_zoom)
    else:
        axins_bottom.set_ylim(-1, 1)  # Fallback
    
    # Style the insets
    axins_top.tick_params(labelsize=14)
    axins_top.set_xticklabels([])  # Remove x-axis labels from top inset
    
    axins_bottom.tick_params(labelsize=14)
    
    # Set the inset reference for mark_inset (use the top inset)
    axins = axins_top
      
    # Plot difference on bottom subplot
    ax2.plot(difference_x, difference, color='#666666', linewidth=1.2, linestyle='-')
    ax2.axhline(y=0, color='#999999', linestyle='--', linewidth=0.8)
    
    # Add horizontal grid lines to the difference plot
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Add labels and set limits for bottom subplot
    ax2.set_xlabel(r'$r$ / Å')
    ax2.set_ylabel(r'$\Delta$$g$($r$)')
    
    # Add reliability factor as text in the plot
    ax2.text(0.98, 0.3, f"$R_f$= {r_factor:.2f}%", fontsize=16, 
             transform=ax2.transAxes, ha='right', va='top')    
    
    # Calculate appropriate y-limits for difference plot
    diff_abs_max = max(abs(np.nanmin(difference)), abs(np.nanmax(difference)))
    y_limit = min(diff_abs_max * 1.2, 3)  # Cap at 3 to avoid extreme values
    ax2.set_ylim(-y_limit, y_limit)
    
    # Set x-limits for both plots
    ax1.set_xlim(0, 10)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    
    # Save the figure
    plt.savefig(output_file, format='svg', dpi=400, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Show the plot
    plt.show()


def main():
    # Define the path patterns to the RDF files
    aimd_file_pattern = './rdf_analysis_aimd/total_rdf.*.txt'
    mlip_file_pattern = './rdf_analysis_mlip/total_rdf.*.txt'
    
    # Process the AIMD RDF files
    print("Processing AIMD RDF files...")
    try:
        aimd_distances, aimd_avg_rdf = process_rdf_files(aimd_file_pattern)
        
        # Smooth the AIMD RDF data
        print("Smoothing AIMD RDF data...")
        aimd_smoothed_rdf = smooth_rdf(aimd_distances, aimd_avg_rdf)
        
        # Process the MLIP RDF files
        print("Processing MLIP RDF files...")
        mlip_distances, mlip_avg_rdf = process_rdf_files(mlip_file_pattern)
        
        # Smooth the MLIP RDF data
        print("Smoothing MLIP RDF data...")
        mlip_smoothed_rdf = smooth_rdf(mlip_distances, mlip_avg_rdf)
        
        # Plot the comparison
        print("Creating comparison plot...")
        plot_comparison_rdf(
            (aimd_distances, aimd_smoothed_rdf),
            (mlip_distances, mlip_smoothed_rdf)
        )
        
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
