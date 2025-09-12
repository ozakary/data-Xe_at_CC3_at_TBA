import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, Dict, List, Any

def create_joint_plot(x_data, y_data, xlabel, ylabel, filename, scale_factor=1.0):
    """Create and save a joint plot with marginal distributions.
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        scale_factor: Factor by which data has been scaled for display (default=1.0)
    """
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
    
    # Create 4 evenly spaced ticks
    x_ticks = np.linspace(x_min, x_max, 4)
    y_ticks = np.linspace(y_min, y_max, 4)
    
    g.ax_joint.set_xticks(x_ticks)
    g.ax_joint.set_yticks(y_ticks)
 
    # Format tick labels to 3 decimal places
    g.ax_joint.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    g.ax_joint.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
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
    
    # Display the statistics (raw data)
    stats_text = r'R$^{2}$' + f'={r_value**2:.4f}\nRMSE = {rmse:.1f}\nMAE = {mae:.1f}'
    g.ax_joint.text(0.05, 0.75, stats_text,
                   transform=g.ax_joint.transAxes)
 
    # Add regression line (for display data)
    slope_display, intercept_display, _, _, _ = stats.linregress(x_data, y_data)
    line_x = np.array([min(x_data), max(x_data)])
    line_y = slope_display * line_x + intercept_display
    g.ax_joint.plot(line_x, line_y, alpha=1, label=f'y = {slope_display:.4f}x + {intercept_display:.4f}', color='#264653')
#    g.ax_joint.legend(loc='lower right')

    # Plot identity line (x=y) for reference
    min_val = min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0])
    max_val = max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    # Save the plot as SVG instead of PNG
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.savefig(filename.replace('.svg', '.png'), format='png', dpi=360, bbox_inches='tight')
    plt.close()

# CSV export function removed as it's not needed

def main():
    # Read the CSV file
    print("Reading NMR data from CSV...")
    df = pd.read_csv('matched_nmr_data.csv', header=None)
    
    # Assuming format: Element, DFT value, ML value
    # Extract DFT and ML values
    dft_nmr = df[15].values  # 6th column (DFT values)
    ml_nmr = df[25].values   # 16th column (ML values)
    
    print(f"Read {len(dft_nmr)} data points")
    
    # No scaling needed for this data - using scale_factor=1.0
    create_joint_plot(
        dft_nmr, ml_nmr,
        r'$^{129}$Xe $\sigma_{zz}^{\text{DFT}}$ / ppm', r'$^{129}$Xe $\sigma_{zz}^{\text{ML}}$ / ppm',
        'comparison_sigma_zz.svg',
        scale_factor=1.0
    )
    
    print("\nAnalysis complete. Plot has been saved as 'comparison_sigma_zz.svg' and 'comparison_sigma_zz.png'.")

if __name__ == "__main__":
    main()
