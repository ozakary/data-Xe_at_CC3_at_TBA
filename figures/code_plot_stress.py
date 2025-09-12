import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, Dict, List, Any


# Read the CSV file
def plot_metrics(csv_file):
    # Read the CSV file, handling the whitespace in the headers
    df = pd.read_csv(csv_file, skipinitialspace=True)
    
    # Create a figure and axis
    plt.figure(figsize=(10, 5))
    
    # Plot the metrics on log-log scale
    plt.loglog(df['epoch'], df['training_loss_stress'], '-', color='#264653', label=r'$\boldsymbol{s}_{\text{Loss}}^{\text{T}}$')   
    plt.loglog(df['epoch'], df['training_stress_mae'], '-', color='#2A9D8F', label=r'$\boldsymbol{s}_{\text{MAE}}^{\text{T}}$')
    plt.loglog(df['epoch'], df['training_stress_rmse'], '-', color='#E76F51', label=r'$\boldsymbol{s}_{\text{RMSE}}^{\text{T}}$')
    plt.loglog(df['epoch'], df['validation_loss_stress'], '--', color='#264653', label=r'$\boldsymbol{s}_{\text{Loss}}^{\text{V}}$')   
    plt.loglog(df['epoch'], df['validation_stress_mae'], '--', color='#2A9D8F', label=r'$\boldsymbol{s}_{\text{MAE}}^{\text{V}}$')
    plt.loglog(df['epoch'], df['validation_stress_rmse'], '--', color='#E76F51', label=r'$\boldsymbol{s}_{\text{RMSE}}^{\text{V}}$') 
    plt.loglog(df['epoch'], df['LR']*0.00001, '-', color='#E9C46A', label=r'$LR \times 10^{-6}$') 

    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, fontsize=16, facecolor="white", edgecolor="gray", framealpha=0.8)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig("plot_ml_process_stress.svg", bbox_inches='tight')
    plt.show()

# Call the function with your CSV file
plot_metrics('../../MLP-2Xe_CC3_TBA_new_output/2Xe_CC3_TBA_new_mlp_vf/metrics_epoch.csv')  # Replace with your actual file name
