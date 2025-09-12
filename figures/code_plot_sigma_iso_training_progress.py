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
def plot_metrics(csv_file_1, csv_file_2, csv_file_3):
    # Read the CSV file, handling the whitespace in the headers
    df_1 = pd.read_csv(csv_file_1, skipinitialspace=True)
    df_2 = pd.read_csv(csv_file_2, skipinitialspace=True)
    df_3 = pd.read_csv(csv_file_3, skipinitialspace=True)
       
    # Create a figure and axis
    plt.figure(figsize=(10, 5))
    
    # Plot the metrics on log-log scale
    plt.loglog(df_1['crimson-galaxy-25 - _step'], df_1['crimson-galaxy-25 - train/loss/nmr_tensor'], '-', color='#2A9D8F', label=r'$\sigma_{\text{Loss}}^{\text{T}}$')   
    plt.loglog(df_2['crimson-galaxy-25 - _step'], df_2['crimson-galaxy-25 - val/loss/nmr_tensor'], '--', color='#E76F51', label=r'$\sigma_{\text{Loss}}^{\text{V}}$')
    plt.loglog(df_3['crimson-galaxy-25 - _step'], df_3['crimson-galaxy-25 - metric_val/MeanAbsoluteError/nmr_tensor'], '--', color='#264653', label=r'$\sigma_{\text{MAE}}^{\text{V}}$')
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, fontsize=16, facecolor="white", edgecolor="gray", framealpha=0.8)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig("plot_ml_process_sigma_iso.svg", bbox_inches='tight')
    plt.show()

# Call the function with your CSV file
plot_metrics('./training_loss_nmr_tensor.csv', 'validation_loss_nmr_tensor.csv', 'validation_mae_nmr_tensor.csv')  # Replace with your actual file name
