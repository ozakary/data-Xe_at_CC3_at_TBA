import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gc
import psutil
import os
import json
import hashlib

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LinearLocator


# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def get_soap_params_hash(soap_params):
    """Create hash of SOAP parameters for caching"""
    params_str = json.dumps(soap_params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]

def check_cached_data(soap_params, xyz_file):
    """Check if cached data exists and is valid"""
    params_hash = get_soap_params_hash(soap_params)
    cache_file = f"soap_pca_cache_{params_hash}.npz"
    
    if not os.path.exists(cache_file):
        print(f"No cached data found for these parameters.")
        return None, cache_file
    
    # Check if xyz file is newer than cache
    xyz_mtime = os.path.getmtime(xyz_file)
    cache_mtime = os.path.getmtime(cache_file)
    
    if xyz_mtime > cache_mtime:
        print(f"XYZ file is newer than cache. Will recompute.")
        return None, cache_file
    
    print(f"Loading cached data from {cache_file}...")
    try:
        cached = np.load(cache_file, allow_pickle=True)
        stored_params = cached['soap_params'].item()
        if stored_params == soap_params:
            return cached, cache_file
        else:
            print("Cached parameters don't match. Will recompute.")
            return None, cache_file
    except Exception as e:
        print(f"Error loading cache: {e}. Will recompute.")
        return None, cache_file

def estimate_soap_memory(atoms_list, soap):
    """Estimate memory requirements for SOAP computation"""
    test_soap = soap.create(atoms_list[0])
    features_per_atom = test_soap.shape[1]
    total_atoms = sum(len(atoms) for atoms in atoms_list)
    estimated_gb = (total_atoms * features_per_atom * 8) / (1024**3)
    
    print(f"SOAP descriptor info:")
    print(f"  Features per atom: {features_per_atom}")
    print(f"  Total atoms: {total_atoms:,}")
    print(f"  Estimated memory for all SOAP descriptors: {estimated_gb:.2f} GB")
    
    return estimated_gb

def compute_soap_pca(xyz_file, soap_params):
    """Compute SOAP descriptors and PCA"""
    
    # Check for cached data
    cached_data, cache_file = check_cached_data(soap_params, xyz_file)
    if cached_data is not None:
        print("Using cached data!")
        return cached_data['pca_result'], cached_data['explained_variance_ratio'], cached_data['n_structures']
    
    # Load dataset
    print("Loading xyz file...")
    start_memory = get_memory_usage()
    print(f"Initial memory usage: {start_memory:.2f} GB")
    
    atoms_list = read(xyz_file, index=":")
    print(f"Loaded {len(atoms_list)} structures")
    print(f"Memory after loading: {get_memory_usage():.2f} GB")
    
    # Create SOAP descriptor
    soap = SOAP(**soap_params)
    
    # Estimate memory and determine batch size
    estimated_memory = estimate_soap_memory(atoms_list, soap)
    available_memory = psutil.virtual_memory().available / (1024**3)
    print(f"Available system memory: {available_memory:.2f} GB")
    
    if estimated_memory > available_memory * 0.5:
        recommended_batch = max(10, int(len(atoms_list) * (available_memory * 0.3) / estimated_memory))
        batch_size = min(50, recommended_batch)
        print(f"Large dataset detected. Using batch size: {batch_size}")
    else:
        batch_size = 100
        print(f"Using batch size: {batch_size}")
    
    # Compute SOAP descriptors in batches
    print("\nComputing SOAP descriptors in batches...")
    start_time = time.time()
    soap_averaged_list = []
    processed_structures = 0
    
    for batch_start in tqdm(range(0, len(atoms_list), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(atoms_list))
        batch_atoms = atoms_list[batch_start:batch_end]
        
        batch_soap_averaged = []
        for atoms in batch_atoms:
            soap_desc = soap.create(atoms)
            soap_avg = soap_desc.mean(axis=0)
            batch_soap_averaged.append(soap_avg)
            processed_structures += 1
        
        soap_averaged_list.extend(batch_soap_averaged)
        del batch_soap_averaged, batch_atoms
        gc.collect()
        
        if (batch_start // batch_size + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (batch_start // batch_size + 1)
            remaining_batches = (len(atoms_list) - batch_end) / batch_size
            remaining_time = remaining_batches * avg_time_per_batch
            current_memory = get_memory_usage()
            print(f"  Processed {processed_structures}/{len(atoms_list)} structures. "
                  f"Memory: {current_memory:.2f} GB. "
                  f"Est. time remaining: {remaining_time/60:.1f} min")
    
    # Convert to numpy array and perform PCA
    print("Converting to numpy array...")
    soap_array = np.array(soap_averaged_list)
    print(f"Final SOAP descriptor shape: {soap_array.shape}")
    
    del soap_averaged_list, atoms_list
    gc.collect()
    
    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(soap_array)
    
    print(f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Save to cache
    print(f"Saving data to cache: {cache_file}")
    np.savez(cache_file,
             pca_result=pca_result,
             explained_variance_ratio=pca.explained_variance_ratio_,
             soap_params=soap_params,
             n_structures=len(pca_result),
             soap_array=soap_array)  # Also save SOAP array for potential future use
    
    print(f"Data cached successfully!")
    return pca_result, pca.explained_variance_ratio_, len(pca_result)

def plot_pca_results(pca_result, explained_variance_ratio, n_structures, soap_params):
    """Create and display PCA scatter plot"""
    
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    
    # Create simple scatter plot with single color
    plt.scatter(pca_result[:, 0], pca_result[:, 1], 
               c='#E76F51', 
               alpha=0.4, 
               s=20)
    
    # Labels and title
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
#    ax.set_title('Xe@CC3@TBA')
    
    ax.set_xlim(-0.12,0.12)
    ax.set_ylim(-0.12,0.12)
    
    ax.xaxis.set_major_locator(LinearLocator(numticks=3))
    ax.yaxis.set_major_locator(LinearLocator(numticks=3))
   
    # Add parameter info box
    param_text = (f"Structures: {n_structures}\n"
                 f"Total variance: {explained_variance_ratio.sum():.1%}\n"
                 f"r_cut: {soap_params['r_cut']}\n"
                 f"n_max: {soap_params['n_max']}, l_max: {soap_params['l_max']}")
    
    plt.text(0.02, 0.98, param_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
#    plt.tight_layout()
    
    # Save plot as SVG
    params_hash = get_soap_params_hash(soap_params)
    plot_filename = f"pca_plot_{params_hash}.svg"
    plt.savefig(plot_filename, format='svg', bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    plt.show()
    
    return plot_filename

def analyze_outliers(pca_result):
    """Analyze and print outlier statistics"""
    print("\n=== Outlier Analysis ===")
    pc1_mean, pc1_std = pca_result[:,0].mean(), pca_result[:,0].std()
    pc2_mean, pc2_std = pca_result[:,1].mean(), pca_result[:,1].std()
    
    print(f"PC1: mean={pc1_mean:.3f}, std={pc1_std:.3f}")
    print(f"PC2: mean={pc2_mean:.3f}, std={pc2_std:.3f}")
    
    for threshold in [2, 2.5, 3]:
        pc1_outliers = np.abs(pca_result[:,0] - pc1_mean) > threshold * pc1_std
        pc2_outliers = np.abs(pca_result[:,1] - pc2_mean) > threshold * pc2_std
        outliers = pc1_outliers | pc2_outliers
        n_outliers = np.sum(outliers)
        
        print(f"Outliers (>{threshold}σ): {n_outliers} ({n_outliers/len(pca_result)*100:.1f}%)")
        
        if n_outliers > 0 and n_outliers < 20:
            outlier_indices = np.where(outliers)[0]
            print(f"  Indices: {outlier_indices}")

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    # Configuration
    xyz_file = "../../dataset_2xe_cc3_tba.xyz"
    
    # SOAP parameters - modify these to trigger recomputation
    soap_params = {
        'species': ["H", "C", "N", "O", "F", "Xe"],
        'periodic': True,
        'r_cut': 5,      # Change this to recompute
        'n_max': 8,        # Change this to recompute  
        'l_max': 6,        # Change this to recompute
        'sigma': 0.3
    }
    
    print("SOAP PCA Analysis with Caching")
    print("="*40)
    print(f"XYZ file: {xyz_file}")
    print(f"SOAP parameters: {soap_params}")
    print()
    
    # Compute or load cached data
    pca_result, explained_variance_ratio, n_structures = compute_soap_pca(xyz_file, soap_params)
    
    # Create plot
    plot_filename = plot_pca_results(pca_result, explained_variance_ratio, n_structures, soap_params)
    
    # Analyze outliers
    analyze_outliers(pca_result)
    
    print(f"\nAnalysis complete!")
    print(f"Plot saved as: {plot_filename}")
    print(f"Next time you run with the same parameters, cached data will be used.")
