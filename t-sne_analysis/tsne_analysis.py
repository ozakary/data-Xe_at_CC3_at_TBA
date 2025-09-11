import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
import seaborn as sns

# Import the figure_formatting module
import figure_formatting_v3 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def get_params_hash(params):
    """Create hash of parameters for caching"""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]

def check_cached_data(all_params, xyz_file):
    """Check if cached data exists and is valid"""
    params_hash = get_params_hash(all_params)
    cache_file = f"tsne_cache_{params_hash}.npz"
    
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
        stored_params = cached['all_params'].item()
        if stored_params == all_params:
            return cached, cache_file
        else:
            print("Cached parameters don't match. Will recompute.")
            return None, cache_file
    except Exception as e:
        print(f"Error loading cache: {e}. Will recompute.")
        return None, cache_file

def compute_cartesian_features(atoms):
    """Compute geometric/cartesian features for each atom"""
    positions = atoms.get_positions()
    n_atoms = len(atoms)
    features = []
    
    for i in range(n_atoms):
        # For each atom, compute distances to all other atoms
        distances = []
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        # Sort distances and take first 20 (or pad with zeros if fewer atoms)
        distances = sorted(distances)[:20]
        while len(distances) < 20:
            distances.append(0.0)  # Pad with zeros
        
        # Add coordination-like features
        coord_1 = sum(1 for d in distances if d < 2.0)  # Very close neighbors
        coord_2 = sum(1 for d in distances if d < 3.0)  # Close neighbors
        coord_3 = sum(1 for d in distances if d < 4.0)  # Medium neighbors
        
        # Combine all features for this atom
        atom_features = distances + [coord_1, coord_2, coord_3]
        features.append(atom_features)
    
    return np.array(features)

def compute_representations(xyz_file, soap_params, tsne_params, max_neighborhoods):
    """Compute both Cartesian and SOAP representations with t-SNE"""
    
    # Create combined parameters for caching
    all_params = {
        'soap_params': soap_params,
        'tsne_params': tsne_params,
        'max_neighborhoods': max_neighborhoods
    }
    
    # Check for cached data
    cached_data, cache_file = check_cached_data(all_params, xyz_file)
    if cached_data is not None:
        print("Using cached data!")
        return (cached_data['tsne_cartesian'], cached_data['tsne_soap'],
                cached_data['atom_types'], cached_data['structure_indices'])
    
    # Load dataset
    print("Loading xyz file...")
    start_memory = get_memory_usage()
    print(f"Initial memory usage: {start_memory:.2f} GB")
    
    atoms_list = read(xyz_file, index=":")
    print(f"Loaded {len(atoms_list)} structures")
    
    # Create SOAP descriptor
    soap = SOAP(**soap_params)
    
    print("Computing representations for individual atomic neighborhoods...")
    
    # Store data for all atomic neighborhoods
    all_cartesian_features = []
    all_soap_features = []
    atom_types = []
    structure_indices = []
    
    # Process in batches to manage memory
    batch_size = min(50, len(atoms_list))
    processed_neighborhoods = 0
    
    for batch_start in tqdm(range(0, len(atoms_list), batch_size), desc="Processing structures"):
        batch_end = min(batch_start + batch_size, len(atoms_list))
        
        for struct_idx in range(batch_start, batch_end):
            atoms = atoms_list[struct_idx]
            
            # Compute Cartesian features
            cartesian_features = compute_cartesian_features(atoms)
            
            # Compute SOAP features
            soap_features = soap.create(atoms)
            
            # Store each atomic neighborhood
            for atom_idx in range(len(atoms)):
                all_cartesian_features.append(cartesian_features[atom_idx])
                all_soap_features.append(soap_features[atom_idx])
                atom_types.append(atoms[atom_idx].symbol)
                structure_indices.append(struct_idx)
                processed_neighborhoods += 1
                
                # Stop if we've reached the maximum
                if processed_neighborhoods >= max_neighborhoods:
                    break
            
            if processed_neighborhoods >= max_neighborhoods:
                break
        
        # Memory management
        if batch_start % (batch_size * 5) == 0:
            gc.collect()
            print(f"  Processed {processed_neighborhoods} neighborhoods. Memory: {get_memory_usage():.2f} GB")
        
        if processed_neighborhoods >= max_neighborhoods:
            break
    
    print(f"Total neighborhoods collected: {processed_neighborhoods}")
    
    # Convert to numpy arrays
    cartesian_array = np.array(all_cartesian_features)
    soap_array = np.array(all_soap_features)
    atom_types = np.array(atom_types)
    structure_indices = np.array(structure_indices)
    
    print(f"Cartesian features shape: {cartesian_array.shape}")
    print(f"SOAP features shape: {soap_array.shape}")
    
    # Clear intermediate data
    del all_cartesian_features, all_soap_features, atoms_list
    gc.collect()
    
    # Standardize features
    print("Standardizing features...")
    scaler_cartesian = StandardScaler()
    scaler_soap = StandardScaler()
    
    cartesian_scaled = scaler_cartesian.fit_transform(cartesian_array)
    soap_scaled = scaler_soap.fit_transform(soap_array)
    
    del cartesian_array, soap_array
    gc.collect()
    
    # Perform t-SNE on both representations
    print("Performing t-SNE on Cartesian representation...")
    tsne_cartesian_model = TSNE(**tsne_params)
    tsne_cartesian = tsne_cartesian_model.fit_transform(cartesian_scaled)
    print(f"Cartesian t-SNE KL divergence: {tsne_cartesian_model.kl_divergence_:.4f}")
    
    del cartesian_scaled
    gc.collect()
    
    print("Performing t-SNE on SOAP representation...")
    tsne_soap_model = TSNE(**tsne_params)
    tsne_soap = tsne_soap_model.fit_transform(soap_scaled)
    print(f"SOAP t-SNE KL divergence: {tsne_soap_model.kl_divergence_:.4f}")
    
    del soap_scaled
    gc.collect()
    
    # Save to cache
    print(f"Saving data to cache: {cache_file}")
    np.savez(cache_file,
             tsne_cartesian=tsne_cartesian,
             tsne_soap=tsne_soap,
             atom_types=atom_types,
             structure_indices=structure_indices,
             all_params=all_params,
             kl_divergence_cartesian=tsne_cartesian_model.kl_divergence_,
             kl_divergence_soap=tsne_soap_model.kl_divergence_)
    
    print("Data cached successfully!")
    return tsne_cartesian, tsne_soap, atom_types, structure_indices

def plot_tsne_results(tsne_result, atom_types, title, representation_name, all_params):
    """Create t-SNE plot colored by atom type"""
    
    # Get unique atom types and assign seaborn Set1 colors
    unique_atoms = sorted(np.unique(atom_types))  # Sort for consistent colors
    colors = sns.color_palette("Set2", n_colors=len(unique_atoms))
    
    # Create atom color mapping
    atom_colors = dict(zip(unique_atoms, colors))
    
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    
    # Plot each atom type separately for clean legend
    for atom_type in unique_atoms:
        mask = atom_types == atom_type
        if np.any(mask):
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                       c=[atom_colors[atom_type]],  # Seaborn color
                       label=f'{atom_type}',# ({np.sum(mask):,} atoms)',
                       alpha=0.7,
                       s=8)  # Smaller points for better visibility with many points
    
    # Labels and title
#    ax.set_xlabel('t-SNE 1')
#    ax.set_ylabel('t-SNE 2')
#    ax.set_title('Xe@CC3@TBA')

#    ax.set_xlim(-100,100)
#    ax.set_ylim(-100,100)
    
    ax.xaxis.set_major_locator(LinearLocator(numticks=3))
    ax.yaxis.set_major_locator(LinearLocator(numticks=3))
    
    # Legend and grid
    plt.legend(frameon=False, bbox_to_anchor=(0.5, -0.05), loc='upper center', 
           ncol=3, 
           columnspacing=0.4,     # Tight column spacing
           handletextpad=0.2,     # Small gap between marker and text
           borderpad=0.2)         # Small padding around legend
    
    # Add parameter info
    soap_params = all_params['soap_params']
    tsne_params = all_params['tsne_params']
    
    param_text = (f"Total neighborhoods: {len(tsne_result)}\n"
                 f"Perplexity: {tsne_params['perplexity']}\n"
                 f"Learning rate: {tsne_params['learning_rate']}")
    
    if representation_name == 'soap':
        param_text += (f"\nr_cut: {soap_params['r_cut']}\n"
                      f"n_max: {soap_params['n_max']}, l_max: {soap_params['l_max']}")
    
#    plt.text(0.02, 0.02, param_text,
#             transform=plt.gca().transAxes,
#             verticalalignment='bottom',
#             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
#             fontsize=9)
    
#    plt.tight_layout()
    
    # Save as SVG
    params_hash = get_params_hash(all_params)
    filename = f"tsne_{representation_name}_{params_hash}.png"
    plt.savefig(filename, format='png', dpi=400, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()
    
    return filename

def analyze_tsne_results(tsne_cartesian, tsne_soap, atom_types):
    """Analyze and compare t-SNE results"""
    print("\n=== t-SNE Analysis Results ===")
    
    # Basic statistics
    print(f"Total neighborhoods analyzed: {len(tsne_cartesian)}")
    
    # Atom type distribution
    unique, counts = np.unique(atom_types, return_counts=True)
    print(f"\nAtom type distribution:")
    for atom_type, count in zip(unique, counts):
        print(f"  {atom_type}: {count} ({count/len(atom_types)*100:.1f}%)")
    
    # Spread analysis for both representations
    print(f"\nSpread analysis:")
    print(f"Cartesian representation:")
    print(f"  t-SNE 1: range=[{tsne_cartesian[:, 0].min():.2f}, {tsne_cartesian[:, 0].max():.2f}], std={tsne_cartesian[:, 0].std():.2f}")
    print(f"  t-SNE 2: range=[{tsne_cartesian[:, 1].min():.2f}, {tsne_cartesian[:, 1].max():.2f}], std={tsne_cartesian[:, 1].std():.2f}")
    
    print(f"SOAP representation:")
    print(f"  t-SNE 1: range=[{tsne_soap[:, 0].min():.2f}, {tsne_soap[:, 0].max():.2f}], std={tsne_soap[:, 0].std():.2f}")
    print(f"  t-SNE 2: range=[{tsne_soap[:, 1].min():.2f}, {tsne_soap[:, 1].max():.2f}], std={tsne_soap[:, 1].std():.2f}")
    
    # Atom type clustering analysis
    print(f"\nAtom type separation analysis:")
    for atom_type in unique:
        mask = atom_types == atom_type
        if np.any(mask) and np.sum(mask) > 1:
            # Cartesian
            cart_coords = tsne_cartesian[mask]
            cart_spread = np.sqrt(cart_coords.var(axis=0).sum())
            
            # SOAP  
            soap_coords = tsne_soap[mask]
            soap_spread = np.sqrt(soap_coords.var(axis=0).sum())
            
            print(f"  {atom_type}: Cartesian spread={cart_spread:.2f}, SOAP spread={soap_spread:.2f}")

# ========================
# MAIN EXECUTION  
# ========================

if __name__ == "__main__":
    # Configuration
    xyz_file = "../../dataset_2xe_cc3_tba.xyz"
    
    # SOAP parameters
    soap_params = {
        'species': ["H", "C", "N", "O", "F", "Xe"],
        'periodic': True,      # Use periodic boundary conditions
        'r_cut': 5,
        'n_max': 8,
        'l_max': 6,
        'sigma': 0.3
    }
    
    # t-SNE parameters
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': 200,
        'n_iter': 1000,
        'random_state': 42,
        'method': 'barnes_hut',
        'verbose': 1
    }
    
    # Limit neighborhoods for computational efficiency
    max_neighborhoods = 30000  # 15000 previous value, Adjust based on your computational resources
    
    print("t-SNE Analysis with Multiple Representations")
    print("="*50)
    print(f"XYZ file: {xyz_file}")
    print(f"SOAP parameters: {soap_params}")
    print(f"t-SNE parameters: {tsne_params}")
    print(f"Max neighborhoods: {max_neighborhoods}")
    print()
    
    # Compute or load cached data
    tsne_cartesian, tsne_soap, atom_types, structure_indices = compute_representations(
        xyz_file, soap_params, tsne_params, max_neighborhoods)
    
    # Create plots
    print("\nCreating plots...")
    
    # Combined parameters for file naming
    all_params = {
        'soap_params': soap_params,
        'tsne_params': tsne_params,
        'max_neighborhoods': max_neighborhoods
    }
    
    # Plot Cartesian representation
    cart_filename = plot_tsne_results(
        tsne_cartesian, atom_types,
        "t-SNE: Cartesian Representation", 
        "cartesian", all_params)
    
    # Plot SOAP representation  
    soap_filename = plot_tsne_results(
        tsne_soap, atom_types,
        "t-SNE: SOAP (Symmetry Function) Representation",
        "soap", all_params)
    
    # Analyze results
    analyze_tsne_results(tsne_cartesian, tsne_soap, atom_types)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Cartesian plot saved as: {cart_filename}")
    print(f"SOAP plot saved as: {soap_filename}")
    print(f"Next run with same parameters will use cached data.")
