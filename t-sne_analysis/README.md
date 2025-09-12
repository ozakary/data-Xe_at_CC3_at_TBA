# t-distributed Stochastic Neighbor Embedding (t-SNE) Analysis of the Xe@CC3@TBA DFT-2 Dataset

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the [code](./tsne_analysis.py) for performing t-SNE analysis on individual atomic neighborhoods from the DFT-2 dataset.

### Dual Representation Approach
The analysis compares two different ways to encode atomic environments:

1. **Cartesian Representation**: Geometric features based on interatomic distances
2. **SOAP Representation**: Rotationally invariant descriptors capturing chemical environments

## Input Data

### Dataset
- **File**: `dataset_2xe_cc3_tba.xyz` (to be found in the [Zenodo](https://doi.org/10.5281/zenodo.17105321) repository as `1-dataset_2xe_cc3_tba.xyz.tar.bz2`)
- **System**: Xe@CC3@TBA
- **Sampling**: Up to 30,000 neighborhoods (configurable)

### Feature Representations

#### Cartesian Features
For each atom, the code computes:
- **Distance features**: Sorted distances to 20 nearest neighbors (zero-padded)
- **Coordination numbers**: 
  - Very close neighbors (< 2.0 Å)
  - Close neighbors (< 3.0 Å) 
  - Medium neighbors (< 4.0 Å)
- **Total features**: 23 per atom (20 distances + 3 coordination numbers)

#### SOAP Features
Standard SOAP descriptor parameters:
```python
soap_params = {
    'species': ["H", "C", "N", "O", "F", "Xe"],
    'periodic': True,
    'r_cut': 5,        # Cutoff radius (Å)
    'n_max': 8,        # Radial basis functions
    'l_max': 6,        # Angular momentum
    'sigma': 0.3       # Gaussian width
}
```

## Algorithm Implementation

### Feature Computation Pipeline
1. **Structure processing**: Batch loading of XYZ configurations
2. **Cartesian features**: Distance-based geometric descriptors
3. **SOAP features**: Physics-informed local environment descriptors
4. **Standardization**: Feature scaling for both representations
5. **t-SNE embedding**: Non-linear dimensionality reduction to 2D

### Memory Management
- **Adaptive batching**: Processes structures in memory-efficient batches
- **Neighborhood sampling**: Limits total neighborhoods to prevent memory overflow
- **Garbage collection**: Explicit memory cleanup between processing steps
- **Progress monitoring**: Real-time memory usage and progress tracking

### Caching System
- **Parameter hashing**: MD5 hash of all parameters for unique identification
- **Automatic validation**: Checks file timestamps and parameter consistency
- **Complete workflow caching**: Stores both representations and t-SNE results
- **KL divergence tracking**: Records optimization quality metrics

## t-SNE Parameters

```python
tsne_params = {
    'n_components': 2,           # 2D visualization
    'perplexity': 30,            # Local neighborhood size
    'learning_rate': 200,        # Optimization step size
    'n_iter': 1000,              # Maximum iterations
    'random_state': 42,          # Reproducibility
    'method': 'barnes_hut',      # Fast approximation
    'verbose': 1                 # Progress output
}
```

### Parameter Interpretation
- **Perplexity (30)**: Balances local vs. global structure emphasis
- **Learning rate (200)**: Conservative rate for stable convergence
- **Iterations (1000)**: Sufficient for convergence on most datasets
- **Barnes-Hut method**: O(N log N) complexity for large datasets

## Visualization and Analysis

### Plots Generated
Both representations produce scatter plots showing:
- **Color coding**: Atom types (H, C, N, O, F, Xe) with distinct colors
- **Point density**: Semi-transparent points to show clustering

### Statistical Analysis
The code automatically computes:
- **Neighborhood distribution**: Count and percentage by atom type
- **Spread metrics**: Range and standard deviation for each t-SNE axis
- **Clustering assessment**: Atom type separation quality
- **KL divergence**: t-SNE optimization quality (lower is better)

### Output Files
- `tsne_cartesian_{hash}.png`: Cartesian representation plot
- `tsne_soap_{hash}.png`: SOAP representation plot  
- `tsne_cache_{hash}.npz`: Cached computation results
- Console output with detailed statistics

## Usage

### Basic Execution
```bash
python tsne_analysis.py
```

### Configuration Options
Modify key parameters in the script:

```python
# Computational limits
max_neighborhoods = 30000    # Reduce for faster computation

# t-SNE parameters
tsne_params['perplexity'] = 50      # Try 15-50 range
tsne_params['learning_rate'] = 300   # Adjust if poor convergence

# SOAP parameters  
soap_params['r_cut'] = 6            # Larger cutoff radius
soap_params['n_max'] = 10           # More radial resolution
```

### Performance Scaling
- **Small datasets** (< 10,000 neighborhoods): Fast processing
- **Medium datasets** (10,000-50,000): Moderate computation time
- **Large datasets** (> 50,000): Consider reducing `max_neighborhoods`

## Dependencies

### Required Packages
```python
import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import psutil
```

### Installation
```bash
pip install dscribe ase scikit-learn matplotlib seaborn tqdm psutil
```

### System Requirements
- **Python 3.7+**

---

For further details, please refer to the respective folders or contact the author via the provided email.	
