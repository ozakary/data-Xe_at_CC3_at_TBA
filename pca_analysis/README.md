# Principal Component Analysis (PCA) of Structural Diversity of Xe@CC3@TBA DFT-2 Dataset

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the [code](./pca_analysis.py) for performing Principal Component Analysis (PCA) on the DFT-2 dataset using Smooth Overlap of Atomic Positions (SOAP) descriptors.

## Code Features

### Memory Management
- **Batching**: Automatically adjusts batch size based on available memory
- **Memory monitoring**: Tracks RAM usage throughout computation
- **Garbage collection**: Explicit memory cleanup to handle large datasets

### Caching System
- **Automatic caching**: Stores computed SOAP descriptors and PCA results
- **Hash-based validation**: Uses parameter hashing to ensure cache validity
- **Timestamp checking**: Recomputes if input data is newer than cache
- **Parameter tracking**: Invalidates cache when SOAP parameters change

### Progress Monitoring
- **Real-time progress**: Shows processing status with time estimates
- **Memory usage tracking**: Displays current memory consumption
- **Batch processing**: Handles large datasets efficiently

## Input Data

### Dataset File
- **File**: `dataset_2xe_cc3_tba.xyz` (to be found in the [Zenodo](https://doi.org/10.5281/zenodo.17105321) repository as `1-dataset_2xe_cc3_tba.xyz.tar.bz2`)
- **Format**: Extended XYZ with atomic coordinates and metadata
- **System**: Xe@CC3@TBA

### SOAP Parameters
```python
soap_params = {
    'species': ["H", "C", "N", "O", "F", "Xe"],
    'periodic': True,
    'r_cut': 5,        # Cutoff radius (Å)
    'n_max': 8,        # Number of radial basis functions
    'l_max': 6,        # Maximum angular momentum
    'sigma': 0.3       # Gaussian smearing width
}
```

## Output Analysis

### PCA Visualization
The code generates a scatter plot showing:
- **PC1 vs PC2**: First two principal components

### Statistical Analysis
- **Explained variance ratio**: Contribution of each principal component
- **Outlier detection**: Identification of structures beyond 2σ, 2.5σ, and 3σ thresholds
- **Data distribution**: Mean and standard deviation for each PC

### Output Files
- **SVG plot**: `pca_plot_{hash}.svg`
- **Cache file**: `soap_pca_cache_{hash}.npz`
- **Console output**: Detailed statistics and analysis results

## Usage

### Basic Execution
```bash
python pca_analysis.py
```

### Modifying Parameters
To recompute with different SOAP parameters, edit the `soap_params` dictionary:
```python
soap_params = {
    'r_cut': 6,        # Increase cutoff radius
    'n_max': 10,       # More radial functions
    'l_max': 8,        # Higher angular resolution
    # ... other parameters
}
```

### Memory Requirements
The script automatically estimates memory requirements and adjusts processing accordingly:
- **Small datasets** (< 50% available RAM): Process in large batches
- **Large datasets** (> 50% available RAM): Use smaller batches with progress monitoring

## Dependencies

### Required Packages
```python
import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import hashlib
```

### Installation
```bash
pip install dscribe ase scikit-learn matplotlib tqdm psutil
```

### System Requirements
- **Python 3.7+**

---

For further details, please refer to the respective folders or contact the author via the provided email.
