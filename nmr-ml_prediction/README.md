# Prediction of <sup>129</sup>Xe σ from Pre-trained NMR-ML Model for Xe@CC3@TBA

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)
---

This directory contains the scripts and workflow for predicting <sup>129</sup>Xe NMR magnetic shielding tensors using the pre-trained MatTen model.

## Directory Structure

```
./
├── dataset_test_structures.xyz        # Input structures for prediction
├── predict_atomic_tensor.py           # Main prediction script
├── script_predict.job                 # SLURM job submission script
├── output_prediction.csv              # Prediction results
├── matten.log                         # Application log
├── nmr-ml_matten-output_*.txt         # SLURM stdout
└── nmr-ml_matten-errors_*.txt         # SLURM stderr
```

## Input Data Format

### XYZ Structure Files (`dataset_test_structures.xyz` to be found in the [Zenodo](./) repository)
The input file contains multiple molecular structures in extended XYZ format:

```
1170
Lattice="23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077" Properties=species:S:1:pos:R:3
O  23.4449080892 19.9457264109 16.8624230406
C  23.6869558502 13.4755800908 15.7803088909
...
Xe  6.82709 11.3068 8.4432
...
```

## Prediction Script ([`predict_atomic_tensor.py`](./predict_atomic_tensor.py))

### Core Functionality
The script provides a complete pipeline for NMR tensor prediction:

#### XYZ File Parsing
```python
def parse_xyz_file(xyz_file):
    """Parse multiple structures and extract lattice information"""
    # - Reads multi-frame XYZ files
    # - Extracts lattice parameters from comment lines
    # - Handles variable structure sizes
    # - Provides progress monitoring
```

#### Structure Conversion
```python
def create_pymatgen_structure(structure_info):
    """Convert to PyMatGen Structure objects"""
    # - Creates periodic structures with proper lattice
    # - Handles Cartesian coordinate conversion
    # - Maintains atomic species information
```

#### NMR Tensor Prediction
```python
def predict_nmr_tensors(structures, model_path, checkpoint_name):
    """Predict tensors for all Xe atoms"""
    # - Identifies Xe atoms in each structure
    # - Applies trained MatTen model
    # - Extracts symmetric tensor components
    # - Calculates isotropic shielding values
```

### Model Configuration
```python
# Default model paths (adjust as needed)
model_path = "/path/to/matten_logs/checkpoints/"
checkpoint_name = "last.ckpt"  # or "best.ckpt"
```

### Key Features
- **Automatic Xe detection**: Identifies xenon atoms for prediction
- **Batch processing**: Handles multiple structures efficiently
- **Error handling**: Graceful failure recovery for problematic structures
- **Progress monitoring**: Real-time processing updates
- **Tensor symmetrization**: Ensures physical validity of predictions

## Job Submission ([`script_predict.job`](./script_predict.job))

### SLURM Configuration
```bash
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=8        # 8 CPU cores
#SBATCH --time=08:00:00            # 8-hour time limit
#SBATCH --account=project_462000738
#SBATCH --exclusive                # Exclusive node access
```

### Environment Setup
```bash
module load cray-python
source /path/to/matten_env/bin/activate
ulimit -s unlimited               # Unlimited stack size
```

### Execution Command
```bash
python3 predict_atomic_tensor.py dataset_test_structures.xyz output_prediction.csv
```

## Output Format

### Prediction Results (`output_prediction.csv` to be found in the [Zenodo](./) repository)
The output CSV contains comprehensive NMR data for each Xe atom:

| Column | Description | Units | Example |
|--------|-------------|--------|---------|
| structure_id | Frame index | - | 0, 1, 2, ... |
| atom_index | Atom index within structure | - | 265, 918, ... |
| element | Chemical element | - | Xe |
| x, y, z | Cartesian coordinates | Å | 6.827, 11.307, 8.443 |
| sigma_iso | Isotropic shielding | ppm | 5697.37 |
| tensor_xx, tensor_xy, ... | Full tensor components | ppm | 5694.35, -20.91, ... |

### Tensor Components
The full 3×3 magnetic shielding tensor is stored as:
```
σ = | tensor_xx  tensor_xy  tensor_xz |
    | tensor_yx  tensor_yy  tensor_yz |
    | tensor_zx  tensor_zy  tensor_zz |
```

### Isotropic Shielding Calculation
```python
sigma_iso = (tensor_xx + tensor_yy + tensor_zz) / 3.0
```

## Usage Instructions

### 1. Model Preparation
Ensure the trained MatTen model is available:
```bash
# Verify model checkpoint exists
ls /path/to/matten_logs/checkpoints/last.ckpt
```

### 2. Input Structure Preparation
Prepare XYZ file with structures to predict:
- MD trajectory snapshots
- Test set structures
- New molecular configurations

### 3. Configuration
Update model path in `predict_atomic_tensor.py`:
```python
model_path = "/your/model/path/"
checkpoint_name = "best.ckpt"  # or "last.ckpt"
```

### 4. Job Submission
```bash
# Submit prediction job
sbatch script_predict.job

# Monitor progress
tail -f nmr-ml_matten-output_*.txt
```

---

For further details, please refer to the respective folders or contact the author via the provided email.	
