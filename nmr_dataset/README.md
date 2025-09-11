# Dataset Formatting for MatTen Architecture

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)
---

This directory contains the workflow for processing DFT-calculated <sup>129</sup>Xe NMR magnetic shielding tensors and converting them into the JSON format required by the *MatTen* architecture.

## Overview

The dataset formatting process transforms raw TURBOMOLE NMR calculation outputs into a standardized machine learning dataset through a multi-stage pipeline that includes data extraction, tensor alignment, multi-level theory combination, and JSON formatting for the MatTen neural network architecture.

## Computational Strategy: Multi-Level Theory Combination

### Theory Level Combination Formula
The final magnetic shielding tensor is computed using:
```
σ_final = σ_BHandHLYP/SVP + σ_PBE/TZVP - σ_PBE/SVP
```

## Workflow Pipeline

### Stage 1: Individual Theory Level Processing

#### For Each Theory Level Directory
```
./BHandHLYP_SVP/FINISHED/
./PBE_TZVP/FINISHED/
./PBE_SVP/FINISHED/
```

#### Step 1.1: Data Extraction ([`ml_nmr_schnet_dataset_oz-t.py`](./ml_nmr_schnet_dataset_oz-t.py))
```bash
cd ./[THEORY_LEVEL]/FINISHED/
python3 ml_nmr_schnet_dataset_oz-t.py
```

**Functionality:**
- Extracts atomic coordinates from `coord_${i}.xyz` files
- Parses magnetic shielding tensors from `mpshift.out` files
- Handles missing tensor data with zero-padding
- Creates molecule-wise CSV files for structures and tensors
- Generates concatenated datasets

**Key Features:**
- **Tensor extraction**: Parses TURBOMOLE output format
- **Error handling**: Zero tensors for atoms without calculated shielding
- **Data validation**: Tracks missing calculations and provides statistics
- **File organization**: Individual and concatenated CSV outputs

**Output Files:**
```
dataset_schnet_shielding_tensors/
├── Xe_TBA_*.csv              # Individual molecule tensors
└── magnetic_shielding_tensors.csv  # Concatenated tensor data

dataset_schnet_atomic_coordinates/
├── Xe_TBA_*.csv              # Individual molecule coordinates
└── structures.csv            # Concatenated coordinate data

train.csv                     # Training set identifiers
```

#### Step 1.2: Tensor Alignment ([`alignements.py`](./alignements.py))
```bash
python3 alignements.py
```

**Purpose**: Aligns tensor data with atomic structure data, ensuring consistent atom indexing.

**Process:**
1. Matches molecules between structure and tensor files
2. Identifies Xe atom positions from structure data
3. Places non-zero tensors at correct Xe positions
4. Fills remaining positions with zero tensors
5. Ensures consistent atom_index mapping

**Output**: `magnetic_shielding_tensors_modified.csv`

#### Step 1.3: Data Verification ([`grep_commands_verif.sh`](./grep_commands_verif.sh))
```bash
./grep_commands_verif.sh
```

**Quality Control Checks:**
```bash
# Remove zero-tensor entries for verification
grep -vE ',(0\.0|0)(,|$)' dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv > ms_verif.csv

# Extract Xe-only structures for validation
(head -n 1 dataset_schnet_atomic_coordinates/structures.csv && grep ',Xe,' dataset_schnet_atomic_coordinates/structures.csv) > str_verif.csv
```

#### Step 1.4: Cluster Validation ([`info.sh`](./info.sh))
```bash
./info.sh ./
```

**Verification**: Lists all processed cluster directories and validates completeness.

### Stage 2: Multi-Level Theory Combination ([`final_magnetic_shielding.py`](./final_magnetic_shielding.py))

#### Tensor Algebra Operation
```python
# Combine three levels of theory
result_col = file1[col] + file2[col] - file3[col]
```

**Input Files:**
- `./BHandHLYP_SVP/FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv`
- `./PBE_TZVP/FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv`  
- `./PBE_SVP/FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv`

**Process:**
1. Validates file consistency (same dimensions and identifiers)
2. Preserves molecule_name and atom_index columns
3. Applies tensor combination formula to all 9 tensor components
4. Outputs combined tensor data

**Output**: `magnetic_shielding_tensors.csv`

### Stage 3: Dataset Integration ([`code_data.py`](./code_data.py))

#### Isotropic Shielding Calculation
```python
# Symmetrize tensor components
tensor_components = 0.5 * (tensor_components + np.transpose(tensor_components, (0, 2, 1)))

# Compute eigenvalues and isotropic shielding
w, _ = np.linalg.eigh(tensor_components)
sigma_iso = np.mean(w, axis=1)
```

**Process:**
1. Loads structure and tensor data
2. Reshapes tensor components to 3×3 matrices
3. Symmetrizes tensors to ensure physical validity
4. Computes eigenvalues and derives isotropic shielding
5. Merges sigma_iso with structural data

**Input Files:**
- `structures.csv` (from PBE_SVP level)
- `magnetic_shielding_tensors.csv` (combined data)

**Output**: `structures_with_sigma_iso.csv`

### Stage 4: JSON Conversion for MatTen ([`csv_to_json_converter_enhanced.py`](./csv_to_json_converter_enhanced.py))

#### Configuration Setup ([`config.txt`](./config.txt))
```
../structures_with_sigma_iso.csv,../magnetic_shielding_tensors.csv,23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077
```

#### Execution
```bash
cd ./split_80-10-10/
python csv_to_json_converter_enhanced.py config.txt
```

#### JSON Structure for MatTen
```json
{
  "sigma_iso": {"0": [values], "1": [values], ...},
  "Qn": {"0": [values], "1": [values], ...},
  "structure": {"0": pymatgen_structure, "1": pymatgen_structure, ...},
  "species": {"0": [atomic_numbers], "1": [atomic_numbers], ...},
  "nmr_tensor": {"0": [3x3_matrices], "1": [3x3_matrices], ...},
  "atom_selector": {"0": [booleans], "1": [booleans], ...}
}
```

#### Advanced Features

**Tensor Symmetry Validation:**
- Automatic detection of asymmetric tensors
- Symmetrization via averaging: `T_sym = (T + T^T) / 2`
- Recalculation of σ_iso for corrected tensors
- Statistical reporting of correction rates

**Dataset Splitting:**
- **Training**: 80% of configurations
- **Validation**: 10% of configurations  
- **Testing**: 10% of configurations
- Random state seeding for reproducibility

**Quality Assurance:**
- PyMatGen structure validation
- Lattice parameter consistency
- Fractional coordinate conversion
- Species mapping to atomic numbers

#### Output Files
```
dataset_train.json                              # Training set (JSON)
dataset_val.json                               # Validation set (JSON)
dataset_test.json                              # Test set (JSON)
dataset_test_structures.xyz                    # Test structures (XYZ)
structures_with_sigma_iso_and_tensors.csv     # Test data (CSV)
```

## Data Flow Summary

```
TURBOMOLE Calculations
├── BHandHLYP/SVP/FINISHED/cluster_*/
├── PBE/TZVP/FINISHED/cluster_*/
└── PBE/SVP/FINISHED/cluster_*/
    ↓
Individual CSV Extraction (ml_nmr_schnet_dataset_oz-t.py)
    ↓
Tensor Alignment (alignements.py)
    ↓
Multi-Level Combination (final_magnetic_shielding.py)
    ↓
Isotropic Shielding Integration (code_data.py)
    ↓
JSON Formatting for MatTen (csv_to_json_converter_enhanced.py)
    ↓
Training-Ready Dataset
├── dataset_train.json
├── dataset_val.json
└── dataset_test.json
```

## File Dependencies

### Required Input Files
- TURBOMOLE calculation outputs in `cluster_*/` directories
- Coordinate files: `coord_${i}.xyz`
- NMR output files: `mpshift.out`

### Intermediate Files
- `magnetic_shielding_tensors_modified.csv` (per theory level)
- `structures.csv` (atomic coordinates)
- `magnetic_shielding_tensors.csv` (combined tensors)
- `structures_with_sigma_iso.csv` (integrated data)

### Final Output Files
- JSON datasets for MatTen training
- XYZ structures for MatTen testing
- CSV data for analysis

## Quality Control Measures

### Data Validation
- **Tensor symmetry**: Automatic detection and correction
- **Missing data**: Zero-padding for incomplete calculations
- **Consistency checks**: Molecule and atom index alignment
- **Statistical reporting**: Comprehensive processing summaries

### Error Handling
- **Missing files**: Graceful handling with warnings
- **Incomplete calculations**: Zero tensors for missing data
- **Format validation**: Structural consistency checks
- **Memory management**: Efficient processing of large datasets

## Usage Instructions

### Complete Workflow Execution
```bash
# Step 1: Process each theory level
for theory in BHandHLYP_SVP PBE_TZVP PBE_SVP; do
    cd ./${theory}/FINISHED/
    python3 ml_nmr_schnet_dataset_oz-t.py
    python3 alignements.py
    ./grep_commands_verif.sh
    ./info.sh ./
    cd ../../
done

# Step 2: Combine theory levels
python3 final_magnetic_shielding.py

# Step 3: Integrate with structures
python3 code_data.py

# Step 4: Convert to JSON format
cd ./split_80-10-10/
python csv_to_json_converter_enhanced.py config.txt
```

### Configuration Options
- Modify split ratios in converter script
- Adjust lattice parameters in config.txt
- Configure symmetrization tolerance levels

---

For further details, please refer to the respective folders or contact the author via the provided email.	
