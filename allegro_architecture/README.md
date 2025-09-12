# *Allegro* Training, Validation, and Testing

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the configuration files and job scripts for training, validating, and testing the Allegro for the DFT-2 dataset.

## Overview

## System Description

### Target System
- **System**: 2Xe@CC3@TBA
- **Training set**: 801 configurations
- **Validation set**: 100 configurations
- **Test set**: Remaining configurations from dataset (100)
- **Properties predicted**: Energies, forces, and stress tensors

## Model Architecture Configuration

### Network Architecture ([`2xe_cc3_tba-config_new_vf.yaml`](./2xe_cc3_tba-config_new_vf.yaml))

#### Core Allegro Parameters
```yaml
# Symmetry and basis
l_max: 2                          # Angular momentum cutoff
parity: o3_full                   # Full O(3) equivariance
r_max: 6                          # Cutoff radius (Å)
avg_num_neighbors: auto           # Automatic neighbor estimation

# Radial basis
BesselBasis_trainable: true       # Trainable radial basis
PolynomialCutoff_p: 8            # Polynomial cutoff order

# Network layers
num_layers: 4                     # Number of Allegro layers
env_embed_multiplicity: 16        # Environment embedding size
```

#### Multi-Layer Perceptron (MLP) Settings
```yaml
# Two-body interaction MLP
two_body_latent_mlp_latent_dimensions: [32, 64, 128]
two_body_latent_mlp_nonlinearity: silu
two_body_latent_mlp_initialization: uniform

# Latent space MLP
latent_mlp_latent_dimensions: [128]
latent_mlp_nonlinearity: silu
latent_resnet: true               # Residual connections

# Edge energy MLP
edge_eng_mlp_latent_dimensions: [128]
edge_eng_mlp_nonlinearity: null   # Linear final layer
```

#### Chemical Species Mapping
```yaml
chemical_symbol_to_type:
  O: 0    # Oxygen
  C: 1    # Carbon  
  H: 2    # Hydrogen
  F: 3    # Fluorine
  Xe: 4   # Xenon
  N: 5    # Nitrogen
```

## Training Configuration

### Optimization Settings
```yaml
# Training parameters
max_epochs: 10000
learning_rate: 0.002
batch_size: 1                     # Small batch for large systems
train_val_split: random
shuffle: true

# Exponential moving average
use_ema: true
ema_decay: 0.99
ema_use_num_updates: true
```

### Loss Function and Metrics
```yaml
# Multi-property loss coefficients
loss_coeffs:
  forces: 1.0                     # Force loss weight
  stress: 1.0                     # Stress loss weight
  total_energy:
    - 1.0                         # Energy loss weight
    - PerAtomMSELoss             # Per-atom energy normalization
```

### Metrics Tracking
The model tracks multiple error metrics:
- **Forces**: MAE and RMSE
- **Stress**: MAE and RMSE (full tensor and per-component)
- **Total Energy**: MAE and RMSE (absolute and per-atom)

### Learning Rate Scheduling
```yaml
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50         # Epochs before LR reduction
lr_scheduler_factor: 0.5          # LR reduction factor
early_stopping_lower_bounds:
  LR: 1.0e-5                     # Minimum learning rate
early_stopping_patiences:
  validation_loss: 100           # Early stopping patience
```

## Job Scripts and Execution

### Training Job ([`mlip-train_E-F-S.job`](./mlip-train_E-F-S.job))

#### SLURM Configuration
```bash
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1,nvme:10   # A100 GPU with NVMe storage
#SBATCH --cpus-per-task=10
#SBATCH --time=36:00:00             # 36-hour time limit
```

#### Environment Setup
- **PyTorch**: Version 1.13 with GPU support
- **Virtual environment**: Allegro/NequIP installation

#### Training Command
```bash
srun nequip-train 2xe_cc3_tba-config_new_vf.yaml
```

### Benchmarking and Testing Job ([`mlip-bench-test-dep_E-F-S.job`](./mlip-bench-test-dep_E-F-S.job))

#### Performance Benchmarking
```bash
srun nequip-benchmark 2xe_cc3_tba-config_new_vf.yaml
```
Evaluates computational performance with the specified hyperparameters.

#### Model Evaluation
```bash
srun nequip-evaluate --train-dir ./MLP-2Xe_CC3_TBA_new_output/2Xe_CC3_TBA_new_mlp_vf \
                     --batch-size 1 \
                     --output ./dataset_2xe_cc3_tba_out_prediction.xyz
```
Computes test set predictions and error metrics.

#### Model Deployment
```bash
srun nequip-deploy build --train-dir ./MLP-2Xe_CC3_TBA_new_output/2Xe_CC3_TBA_new_mlp_vf \
                         ./MLP-2Xe_CC3_TBA_new_output/2Xe_CC3_TBA_new_mlp_vf/2xe_cc3_tba_new-deployed_vf.pth
```
Creates MLIP-2 model file for production use.

## Output Structure

### Training Outputs
```
MLP-2Xe_CC3_TBA_new_output/
└── 2Xe_CC3_TBA_new_mlp_vf/
    ├── best_model.pth           # Best validation model
    ├── last_model.pth           # Final training state
    ├── metrics.txt              # Training metrics log
    ├── config.yaml              # Complete configuration
    └── results.json             # Final results summary
```

### Deployment Outputs
- **MLIP-2 model**: `2xe_cc3_tba_new-deployed_vf.pth`
- **Test predictions**: `dataset_2xe_cc3_tba_out_prediction.xyz`

## Monitoring and Logging

### Weights & Biases Integration
```yaml
wandb: true
wandb_project: Allegro_MLP-porous-liquids
log_batch_freq: 10               # Logging frequency
```

### Metrics Tracking
Real-time monitoring of:
- Training and validation losses
- Force, energy, and stress errors
- Learning rate schedules
- Model convergence metrics

## Usage Instructions

### 1. Training the Model
```bash
sbatch mlip-train_E-F-S.job
```

### 2. Monitor Training Progress
- Check SLURM output files
- Monitor Weights & Biases dashboard
- Examine training logs in output directory

### 3. Evaluate and Deploy
```bash
sbatch mlip-bench-test-dep_E-F-S.job
```

## Requirements

### Computational Infrastructure
- **GPU**: NVIDIA A100 (recommended) or equivalent
- **Memory**: 40+ GB GPU memory for large systems
- **Storage**: NVMe for fast I/O during training
- **Time**: 12-36 hours depending on convergence (if the convernges take longer than the wall time, then run the `.job` file again and the model will resue from where it left)

### Software Dependencies
```bash
# Core packages
pytorch >= 1.13
nequip
allegro
ase
wandb

# System modules ([CSC Mahti](https://csc.fi/))
module load pytorch/1.13
```

### Virtual Environment Setup
```bash
python -m venv allegro_env
source allegro_env/bin/activate
pip install nequip wandb
```

---

For further details, please refer to the respective folders or contact the author via the provided email.
