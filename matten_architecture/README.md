# MatTen Architecture Training, Validation, and Testing

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the configuration files, training scripts, and outputs for training the MatTen neural network architecture for predicting <sup>129</sup>Xe NMR magnetic shielding tensors.

## Directory Structure

```
./
├── configs/
│   └── atomic_tensor.yaml          # Model configuration
├── datasets/
│   ├── dataset_train.json          # Training dataset
│   ├── dataset_val.json            # Validation dataset
│   └── dataset_test.json           # Test dataset
├── matten_logs/                    # Training logs and checkpoints
├── plot_training_process/          # Training visualization scripts
├── train_atomic_tensor.py          # Main training script
├── script_test.job                 # SLURM job submission script
├── matten.log                      # Application log file
├── nmr-ml_matten-output_*.txt      # SLURM stdout
└── nmr-ml_matten-errors_*.txt      # SLURM stderr
```

## Model Configuration ([`atomic_tensor.yaml`](./atomic_tensor.yaml))

### Core Settings
```yaml
seed_everything: 35                 # Reproducibility seed
log_level: info                     # Logging verbosity
```

### Data Configuration
```yaml
data:
  tensor_target_name: nmr_tensor    # Target property name
  atom_selector: atom_selector      # Atom selection for prediction
  tensor_target_formula: ij=ji      # Symmetric tensor constraint
  r_cut: 5.0                        # Interaction cutoff radius (Å)
  batch_size: 1                     # Batch size (large molecules)
  shuffle: true                     # Data shuffling
```

### Model Architecture

#### Embedding Layers
```yaml
species_embedding_dim: 16           # Atomic species embedding dimension
irreps_edge_sh: 0e + 1o + 2e       # Spherical harmonics for edges
radial_basis_type: bessel          # Radial basis function type
num_radial_basis: 8                # Number of radial functions
radial_basis_end: 5.0              # Radial cutoff matching r_cut
```

#### Message Passing Network
```yaml
num_layers: 3                      # Number of message passing layers
invariant_layers: 2                # Radial network layers
invariant_neurons: 32              # Hidden neurons in radial network
conv_layer_irreps: 32x0o+32x0e + 16x1o+16x1e + 4x2o+4x2e
nonlinearity_type: gate            # Gated nonlinearity
normalization: batch               # Batch normalization
resnet: true                       # Residual connections
```

### Output Configuration
```yaml
output_format: irreps              # Loss computed in irreps space
output_formula: ij=ji              # Symmetric tensor constraint
reduce: mean                       # Pooling for graph-level features
```

### Training Parameters
```yaml
trainer:
  max_epochs: 5000                 # Maximum training epochs
  accelerator: cuda                # GPU acceleration
  devices: 1                       # Single GPU training
```

### Optimization Settings
```yaml
optimizer:
  class_path: torch.optim.Adam
  lr: 0.01                         # Learning rate
  weight_decay: 0.00001            # L2 regularization

lr_scheduler:
  class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
  factor: 0.5                      # LR reduction factor
  patience: 50                     # Epochs before reduction
```

## Training Script ([`train_atomic_tensor.py`](./train_atomic_tensor.py))

### Core Components

#### Data Module Setup
```python
dm = TensorDataModule(**config["data"])
dm.prepare_data()
dm.setup()
```

#### Model Instantiation
```python
model = AtomicTensorModel(
    tasks=TensorRegressionTask(name="nmr_tensor"),
    backbone_hparams=config["model"],
    dataset_hparams=dm.get_to_model_info(),
    optimizer_hparams=config["optimizer"],
    lr_scheduler_hparams=config["lr_scheduler"],
)
```

#### Training Loop
```python
trainer = Trainer(
    callbacks=callbacks,
    logger=lit_logger,
    **config["trainer"],
)
trainer.fit(model, datamodule=dm)
```

### Key Features
- **Automatic configuration loading**: YAML-based setup
- **PyTorch Lightning integration**: Modern training framework
- **Logging integration**: Weights & Biases support
- **Checkpoint management**: Best model saving
- **Testing pipeline**: Automated evaluation

## Job Submission ([`script_test.job`](./script_test.job))

### SLURM Configuration
```bash
#SBATCH --partition=gpu
#SBATCH --account=plantto
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20         # 20 CPU cores
#SBATCH --mem-per-cpu=8GB          # 160 GB total memory
#SBATCH --time=40:00:00            # 40-hour time limit
#SBATCH --gres=gpu:v100:1          # Single V100 GPU
```

### Environment Setup
```bash
module load python-data/3.10
source /scratch/plantto/zakaryou/packages/matten_installation/matten_env/bin/activate
ulimit -s unlimited                # Unlimited stack size
```

### Execution
```bash
python3 train_atomic_tensor.py
```

## Training Monitoring

### Callbacks
```yaml
callbacks:
  - ModelCheckpoint:              # Save best models
      monitor: val/score
      save_top_k: 3
  - EarlyStopping:               # Prevent overfitting
      patience: 150
  - ModelSummary:                # Architecture summary
```

### Logging
```yaml
logger:
  WandbLogger:                   # Weights & Biases integration
    project: matten_proj
    save_dir: matten_logs
```

## Output Files

### Log Files
- **matten.log**: Application-level logging
- **matten_logs/**: PyTorch Lightning logs and checkpoints
- **nmr-ml_matten-output_*.txt**: Training output and metrics
- **nmr-ml_matten-errors_*.txt**: Error messages and warnings

### NMR-ML Model
- **Best checkpoint**: `last.ckpt` (to be found in [Zenodo](https://zenodo.org/records/17105321?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI1ZjkyMGVkLTU0MDktNDI2ZC04ZTkxLTNmODU2ZTc1OGMzNCIsImRhdGEiOnt9LCJyYW5kb20iOiI0NThhZTk0ZjI0MjgwYzgzMTYxZTNiOWJjMDU5YWY5ZSJ9._FOA8Roxy1PJr4DsdrR6_inETDRe-Qn8NIRbg6p_YRDgWvG5x_RbhH74y0ALEEgMjwKMQ1BFvfkgi_BxK2mb0g) repository as `4-last.ckpt.tar.bz2`)

## Usage Instructions

### 1. Environment Setup
```bash
# Load required modules
module load python-data/3.10

# Activate MatTen environment
source /path/to/matten_env/bin/activate
```

### 2. Dataset Preparation
Ensure JSON datasets are in the `datasets/` directory:
- `dataset_train.json`
- `dataset_val.json` 
- `dataset_test.json`

### 3. Configuration
Modify `configs/atomic_tensor.yaml` as needed:
- Adjust model hyperparameters
- Change training settings
- Configure logging options

### 4. Training Execution
```bash
# Interactive training
python3 train_atomic_tensor.py

# Or submit to SLURM
sbatch script_test.job
```

## Requirements

### Computational 
- **GPU**: NVIDIA V100 or equivalent (16+ GB VRAM)
- **Memory**: 16+ GB system RAM
- **Storage**: 50+ GB for datasets and checkpoints
- **Time**: 40+ hours for complete training

### Software Dependencies
```python
# Core packages
pytorch-lightning
torch-geometric
matten
wandb
loguru
pyyaml
```

---

For further details, please refer to the respective folders or contact the author via the provided email.
