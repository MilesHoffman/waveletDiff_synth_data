# WaveletDiff: Wavelet-Based Diffusion for Financial Time Series Synthesis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.0+-purple.svg" alt="Lightning">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</p>

A specialized diffusion model for generating high-fidelity synthetic OHLCV (Open, High, Low, Close, Volume) financial time series data. WaveletDiff uniquely combines **wavelet decomposition**, **level-specific transformers**, and **domain-aware normalization** to produce realistic synthetic market data.

---

## ✨ Key Features

- **Wavelet-Based Architecture**: Multi-resolution decomposition captures both trends and fine-grained patterns
- **Level-Specific Transformers**: Dedicated networks for each frequency band with cross-level attention
- **OHLC Constraint Preservation**: Domain-specific reparameterization ensures valid price relationships (High ≥ Low, etc.)
- **ATR Conditioning**: Generate samples with specific volatility characteristics
- **DDPM/DDIM Sampling**: Standard stochastic or accelerated deterministic generation
- **torch.compile Ready**: Optimized for CUDAGraph and reduce-overhead compilation

---

## 🏗️ Architecture Overview

```
Raw OHLCV → OHLC Reparameterization → Wavelet Transform → Level Transformers
                                                                    ↓
Synthetic OHLCV ← Inverse Reparam ← Inverse Wavelet ← Cross-Level Attention
```

### Core Components

| Component | Description |
|-----------|-------------|
| **WaveletDiffusionTransformer** | Main model with level-specific processing |
| **CrossLevelAttention** | Information exchange between wavelet bands |
| **OHLC Reparameterization** | ATR-normalized percentage-space features |
| **HybridTimestepSampler** | Importance-weighted training |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/MilesHoffman/waveletDiff_synth_data.git
cd waveletDiff_synth_data

# Create conda environment (recommended)
conda create -n waveletdiff python=3.10
conda activate waveletdiff

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pytorch-lightning>=2.0
pywavelets
scipy
pandas
numpy
tqdm
scikit-learn
seaborn
matplotlib
statsmodels
```

---

## 🚀 Quick Start

### Training

```bash
cd src

# Train on stocks dataset (default config)
python train.py --experiment_name my_experiment --dataset stocks

# Train with custom configuration
python train.py \
    --experiment_name custom_run \
    --dataset stocks \
    --epochs 2000 \
    --batch_size 256 \
    --compile_mode reduce-overhead
```

### Sample Generation

```bash
cd src

# Generate samples using DDPM
python sample.py \
    --experiment_name my_experiment \
    --dataset stocks \
    --num_samples 10000 \
    --sampling_method ddpm

# Accelerated DDIM sampling
python sample.py \
    --experiment_name my_experiment \
    --num_samples 10000 \
    --sampling_method ddim
```

### Evaluation

Open `evaluation.ipynb` in Google Colab or Jupyter:

```python
# Configure paths
EXPERIMENT_NAME = "my_experiment"
SAMPLING_METHOD = "ddpm"

# Run all cells to generate:
# - t-SNE/PCA visualizations
# - Distribution comparisons
# - Discriminative/Predictive scores
# - Summary scorecard
```

---

## 📁 Project Structure

```
waveletDiff_synth_data/
├── src/
│   ├── models/
│   │   ├── transformer.py      # WaveletDiffusionTransformer
│   │   ├── layers.py           # TimeEmbed, AdaLayerNorm, etc.
│   │   ├── attention.py        # CrossLevelAttention
│   │   └── wavelet_losses.py   # Balanced wavelet loss
│   │
│   ├── data/
│   │   ├── module.py           # WaveletTimeSeriesDataModule
│   │   └── loaders.py          # Dataset loaders (stocks, ETT, etc.)
│   │
│   ├── training/
│   │   ├── diffusion_process.py # DDPM/DDIM samplers
│   │   └── inline_evaluation.py # Training-time metrics
│   │
│   ├── evaluation/             # Evaluation metrics & visualization
│   ├── utils/                  # Config, noise schedules, samplers
│   ├── train.py                # Training entry point
│   └── sample.py               # Generation entry point
│
├── configs/
│   ├── default.yaml            # Base configuration
│   └── datasets/               # Dataset-specific overrides
│
├── data/                       # Dataset storage
├── evaluation.ipynb            # Evaluation notebook
└── waveletDiff_training.ipynb  # Colab training notebook
```

---

## ⚙️ Configuration

Configuration is managed via YAML files in `configs/`:

```yaml
# configs/default.yaml

training:
  epochs: 5000
  batch_size: 512

model:
  embed_dim: 256
  num_heads: 8
  num_layers: 8
  time_embed_dim: 128
  dropout: 0.1
  prediction_target: "noise"

attention:
  use_cross_level_attention: true

noise:
  schedule: "exponential"  # cosine, linear, exponential

sampling:
  method: "ddpm"
  ddim_eta: 0.0
  ddim_steps: null

wavelet:
  type: "auto"   # auto, db2, db4, sym2, etc.
  levels: "auto"

optimizer:
  scheduler_type: "onecycle"
  lr: 0.0002
```

Dataset-specific configs in `configs/datasets/`:
- `stocks.yaml` - Stock OHLCV data
- `etth1.yaml` - ETT-small H1 dataset
- `exchange_rate.yaml` - Exchange rate data

---

## 📊 Supported Datasets

| Dataset | Features | Seq Length | Description |
|---------|----------|------------|-------------|
| `stocks` | OHLCV (5) | 24 | Stock market data with reparameterization |
| `etth1/etth2` | 7 | 24-96 | Electricity Transformer Temperature |
| `exchange_rate` | 8 | 24 | Currency exchange rates |
| `fmri` | Variable | 24 | fMRI brain activity |
| `eeg` | 14 | 24 | EEG eye state |

### Custom Datasets

Add a loader function in `src/data/loaders.py`:

```python
def load_custom_data(data_dir, seq_len=24, normalize_data=True):
    # Load your data
    # Return: torch.Tensor, norm_stats dict
    pass
```

---

## 🔬 Technical Details

### OHLC Reparameterization

Raw OHLC prices are transformed into ATR-normalized percentage-space:

```
anchor = Open[0]
ATR_pct = mean(ATR) / anchor × 100

open_norm = (Open - anchor) / anchor / ATR_pct × 100
body_norm = (Close - Open) / anchor / ATR_pct × 100
wick_high_norm = (High - max(O,C)) / anchor / ATR_pct × 100  ≥ 0
wick_low_norm = (min(O,C) - Low) / anchor / ATR_pct × 100    ≥ 0
```

This ensures generated samples automatically satisfy OHLC constraints.

### Wavelet Decomposition

Time series are decomposed using Discrete Wavelet Transform (DWT):
- **Approximation coefficients**: Low-frequency trend
- **Detail coefficients**: High-frequency patterns at each scale

Auto-detection selects appropriate wavelet (db2-db8) based on sequence length.

### Level-Specific Processing

Each wavelet level has its own transformer:
- **Level 0 (Approximation)**: 2× embedding dimension, +2 layers
- **Detail Levels**: Standard capacity

Cross-level attention enables information exchange between scales.

---

## 📈 Evaluation Methodology: Index-100

To eliminate "price scale noise" and focus purely on temporal dynamics, WaveletDiff uses an **Index-100 Evaluation** style:

1.  **Reparameterized Data**: Samples are generated in percentage-return space (normalized by ATR).
2.  **Fixed Reconstruction**: During evaluation, the `anchor` price for **both** real and synthetic samples is fixed to **100.0**.
3.  **Cumulative Dynamics**: This transforms the "Dollar Space" metrics into a pure study of cumulative returns and internal "vibe."

This ensures that a stock at $10 and a stock at $1000 are compared on an equal playing field, revealing whether the model has truly mastered the statistical texture of the market.

---

## 📈 Evaluation Metrics

The evaluation suite includes:

| Category | Metrics |
|----------|---------|
| **Visual** | t-SNE, PCA, PDF comparison |
| **Discriminative** | Post-hoc RNN classifier accuracy |
| **Predictive** | Post-hoc RNN prediction MAE |
| **Temporal** | DTW-JS Divergence, Cross-Correlation |
| **Financial** | ACF similarity, Volatility clustering |
| **Quality** | Context-FID, Memorization ratio, Coverage |

---

## 🎮 Google Colab

Training and evaluation notebooks are Colab-ready:

1. **Training**: `waveletDiff_training.ipynb`
   - Clones repo, installs deps, mounts Drive
   - Configurable hyperparameters via Colab forms
   - Saves checkpoints to Drive

2. **Evaluation**: `evaluation.ipynb`
   - Loads model from Drive
   - Generates/caches samples
   - Runs full evaluation suite

---

## 🔧 Advanced Usage

### torch.compile Acceleration

```bash
python train.py \
    --experiment_name fast_train \
    --compile_mode reduce-overhead  # or: default, max-autotune
```

### Custom Noise Schedule

```yaml
# In config file
noise:
  schedule: "exponential"
  beta_start: 0.0001
  beta_end: 0.02
  gamma: 2.0  # Exponential decay rate
```

### ATR Conditioning

During training, the model learns to condition on ATR percentage. At inference:

```python
# Generate with specific volatility
samples = model.generate(n_samples=1000, scale=2.5)  # 2.5% ATR
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{waveletdiff2026,
  title = {WaveletDiff: Wavelet-Based Diffusion for Financial Time Series},
  author = {Hoffman, Miles},
  year = {2026},
  url = {https://github.com/MilesHoffman/waveletDiff_synth_data}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) for baseline diffusion architecture
- [TimeGAN](https://github.com/jsyoon0823/TimeGAN) for evaluation metrics
- PyWavelets for wavelet transform implementation
