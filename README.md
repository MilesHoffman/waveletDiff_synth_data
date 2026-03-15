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
  - **SOTA 22-Channel Feature Pipeline**: Deep structural decomposition of OHLC using Robust Scaled Log-Returns, rolling indicators (Hurst, YZ Volatility, Skewness, Amihud), and Volume Log-Deviation.
  - **Ratio-Based Wicks**: Wicks modeled as ratios of the bar range [0, 1] for guaranteed valid OHLC construction
  - **Contextual Encoding**: 'Day of Week' (sin/cos) and normalized Gap returns for realistic market microsctructure
  - **OHLC Constraint Preservation**: Mathematical guarantees that High ≥ Open/Close ≥ Low
- **4-Quadrant Conditioning**: Generate samples guided by specific trend, volatility, and order-flow regimes (YZ, Hurst, Skew, Amihud).
- **DDPM/DDIM Sampling**: Standard stochastic or accelerated deterministic generation
- **torch.compile Ready**: Optimized for CUDAGraph and reduce-overhead compilation

---

## 🏗️ Architecture Overview

```
Raw OHLCV → 9-Channel Reparameterization → Wavelet Transform → Level Transformers
                                                                     ↓
Synthetic OHLCV ← Inverse Reparam ← Inverse Wavelet ← Cross-Level Attention
```

### Core Components

| Component | Description |
|-----------|-------------|
| **WaveletDiffusionTransformer** | Main model with level-specific processing |
| **CrossLevelAttention** | Information exchange between wavelet bands |
| **9-Channel Reparam** | Ratio-based wicks, normalized gaps, and volume |
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
python train.py --experiment_name my_experiment --dataset stocks --export_onnx true

# Train with custom configuration
python train.py \
    --experiment_name custom_run \
    --dataset stocks \
    --epochs 2000 \
    --batch_size 256 \
    --use_ema true \
    --export_onnx true \
    --compile_enabled true \
    --compile_mode reduce-overhead
```

### Sample Generation

Sampling now utilizes **ONNX Runtime** for high-performance inference. Ensure you have run `train.py` with `--export_onnx true` before sampling.

```bash
cd src

# Generate samples using Student-t (t-EDM) - Recommended for heavy tails
python sample.py \
    --experiment_name my_experiment \
    --num_samples 10000 \
    --sampling_method t-edm

# Standard stochastic DDPM sampling
python sample.py \
    --experiment_name my_experiment \
    --num_samples 10000 \
    --sampling_method ddpm

# Accelerated DDIM sampling
python sample.py \
    --experiment_name my_experiment \
    --num_samples 10000 \
    --sampling_method ddim \
    --ddim_steps 50
```

### Evaluation

Open `evaluation.ipynb` in Google Colab or Jupyter:

```python
# Configure paths
EXPERIMENT_NAME = "my_experiment"
SAMPLING_METHOD = "t-edm" # Matches the generated files: t-edm_samples.npy

# Run all cells to generate:
# - t-SNE/PCA visualizations
# - Distribution comparisons
# - Discriminative/Predictive scores
# - Summary scorecard
```

---

## 🛠️ CLI Reference

### `train.py`

| Category | Flag | Description | Default |
|----------|------|-------------|---------|
| **Core** | `--experiment_name` | Unique ID for the run | `default` |
| | `--dataset` | Dataset name (stocks, etth1, etc.) | `stocks` |
| **Model** | `--prediction_target` | `noise` or `coefficient` | `noise` |
| | `--use_cross_level_attention` | Enable attention between wavelet bands | `true` |
| **Diffusion** | `--noise_prior` | `gaussian` or `student-t` | `gaussian` |
| | `--nu` | Student-T degrees of freedom | `3.0` |
| | `--noise_schedule` | `exponential`, `cosine`, `linear` | `exponential` |
| **Sampling** | `--sampling_method` | `ddpm`, `ddim`, `t-edm` | `ddpm` |
| | `--exploration_ratio` | Adaptive vs Min-SNR probability | `0.3` |
| **Optimization** | `--use_ema` | Track Exponential Moving Average of weights | `true` |
| | `--export_onnx` | Automatically export model to ONNX after training | `false` |
| **Performance** | `--compile_enabled` | Enable `torch.compile` (A100 optimized) | `false` |
| | `--precision` | `32`, `bf16-mixed`, `16-mixed` | `32` |

### `sample.py`

| Flag | Description |
|------|-------------|
| `--experiment_name` | Name of the experiment to load |
| `--sampling_method` | `ddpm`, `ddim`, `t-edm` (must match training prior) |
| `--num_samples` | Number of synthetic samples to generate |
| `--guidance_scale` | CFG scale (>1.0 to amplify conditioning) |
| `--inference_engine` | `onnx` (default) or `tensorrt` |


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
| `stocks` | 22 | 64 | OHLCV + SOTA Rollers (Hurst, YZ Vol, Amihud, Skew) |
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

### SOTA 22-Channel Feature Matrix

WaveletDiff uses a **Ratio-Based** structural decomposition combined with Robust Scaled Log-Returns and advanced Technical/Microstructure proxies:

```
# --- Core OHLC Structure & Cyclic Time (8 Channels) ---
[0] gap_norm = RobustScale( log(Open_t / Close_{t-1}) )
[1] intraday_return = RobustScale( log(Close_t / Open_t) )
[2] cum_ret_norm = RobustScale( log(Close_t / Open_0) )
[3] range_norm = RobustScale( (High - Low) / Close_{t-1} )
[4] wick_high_ratio = (High - max(O,C)) / (High - Low)  # [0, 1]
[5] wick_low_ratio = (min(O,C) - Low) / (High - Low)    # [0, 1]
[6] day_sin, [7] day_cos = Cyclic Day Encoding

# --- Market Microstructure & Structural Variance (7 Channels) ---
[8] log_ret_sq = (log(Close_t / Close_{t-1}))^2
[9] semivar_down = Rolling Downside Semivariance (20 periods)
[10] mfm_norm = Money Flow Multiplier [-1, 1]
[11] yz_vol_norm = Yang-Zhang Volatility 
[12] hurst_norm = Rolling Hurst Exponent (Trend Memory)
[13] skew_norm = Rolling Skewness (Tail Asymmetry)
[14] amihud_norm = Rolling Amihud Illiquidity Ratio

# --- Price Distance & Moving Averages (7 Channels) ---
[15] sma_200_dev = log(Close / SMA_200)
[16] sma_100_dev = log(Close / SMA_100)
[17] sma_50_dev  = log(Close / SMA_50)
[18] sma_20_dev  = log(Close / SMA_20)
[19] ema_9_dev   = log(Close / EMA_9)
[20] trange_dev  = log(TrueRange / SMA_20(TrueRange))
[21] vol_log_dev = (log(Volume) - Rolling_Median_Vol) / (Rolling_IQR_Vol / 1.349)
```

**OHLC Constraints** are mathematically guaranteed perfectly during Log-Return inverse chaining:
1.  `High >= Low` (by definition of range)
2.  `High >= max(Open, Close)` (by definition of wick ratios)
3.  `Low <= min(Open, Close)` (by definition of wick ratios)

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

1.  **Reparameterized Data**: Samples are generated in standard stationary feature space (Log-Returns, Feature Deviations).
2.  **Fixed Reconstruction**: During evaluation, the `fixed_anchor` price for **both** real and synthetic samples is fixed to **100.0**, and the volume anchors use matching samples.
3.  **Cumulative Dynamics**: This transforms the "Dollar Space" metrics into a pure study of cumulative returns and internal "vibe."

This ensures that a stock at $10 and a stock at $1000 are compared on an equal playing field, revealing whether the model has truly mastered the statistical texture of the market.

---

## 📈 Evaluation Metrics

The evaluation suite includes:

| Category | Metrics | Description |
|----------|---------|-------------|
| **Visual** | t-SNE, PCA, PDF comparisons | Qualitative manifold and distribution overlap |
| **Discriminative** | Hardened (LSTM) & Legacy (GRU) | Fidelity of synthetic data against trained classifiers |
| **Predictive** | Hardened (5-step) & Legacy (1-step) | Utility of synthetic data for downstream forecasting |
| **Contextual** | **Context-FID** (via TS2Vec) | Deep temporal alignment of patterns and context |
| **Temporal** | DTW-JS Divergence, Correlation | Statistical preservation of cross-correlations and time-warps |
| **Extreme Value Theory (EVT)** | Tail Index Error (Hill Estimator), Empirical VaR/ES MAE | Precision of power-law fat-tail decay and structural extreme risk |
| **Volume Microstructure** | Price-Volume Asymmetry, Volume ACF MAE | Algorithmic trading persistence and directional leverage effects |
| **Quality** | DCR, Memorization Ratio, Precision/Recall | Manifold coverage vs. training data leakage |

---

## 🔬 Adaptive Timestep Sampling (Importance Weighting)

WaveletDiff implements a **Hybrid Timestep Sampler** that transitions from broad stability (Min-SNR-γ) to targeted "hard" example sampling. This ensures the model spends more capacity on the most difficult diffusion stages once the base manifold is established.

- **Warmup Phase**: The sampler stays in high-stability Min-SNR-γ mode for a customizable percentage of the total steps (e.g., `adaptive_start_pct = 0.8`), ensuring robust initial manifold learning.
- **Adaptive Phase**: Automatically increases the sampling probability of timesteps with higher relative loss for the remainder of training.
- **Stability**: Uses a high EMA decay (e.g. `0.997`) to prevent the sampling distribution from "shaking", alongside a customizable `exploration_ratio` (e.g., `0.3`) to guarantee the entire diffusion path remains covered.

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

### Performance Optimization for A100 GPUs

WaveletDiff is optimized for NVIDIA A100 GPUs. For maximum training speed:

```bash
python train.py \
    --experiment_name a100_optimized \
    --dataset stocks \
    --precision bf16-mixed \
    --matmul_precision high \
    --compile_enabled true \
    --compile_mode reduce-overhead \
    --batch_size 512
```

**Key optimizations:**
- **`bf16-mixed`**: Utilizes A100's BF16 Tensor Cores (312 TFLOPS vs 19.5 TFLOPS fp32) → ~2× speedup
- **`torch.compile`**: Reduces kernel launch overhead via CUDAGraphs → +20-40% on top of bf16
- **`matmul_precision high`**: Enables TF32 for matrix operations (A100 default)
- **Batch size**: 512-1024 recommended for A100 (80GB) to saturate GPU utilization

Expected speedup: **~2-3× faster** than default `fp32` training with no quality loss.

### torch.compile Acceleration

```bash
python train.py \
    --experiment_name fast_train \
    --compile_enabled true \
    --compile_mode reduce-overhead  # or: default, max-autotune
```

### Heavy-Tailed Generation (Student-T Prior)

We highly recommend pairing this with the **Huber Loss** (`--loss_type huber --huber_delta 1.0`) to prevent the enormous gradients of the Student-T extreme samples from destabilizing the optimizer. Always sample using `--sampling_method t-edm` for these models to correctly utilize the Student-T posterior logic.

### Custom Noise Schedule

```yaml
# In config file
noise:
  schedule: "exponential"
  beta_start: 0.0001
  beta_end: 0.02
  gamma: 2.0  # Exponential decay rate
```

### Continuous Conditioning & Augmentation

WaveletDiff supports conditioning on continuous variables to give fine-grained control over the generated market regimes. The primary conditioning inputs include:
- **Quarter Profiles (4-Quadrant Arrays)**: Incorporates macro-level trend conditioning by taking snapshots of Yang-Zhang Volatility, Hurst Exponent, Skewness, and Amihud Illiquidity at each quarter of the generation window.

To prevent the model from overfitting to exact conditioning values seen during training and to enable smooth interpolation at inference time, **Condition Augmentation** is applied. Gaussian noise is added to the conditioning variables during training, improving the model's robustness and generalization.

At inference, you can generate samples with specific conditioning regimes.

### Exponential Moving Average (EMA) Weights

To improve the stability and quality of the generated samples, WaveletDiff supports maintaining an Exponential Moving Average (EMA) of the model weights during training. By evaluating and sampling using the EMA weights rather than the raw active weights, the generated time series exhibit higher fidelity and fewer artifacts.

### Inline Training Evaluation

WaveletDiff features an `InlineEvaluationCallback` that performs evaluation generation and metric computation periodically during the training loop. This allows you to monitor synthesis quality metrics on the fly, transforming standard PyTorch Lightning training into a continuous feedback loop for generative performance.

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
