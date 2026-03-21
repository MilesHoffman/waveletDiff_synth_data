# WaveletDiff: Synthetic Data Generation

## Project Overview
This project implements **WaveletDiff**, a machine learning model for synthetic time series data generation. The core architecture combines Diffusion Models with Wavelet Transforms and Transformers, allowing the model to learn and generate data in the wavelet domain. It is particularly focused on generating high-fidelity time series data, with built-in capabilities to handle financial data (stocks).

## Key Technologies
*   **Deep Learning Framework:** PyTorch, PyTorch Lightning
*   **Signal Processing:** PyWavelets (`PyWavelets`), SciPy
*   **Data Manipulation:** NumPy, Pandas
*   **Configuration:** PyYAML
*   **Visualization & Analysis:** Matplotlib, Seaborn, Jupyter, Scikit-Learn
*   **Data Acquisition:** `yfinance` (for stock data downloading)

## Project Architecture
The codebase is structured as a modular Python project heavily utilizing PyTorch Lightning for training orchestration:

*   **`src/models/`**: Contains the core neural network architectures, notably the `WaveletDiffusionTransformer` along with its components (`transformer.py`, `attention.py`, `layers.py`) and specific loss functions (`wavelet_losses.py`).
*   **`src/training/`**: Holds PyTorch Lightning modules and callbacks, including the main diffusion process (`diffusion_process.py`), inline evaluation (`inline_evaluation.py`), and custom callbacks.
*   **`src/data/`**: Manages data loading and processing via Lightning DataModules (`WaveletTimeSeriesDataModule`).
*   **`src/utils/`**: Utilities like configuration management (`ConfigManager`).
*   **`configs/`**: Stores YAML configuration files (e.g., `default.yaml`) that define hyperparameters for training, model architecture, sampling, and data processing.
*   **Root Scripts/Notebooks**:
    *   `src/train.py`: The primary entry point for training the model.
    *   `src/sample.py`: Script for generating samples from a trained model.
    *   `download_stock_data.py`: A utility script to fetch financial data using `yfinance`.
    *   `*.ipynb`: Jupyter notebooks for exploratory data analysis, evaluation, and interactive training (`waveletDiff_training.ipynb`, `evaluation.ipynb`).

## Building and Running

### 1. Installation
Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
To download sample stock data for training/testing:
```bash
python download_stock_data.py
```
*(Follow the interactive prompts to specify ticker symbols and date ranges.)*

### 3. Training the Model
Training is initiated via the `train.py` script. You can override default configuration values via command-line arguments:
```bash
python src/train.py --experiment_name my_first_experiment --epochs 5000 --batch_size 512
```

### 4. Sampling
Use the `sample.py` script to generate synthetic data from a trained model checkpoint.
```bash
python src/sample.py --experiment_name my_first_experiment # (Add necessary arguments based on sample.py's argparse)
```

## Development Conventions
*   **Configuration-Driven:** The project relies on YAML files (in the `configs/` directory) to manage complex hyperparameters. Avoid hardcoding magic numbers in the source code; instead, expose them in the configs.
*   **PyTorch Lightning Structure:** The training loop, validation, and optimizer stepping are abstracted by PyTorch Lightning (`pl.LightningModule`, `pl.Trainer`). Stick to this paradigm when adding new features to the training pipeline.
*   **Modularity:** Keep network layers, training logic, and data processing distinctly separated across `models/`, `training/`, and `data/` directories.
