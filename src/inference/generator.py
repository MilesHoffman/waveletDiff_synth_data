import torch
import numpy as np
from tqdm.auto import tqdm
from lightning.fabric import Fabric

class WaveletDiffGenerator:
    def __init__(self, model, datamodule, fabric, config):
        self.model = model
        self.datamodule = datamodule
        self.fabric = fabric
        self.config = config
        self.device = fabric.device

    def generate(self, num_samples, batch_size=None, use_ddim=False):
        """
        Generates synthetic samples.
        
        Args:
            num_samples (int): Total number of samples to generate.
            batch_size (int, optional): Batch size for generation.
            use_ddim (bool): Whether to use DDIM sampling.
            
        Returns:
            samples (np.ndarray): Generated samples in time domain.
        """
        if batch_size is None:
            batch_size = self.config['training'].get('batch_size', 64)
            
        self.model.eval()
        
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        print(f"Generating {num_samples} samples using {'DDIM' if use_ddim else 'DDPM'}...")
        
        try:
             from training import DiffusionTrainer
        except ImportError:
             import sys
             from training import DiffusionTrainer

        trainer_util = DiffusionTrainer(self.model)
        
        # Loop over batches with progress bar
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating Batches"):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                
                # Disable internal progress to avoid spamming
                generated_wavelets = trainer_util.generate_samples(
                    current_batch_size, 
                    use_ddim=use_ddim, 
                    show_progress=False
                )
                
                # Convert to time series batch by batch to save GPU memory
                samples_ts_batch = self.datamodule.convert_wavelet_to_timeseries(generated_wavelets)
                all_samples.append(samples_ts_batch.cpu().numpy())
                
        # Concatenate all batches
        return np.concatenate(all_samples, axis=0)
