"""Optuna-integrated trainer for WaveletDiff hyperparameter optimization."""

import os
import time
import torch
import optuna
from optuna.trial import Trial
from typing import Dict, Tuple
import numpy as np
from tqdm.auto import tqdm

from ..optuna_config.search_space import WaveletDiffSearchSpace
from ..optuna_config.objectives import MultiObjectiveTracker


class OptunaWaveletDiffTrainer:
    """
    Optuna-integrated trainer for WaveletDiff with multi-objective optimization.
    
    Optimizes for:
    - Training loss (accuracy)
    - Step time (speed)
    - Gradient stability (prevents exploding gradients)
    """
    
    def __init__(
        self,
        fabric,
        config_base: Dict,
        repo_dir: str,
        data_path: str,
        tune_flags: Dict[str, bool],
        default_hyperparams: Dict,
        checkpoint_dir: str = "/content/drive/MyDrive/personal_drive/trading/optuna_trials",
        trial_steps: int = 2000,
        eval_interval: int = 100,
        compile_mode: str = None
    ):
        """
        Initialize Optuna trainer.
        
        Args:
            fabric: Lightning Fabric instance
            config_base: Base configuration dict
            repo_dir: Repository directory path
            data_path: Path to training data
            tune_flags: Dict of hyperparameters to tune (True) or keep fixed (False)
            default_hyperparams: Default values for hyperparameters
            checkpoint_dir: Directory for trial checkpoints
            trial_steps: Number of training steps per trial
            eval_interval: Steps between reporting intermediate values
        """
        self.fabric = fabric
        self.config_base = config_base
        self.repo_dir = repo_dir
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.trial_steps = trial_steps
        self.eval_interval = eval_interval
        self.compile_mode = compile_mode
        
        self.search_space = WaveletDiffSearchSpace(tune_flags)
        self.default_hyperparams = default_hyperparams
        self.tracker = MultiObjectiveTracker()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def objective(self, trial: Trial) -> Tuple[float, float, float]:
        """
        Optuna objective function for multi-objective optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            (avg_loss, avg_step_time_ms, grad_norm_variance)
        """
        from . import trainer
        
        self.tracker.reset()
        
        # Get hyperparameters for this trial
        hp = self.search_space.suggest_hyperparameters(trial, self.default_hyperparams)
        
        if self.fabric.is_global_zero:
            print(f"\n{'='*60}")
            print(f"Trial {trial.number} Hyperparameters:")
            print(f"{'='*60}")
            for key, value in hp.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")
        
        # Create dataloaders with trial batch size
        train_loader, datamodule, config = trainer.get_dataloaders(
            fabric=self.fabric,
            repo_dir=self.repo_dir,
            dataset_name=self.config_base['dataset']['name'],
            seq_len=self.config_base['dataset']['seq_len'],
            batch_size=hp['batch_size'],
            wavelet_type=self.config_base['wavelet']['type'],
            wavelet_levels=self.config_base['wavelet']['levels'],
            data_path=self.data_path
        )
        
        # Initialize model with trial hyperparameters
        model, optimizer, config = trainer.init_model(
            fabric=self.fabric,
            datamodule=datamodule,
            config=config,
            embed_dim=hp['embed_dim'],
            num_heads=hp['num_heads'],
            num_layers=hp['num_layers'],
            time_embed_dim=hp['time_embed_dim'],
            dropout=hp['dropout'],
            prediction_target=self.config_base['model']['prediction_target'],
            use_cross_level_attention=self.config_base['attention']['use_cross_level_attention'],
            learning_rate=hp['learning_rate'],
            weight_decay=hp['weight_decay'],
            max_lr=hp['max_lr'],
            pct_start=hp['pct_start'],
            compile_mode=self.compile_mode,
            compile_fullgraph=False
        )
        
        # Check Parameter Count Constraint
        total_params = sum(p.numel() for p in model.parameters())
        if self.fabric.is_global_zero:
            print(f"🧠 Model Size: {total_params / 1e6:.2f}M params")
        
        # Hard constraint: 40M - 100M
        if total_params < 40_000_000 or total_params > 100_000_000:
            if self.fabric.is_global_zero:
                print(f"⚠️ Pruning trial {trial.number}: Size {total_params/1e6:.2f}M is out of bounds (40M-100M Params)")
            raise optuna.TrialPruned()

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hp['max_lr'],
            total_steps=self.trial_steps,
            pct_start=hp['pct_start']
        )
        
        # Training loop
        model.train()
        train_iter = iter(train_loader)
        
        # Check if multi-objective (for pruning logic)
        is_multi_objective = len(trial.study.directions) > 1
        
        # Progress bar setup (Rank 0 only)
        pbar = None
        update_interval = max(1, int(self.trial_steps * 0.05))
        if self.fabric.is_global_zero:
            pbar = tqdm(
                total=self.trial_steps,
                desc=f"Trial {trial.number}", 
                leave=False, # Disappear after completion
                miniters=update_interval
            )
        
        last_update_step = 0
        for step in range(self.trial_steps):
            step_start = time.time()
            
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            x_0 = batch[0]
            
            # Forward & Loss
            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, model.T, (x_0.size(0),), device=self.fabric.device)
            loss = model.compute_loss(x_0, t)
            
            # Backward
            self.fabric.backward(loss)
            
            # Calculate gradient norm BEFORE clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Clip gradients
            self.fabric.clip_gradients(
                model, 
                optimizer, 
                max_norm=hp['grad_clip_norm'], 
                error_if_nonfinite=False
            )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            step_time = time.time() - step_start
            self.tracker.add_step(loss.item(), step_time, total_norm)
            
            # Update progress bar every 5% (to prevent spam)
            if pbar is not None:
                if step % update_interval == 0 or step == self.trial_steps - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.update(step - last_update_step)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}',
                        'grad': f'{total_norm:.2f}'
                    })
                    last_update_step = step

            # Report intermediate value for pruning (Single-Objective ONLY)
            # Optuna does NOT support trial.report() for multi-objective optimization.
            if not is_multi_objective and (step + 1) % self.eval_interval == 0:
                intermediate_loss = self.tracker.get_intermediate_loss(window=100)
                trial.report(intermediate_loss, step)
                
                # Check for pruning
                if trial.should_prune():
                    if self.fabric.is_global_zero:
                        print(f"\n⚠️ Trial {trial.number} pruned at step {step+1}")
                    # Close progress bar before raising
                    if pbar is not None:
                        pbar.close()
                    raise optuna.TrialPruned()
            
            # Early termination checks
            if self.tracker.has_exploding_gradients(threshold=100.0):
                if self.fabric.is_global_zero:
                    print(f"\n⚠️ Trial {trial.number} stopped: Exploding gradients detected")
                trial.set_user_attr("stopped_reason", "exploding_gradients")
                break
            
            if self.tracker.has_diverged(loss_threshold=10.0):
                if self.fabric.is_global_zero:
                    print(f"\n⚠️ Trial {trial.number} stopped: Loss diverged")
                trial.set_user_attr("stopped_reason", "diverged_loss")
                break
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
        
        # Get final objectives
        avg_loss, avg_step_time, grad_norm_variance = self.tracker.get_objectives(window=500)
        
        # Get detailed statistics
        stats = self.tracker.get_statistics()
        
        # Log statistics to trial
        for key, value in stats.items():
            trial.set_user_attr(key, value)
        
        # Convert step time to milliseconds for better readability
        avg_step_time_ms = avg_step_time * 1000
        
        if self.fabric.is_global_zero:
            print(f"\n{'='*60}")
            print(f"Trial {trial.number} Complete")
            print(f"{'='*60}")
            print(f"  Final Loss: {avg_loss:.6f}")
            print(f"  Avg Step Time: {avg_step_time_ms:.2f}ms")
            print(f"  Grad Norm Variance: {grad_norm_variance:.6f}")
            print(f"  Max Grad Norm: {stats['max_grad_norm']:.2f}")
            print(f"{'='*60}\n")
        
        # Penalize exploding gradients heavily
        if self.tracker.has_exploding_gradients():
            return (avg_loss * 10, avg_step_time_ms, grad_norm_variance * 10)
        
        return (avg_loss, avg_step_time_ms, grad_norm_variance)
    
    def objective_single(self, trial: Trial) -> float:
        """
        Single-objective wrapper using weighted scalarization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Weighted combination of objectives
        """
        loss, step_time_ms, grad_var = self.objective(trial)
        
        # Default weights (can be customized)
        w_loss = 1.0
        w_speed = 0.001  # ms -> comparable scale
        w_stability = 0.2
        
        weighted_objective = w_loss * loss + w_speed * step_time_ms + w_stability * grad_var
        
        return weighted_objective
