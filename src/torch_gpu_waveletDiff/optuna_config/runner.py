"""Optuna workflow orchestration for WaveletDiff hyperparameter optimization.

This module provides high-level functions to run the entire Optuna optimization workflow,
allowing the Colab notebook to remain minimal and focused on configuration.
"""

import os
import sys
import subprocess
import time
import json
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler, RandomSampler
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_pareto_front
)


def setup_environment(repo_url, repo_dir, drive_base_path, optuna_db_path, checkpoint_dir):
    """
    Setup Colab environment: mount Drive, clone repo, install deps, create directories.
    
    Args:
        repo_url: GitHub repository URL
        repo_dir: Local directory for repository
        drive_base_path: Google Drive base path
        optuna_db_path: Path to Optuna SQLite database
        checkpoint_dir: Directory for trial checkpoints
    
    Returns:
        dict: Status information about setup
    """
    status = {}
    
    # Mount Drive
    try:
        from google.colab import drive
        if os.path.exists('/content/drive'):
            if not os.listdir('/content/drive'):
                print("Force remounting Drive...")
                drive.mount('/content/drive', force_remount=True)
        else:
            drive.mount('/content/drive')
        print("✅ Drive mounted")
        status['drive'] = 'mounted'
    except ImportError:
        print("Not running on Colab. Skipping Drive mount.")
        status['drive'] = 'skipped'
    
    # Clone/Update Repository
    if os.path.exists(repo_dir):
        print(f"Repo exists at {repo_dir}, pulling changes...")
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
        status['repo'] = 'updated'
    else:
        print(f"Cloning {repo_url} into {repo_dir}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        status['repo'] = 'cloned'
    
    print("✅ Repository ready")
    
    # Install Dependencies
    print("Installing dependencies...")
    deps = ["lightning", "pywavelets", "scipy", "pandas", "tqdm", 
            "optuna", "optuna-dashboard", "plotly", "kaleido", "pyngrok"]
    
    # Use sys.executable to ensure we're installing to the correct environment
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=True)
    
    import importlib
    importlib.invalidate_caches()
    
    print("✅ Dependencies installed")
    status['deps'] = 'installed'
    
    # Setup Paths
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    source_path = os.path.join(repo_dir, "src", "copied_waveletDiff", "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    
    # Create Directories
    os.makedirs(os.path.dirname(optuna_db_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    status['paths'] = 'configured'
    
    print("✅ Setup complete")
    return status


def create_study(study_name, storage_url, sampler_type, n_startup_trials,
                pruner_type, pruner_min_resource, pruner_reduction_factor,
                enable_pruning, use_multi_objective, weight_config=None):
    """
    Create or load Optuna study with specified configuration.
    
    Args:
        study_name: Name of the study
        storage_url: SQLite database URL
        sampler_type: "tpe" or "random"
        n_startup_trials: Number of random trials before TPE
        pruner_type: "hyperband", "median", or "none"
        pruner_min_resource: Minimum resource for Hyperband
        pruner_reduction_factor: Reduction factor for Hyperband
        enable_pruning: Whether to enable pruning
        use_multi_objective: Whether to use multi-objective optimization
        weight_config: Dict with 'loss', 'speed', 'stability' weights (single-obj only)
    
    Returns:
        optuna.Study: Created or loaded study
    """
    print(f"📁 Storage: {storage_url}")
    
    # Sampler
    if sampler_type == "tpe":
        sampler = TPESampler(
            n_startup_trials=n_startup_trials,
            multivariate=True,
            group=True,
            constant_liar=True
        )
        print(f"🧠 Sampler: TPE (startup trials: {n_startup_trials})")
    else:
        sampler = RandomSampler()
        print("🎲 Sampler: Random")
    
    # Pruner
    if not enable_pruning or pruner_type == "none":
        pruner = NopPruner()
        print("✂️ Pruner: Disabled")
    elif pruner_type == "hyperband":
        pruner = HyperbandPruner(
            min_resource=pruner_min_resource,
            reduction_factor=pruner_reduction_factor
        )
        print(f"✂️ Pruner: Hyperband (min_resource: {pruner_min_resource}, reduction: {pruner_reduction_factor})")
    elif pruner_type == "median":
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=500
        )
        print("✂️ Pruner: Median")
    
    # Create study
    if use_multi_objective:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            directions=["minimize", "minimize", "minimize"],
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        print("🎯 Mode: Multi-objective (Pareto optimization)")
        print("   Objectives: [loss, step_time_ms, grad_norm_variance]")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        print("🎯 Mode: Single-objective (weighted scalarization)")
        if weight_config:
            print(f"   Weights: loss={weight_config['loss']}, "
                  f"speed={weight_config['speed']}, "
                  f"stability={weight_config['stability']}")
    
    print(f"\n📊 Study: {study_name}")
    print(f"   Previous trials: {len(study.trials)}")
    print(f"   Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"   Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    if len(study.trials) > 0:
        print(f"\n💡 Tip: Study will resume from Trial {len(study.trials)} (load_if_exists=True)")
    
    return study


def launch_dashboard(storage_url, dashboard_port, ngrok_token=None):
    """
    Launch Optuna dashboard with optional ngrok tunnel.
    
    Args:
        storage_url: SQLite database URL
        dashboard_port: Port for dashboard
        ngrok_token: Optional ngrok auth token for public URL
    
    Returns:
        tuple: (dashboard_process, public_url or None)
    """
    # Kill any existing dashboard
    subprocess.run(["pkill", "-f", "optuna-dashboard"], 
                  stderr=subprocess.DEVNULL, check=False)
    
    # Start dashboard in background
    dashboard_process = subprocess.Popen(
        ["optuna-dashboard", storage_url, "--port", str(dashboard_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(3)
    
    public_url = None
    
    # Create ngrok tunnel if auth token provided
    if ngrok_token:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(ngrok_token)
            public_url = ngrok.connect(dashboard_port)
            
            print("="*60)
            print("🎨 OPTUNA DASHBOARD ACTIVE")
            print("="*60)
            print(f"🌐 Public URL: {public_url}")
            print("="*60)
            print("\n Dashboard will remain active during optimization\n")
        except Exception as e:
            print(f"⚠️ Dashboard started locally but ngrok tunnel failed: {e}")
            print(f"   Dashboard available at localhost:{dashboard_port}")
    else:
        print(f"📊 Dashboard running at localhost:{dashboard_port}")
        print("   💡 Tip: Set NGROK_AUTH_TOKEN for public URL")
    
    return dashboard_process, public_url


def run_optimization(study, fabric, base_config, repo_dir, data_path, 
                    tune_flags, default_hyperparams, checkpoint_dir,
                    trial_steps, eval_interval, compile_mode,
                    n_trials, timeout_hours, use_multi_objective,
                    enable_dashboard, dashboard_port, ngrok_token, storage_url):
    """
    Run Optuna optimization with optional live dashboard.
    
    Args:
        study: Optuna study object
        fabric: Lightning Fabric instance
        base_config: Base configuration dict
        repo_dir: Repository directory
        data_path: Training data path
        tune_flags: Dict of hyperparameters to tune
        default_hyperparams: Default hyperparameter values
        checkpoint_dir: Directory for checkpoints
        trial_steps: Steps per trial
        eval_interval: Evaluation interval
        compile_mode: torch.compile mode (or None to disable)
        n_trials: Number of trials to run
        timeout_hours: Timeout in hours (or None)
        use_multi_objective: Whether using multi-objective
        enable_dashboard: Whether to launch dashboard
        dashboard_port: Dashboard port
        ngrok_token: Ngrok auth token
        storage_url: Storage URL for dashboard
    
    Returns:
        dict: Summary of optimization results
    """
    from src.torch_gpu_waveletDiff.train.optuna_trainer import OptunaWaveletDiffTrainer
    
    # Launch Dashboard (if enabled)
    dashboard_process = None
    if enable_dashboard:
        dashboard_process, _ = launch_dashboard(storage_url, dashboard_port, ngrok_token)
    else:
        print("📊 Dashboard disabled. Results will be shown in analysis.")
    
    # Create Optuna trainer
    optuna_trainer = OptunaWaveletDiffTrainer(
        fabric=fabric,
        config_base=base_config,
        repo_dir=repo_dir,
        data_path=data_path,
        tune_flags=tune_flags,
        default_hyperparams=default_hyperparams,
        checkpoint_dir=checkpoint_dir,
        trial_steps=trial_steps,
        eval_interval=eval_interval,
        compile_mode=compile_mode
    )
    
    # Select objective function
    if use_multi_objective:
        objective_fn = optuna_trainer.objective
    else:
        objective_fn = optuna_trainer.objective_single
    
    # Run optimization
    print("="*60)
    print("🚀 STARTING OPTIMIZATION")
    print("="*60)
    print(f"Trials: {n_trials}")
    print(f"Steps per trial: {trial_steps}")
    print(f"Timeout: {timeout_hours or 'None'} hours")
    print(f"Mode: {'Multi-objective' if use_multi_objective else 'Single-objective'}")
    print(f"Compile: {compile_mode or 'Disabled (faster for short trials)'}")
    print(f"Tuning {len([v for v in tune_flags.values() if v])} hyperparameters")
    print("="*60)
    
    try:
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout_hours * 3600 if timeout_hours else None,
            n_jobs=1,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    
    print("\n✅ Optimization complete!")
    
    # Summary
    summary = {
        'total_trials': len(study.trials),
        'completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    }
    
    print(f"   Final trial count: {summary['total_trials']}")
    print(f"   Completed: {summary['completed']}")
    print(f"   Pruned: {summary['pruned']}")
    
    return summary


def analyze_results(study_name, storage_url, use_multi_objective):
    """
    Analyze and visualize optimization results.
    
    Args:
        study_name: Name of the study
        storage_url: SQLite database URL
        use_multi_objective: Whether multi-objective was used
    
    Returns:
        dict: Analysis summary
    """
    # Reload study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )
    
    print("="*60)
    print("📊 OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if use_multi_objective:
        print("\n🎯 Top Pareto-Optimal Trials:")
        pareto_trials = study.best_trials[:5]
        for i, trial in enumerate(pareto_trials):
            print(f"\nTrial {trial.number}:")
            print(f"  Loss: {trial.values[0]:.6f}")
            print(f"  Step Time: {trial.values[1]:.2f}ms")
            print(f"  Grad Variance: {trial.values[2]:.6f}")
            print(f"  Key Params: embed_dim={trial.params.get('embed_dim', 'N/A')}, "
                  f"layers={trial.params.get('num_layers', 'N/A')}, "
                  f"batch={trial.params.get('batch_size', 'N/A')}")
    else:
        print(f"\n🏆 Best Trial: {study.best_trial.number}")
        print(f"   Best Value: {study.best_value:.6f}")
        print(f"   Best Params:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
    
    # Visualizations
    print("\n📈 Generating visualizations...")
    
    try:
        fig1 = plot_optimization_history(study)
        fig1.show()
    except:
        print("Could not plot optimization history")
    
    try:
        if len(study.trials) > 10:
            fig2 = plot_param_importances(study)
            fig2.show()
    except:
        print("Could not plot parameter importances")
    
    try:
        fig3 = plot_parallel_coordinate(study)
        fig3.show()
    except:
        print("Could not plot parallel coordinates")
    
    if use_multi_objective:
        try:
            fig4 = plot_pareto_front(study)
            fig4.show()
        except:
            print("Could not plot Pareto front")
    
    print("\n✅ Analysis complete")
    
    return {'study': study, 'num_trials': len(study.trials)}


def export_best_configs(study_name, storage_url, checkpoint_dir, use_multi_objective):
    """
    Export best trial configurations to JSON files.
    
    Args:
        study_name: Name of the study
        storage_url: SQLite database URL
        checkpoint_dir: Directory to save configs
        use_multi_objective: Whether multi-objective was used
    
    Returns:
        list: Paths to exported config files
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )
    
    if use_multi_objective:
        print("🎯 Exporting top 3 Pareto-optimal configurations:\n")
        best_trials = study.best_trials[:3]
    else:
        print("🏆 Exporting best configuration:\n")
        best_trials = [study.best_trial]
    
    exported_files = []
    
    for i, trial in enumerate(best_trials):
        config_export = {
            "trial_number": trial.number,
            "hyperparameters": trial.params,
            "user_attrs": dict(trial.user_attrs),
            "state": str(trial.state)
        }
        
        if use_multi_objective:
            config_export["objectives"] = {
                "loss": trial.values[0],
                "step_time_ms": trial.values[1],
                "grad_variance": trial.values[2]
            }
        else:
            config_export["objective_value"] = trial.value
        
        # Save to file
        filename = f"{checkpoint_dir}/best_config_trial_{trial.number}.json"
        with open(filename, 'w') as f:
            json.dump(config_export, f, indent=2)
        
        print(f"Trial {trial.number}:")
        print(json.dumps(config_export, indent=2))
        print(f"\n💾 Saved to: {filename}\n")
        print("-"*60)
        
        exported_files.append(filename)
    
    print("\n✅ Configurations exported")
    print(f"\nTo use these hyperparameters, update your training notebook with values from above.")
    
    return exported_files
