from typing import Dict, List, Union
import os
import torch

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Available algorithms
AVAILABLE_ALGORITHMS = ["a2c", "ppo", "ddpg"]

# Base model configuration
BASE_MODEL_CONFIG = {
    "a2c": {
        "learning_rate": 0.0007,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "rms_prop_eps": 1e-05,
        "use_rms_prop": True,
        "use_sde": False,
        "normalize_advantage": False,
        "policy_kwargs": {
            "net_arch": [64, 64]
        }
    },
    "ppo": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "policy_kwargs": {
            "net_arch": [64, 64]
        }
    },
    "ddpg": {
        "learning_rate": 0.001,
        "buffer_size": 1000000,
        "learning_starts": 100,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": -1,
        "action_noise_type": "OrnsteinUhlenbeckActionNoise",
        "policy_kwargs": {
            "net_arch": [64, 64]
        }
    }
}

# Environment configuration
ENV_CONFIG = {
    "hmax": 500,                # Max shares to trade per step
    "initial_amount": 1000000,  # Initial investment amount (changed to $1M)
    "buy_cost_pct": 0.001,     # Transaction cost for buying
    "sell_cost_pct": 0.001,    # Transaction cost for selling
    "reward_scaling": 1e-10,    # Reward scaling factor
    "tech_indicator_list": [    # Technical indicators to use
        "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
        "close_30_sma", "close_60_sma"
    ],
    "use_sentiment": True,      # Whether to use sentiment data
    "sentiment_weight": 0.3     # Weight for sentiment in decision making
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps": 200000,  # Increased timesteps for better training
    "eval_freq": 10000,        # Evaluation frequency
    "n_eval_episodes": 10,     # Number of episodes for evaluation
    "save_freq": 10000,        # Model saving frequency
    "cv_folds": 5,            # Number of cross-validation folds
    "train_test_split": 0.8,  # Train/test split ratio
    "device": DEVICE,         # Training device
    "n_workers": 1 if DEVICE == "mps" else 4,  # Number of workers
    "experiment_name": "hourly_trading"  # Name for this experiment
}

# Optuna hyperparameter tuning configuration
OPTUNA_CONFIG = {
    "n_trials": 100,           # Number of trials for optimization
    "timeout": 600,            # Timeout in seconds
    "study_name": "finrl_optimization",
    "storage": "sqlite:///finrl/models/optuna_studies.db"
}

# Model versioning configuration
MODEL_VERSION_CONFIG = {
    "base_dir": os.path.join("finrl", "models"),
    "version_file": "model_versions.json",
    "metadata_fields": [
        "algorithm",
        "train_date",
        "performance_metrics",
        "hyperparameters",
        "git_commit",
        "data_hash",
        "device"
    ]
}

# Experiment tracking configuration
MLFLOW_CONFIG = {
    "tracking_uri": "sqlite:///finrl/models/mlflow.db",
    "experiment_name": "finrl_trading",
    "run_name_prefix": "trading_run"
} 