import os
import json
import hashlib
import mlflow
import optuna
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import yfinance as yf

from training_config import (
    BASE_MODEL_CONFIG,
    ENV_CONFIG,
    TRAINING_CONFIG,
    OPTUNA_CONFIG,
    MODEL_VERSION_CONFIG,
    MLFLOW_CONFIG
)

# Add device configuration for MPS
def get_device():
    """Get the appropriate device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

def get_model_class(algorithm: str):
    """Get the model class based on algorithm name."""
    model_classes = {
        "a2c": A2C,
        "ppo": PPO,
        "ddpg": DDPG
    }
    return model_classes.get(algorithm)

def calculate_data_hash(data: pd.DataFrame) -> str:
    """Calculate hash of training data for versioning."""
    return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        with open('.git/HEAD', 'r') as f:
            ref = f.read().strip()
        if ref.startswith('ref: '):
            ref_path = os.path.join('.git', ref[5:])
            with open(ref_path, 'r') as f:
                return f.read().strip()
        return ref
    except:
        return "unknown"

def save_model_version(
    model_path: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    performance_metrics: Dict[str, float],
    data_hash: str
) -> None:
    """Save model version information."""
    version_file = os.path.join(MODEL_VERSION_CONFIG["base_dir"], MODEL_VERSION_CONFIG["version_file"])
    
    version_info = {
        "algorithm": algorithm,
        "train_date": datetime.now().isoformat(),
        "model_path": model_path,
        "hyperparameters": hyperparameters,
        "performance_metrics": performance_metrics,
        "git_commit": get_git_commit(),
        "data_hash": data_hash
    }
    
    try:
        with open(version_file, 'r') as f:
            versions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        versions = []
    
    versions.append(version_info)
    
    os.makedirs(os.path.dirname(version_file), exist_ok=True)
    with open(version_file, 'w') as f:
        json.dump(versions, f, indent=4)

def create_callbacks(
    eval_env,
    model_name: str,
    eval_freq: int = TRAINING_CONFIG["eval_freq"],
    save_freq: int = TRAINING_CONFIG["save_freq"]
) -> List:
    """Create training callbacks."""
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"finrl/models/{model_name}/best_model",
        log_path=f"finrl/models/{model_name}/logs",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"finrl/models/{model_name}/checkpoints",
        name_prefix=model_name
    )
    
    return [eval_callback, checkpoint_callback]

def optimize_hyperparameters(
    train_env,
    eval_env,
    algorithm: str,
    n_trials: int = OPTUNA_CONFIG["n_trials"]
) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    def objective(trial):
        # Get base config and modify with trial suggestions
        config = BASE_MODEL_CONFIG[algorithm].copy()
        
        if algorithm == "a2c":
            config.update({
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 3, 10),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9)
            })
        elif algorithm == "ppo":
            config.update({
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 1024, 4096),
                "batch_size": trial.suggest_int("batch_size", 32, 256),
                "n_epochs": trial.suggest_int("n_epochs", 5, 20),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4)
            })
        elif algorithm == "ddpg":
            config.update({
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 500000, 2000000),
                "batch_size": trial.suggest_int("batch_size", 50, 300),
                "tau": trial.suggest_float("tau", 0.001, 0.01)
            })
        
        # Create and train model
        model_class = get_model_class(algorithm)
        model = model_class("MlpPolicy", train_env, **config)
        
        try:
            model.learn(
                total_timesteps=TRAINING_CONFIG["total_timesteps"] // 2,  # Reduced for optimization
                callback=create_callbacks(eval_env, f"optuna_{trial.number}")
            )
            
            # Evaluate
            mean_reward = 0
            n_eval_episodes = TRAINING_CONFIG["n_eval_episodes"]
            for _ in range(n_eval_episodes):
                obs = eval_env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = eval_env.step(action)
                    episode_reward += reward
                mean_reward += episode_reward
            mean_reward /= n_eval_episodes
            
            return -mean_reward  # Negative because Optuna minimizes
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")
    
    # Create and run study
    study = optuna.create_study(
        study_name=OPTUNA_CONFIG["study_name"],
        storage=OPTUNA_CONFIG["storage"],
        load_if_exists=True
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=OPTUNA_CONFIG["timeout"]
    )
    
    return study.best_params

def perform_cross_validation(
    data: pd.DataFrame,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    n_splits: int = TRAINING_CONFIG["cv_folds"]
) -> Tuple[float, float]:
    """Perform time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # Create environments
        train_env = create_env(train_data)
        val_env = create_env(val_data)
        
        # Train model
        model_class = get_model_class(algorithm)
        model = model_class("MlpPolicy", train_env, **hyperparameters)
        model.learn(
            total_timesteps=TRAINING_CONFIG["total_timesteps"] // n_splits,
            callback=create_callbacks(val_env, f"cv_fold_{len(cv_scores)}")
        )
        
        # Evaluate
        mean_reward = 0
        n_eval_episodes = TRAINING_CONFIG["n_eval_episodes"]
        for _ in range(n_eval_episodes):
            obs = val_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = val_env.step(action)
                episode_reward += reward
            mean_reward += episode_reward
        cv_scores.append(mean_reward / n_eval_episodes)
    
    return np.mean(cv_scores), np.std(cv_scores)

def create_env(data: pd.DataFrame):
    """Create a trading environment."""
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    
    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(ENV_CONFIG["tech_indicator_list"]) * stock_dimension
    
    env_kwargs = {
        "hmax": ENV_CONFIG["hmax"],
        "initial_amount": ENV_CONFIG["initial_amount"],
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [ENV_CONFIG["buy_cost_pct"]] * stock_dimension,
        "sell_cost_pct": [ENV_CONFIG["sell_cost_pct"]] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ENV_CONFIG["tech_indicator_list"],
        "action_space": stock_dimension,
        "reward_scaling": ENV_CONFIG["reward_scaling"],
        "device": get_device()  # Add device configuration
    }
    
    env = StockTradingEnv(df=data, **env_kwargs)
    return env.get_sb_env()[0]

def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and trading data for the model.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and trading data
    """
    # Download data
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    print(f"Downloading hourly data from yfinance between {start_date.date()} - {end_date.date()}")
    df = yf.download("NVDA", start=start_date, end=end_date, interval="1h")
    
    # Debug: Check columns
    required_cols = ['High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns or df[col].isnull().all():
            print(f"[DEBUG] DataFrame is missing required column: {col}")
            print("[DEBUG] DataFrame columns:", df.columns)
            print("[DEBUG] DataFrame head:\n", df.head())
            raise ValueError(f"DataFrame missing required column: {col}")
    
    # Add technical indicators
    df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['boll_ub'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['boll_lb'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    df['rsi_30'] = calculate_rsi(df['Close'], 30)
    df['cci_30'] = calculate_cci(df, 30)
    df['dx_30'] = calculate_dx(df, 30)
    df['close_30_sma'] = df['Close'].rolling(window=30).mean()
    df['close_60_sma'] = df['Close'].rolling(window=60).mean()
    
    # Add sentiment data if available
    try:
        sentiment_df = pd.read_csv('sentiment/sentiment_scores.csv', index_col=0, parse_dates=True)
        df = df.join(sentiment_df, how='left')
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
    except:
        print("No sentiment data found, proceeding without sentiment features")
        df['sentiment_score'] = 0
    
    # Split into train and trade sets
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    trade_data = df[train_size:]
    
    print(f"Train size: {len(train_data)} rows")
    print(f"Trade size: {len(trade_data)} rows")
    
    return train_data, trade_data

def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_cci(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Commodity Channel Index"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_ma = tp.rolling(window=period).mean()
    tp_md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - tp_ma) / (0.015 * tp_md)

def calculate_dx(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Directional Movement Index"""
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': abs(df['High'] - df['Close'].shift(1)),
        'lc': abs(df['Low'] - df['Close'].shift(1))
    }).max(axis=1)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean()
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean()
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx

def train_model(model, env, total_timesteps: int):
    """
    Train the model for the specified number of timesteps.
    Args:
        model: The RL model to train
        env: The training environment
        total_timesteps: Number of timesteps to train for
    Returns:
        The trained model
    """
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_model(model, env, n_episodes: int = 10) -> Tuple[float, float, float]:
    """
    Evaluate the model's performance.
    Args:
        model: The trained model
        env: The evaluation environment
        n_episodes: Number of episodes to evaluate
    Returns:
        Tuple of (average returns, sharpe ratio, max drawdown)
    """
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_returns = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_returns.append(reward)
        
        returns.append(sum(episode_returns))
    
    avg_returns = np.mean(returns)
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)
    max_drawdown = np.min(returns)
    
    return avg_returns, sharpe_ratio, max_drawdown 