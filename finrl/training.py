import pandas as pd
import mlflow
from typing import Dict, Any
from datetime import datetime

from training_config import (
    AVAILABLE_ALGORITHMS,
    BASE_MODEL_CONFIG,
    TRAINING_CONFIG,
    MODEL_VERSION_CONFIG
)
from training_utils import (
    setup_mlflow,
    create_env,
    optimize_hyperparameters,
    perform_cross_validation,
    save_model_version,
    calculate_data_hash,
    get_model_class
)
from config import TRAIN_CSV


def train(algorithm: str = "a2c", optimize: bool = True) -> Dict[str, Any]:
    """
    Enhanced training function with support for multiple algorithms, hyperparameter optimization,
    cross-validation, and experiment tracking.

    Args:
        algorithm (str): The RL algorithm to use ('a2c', 'ppo', or 'ddpg')
        optimize (bool): Whether to perform hyperparameter optimization

    Returns:
        Dict[str, Any]: Training results including model path and metrics
    """
    try:
        if algorithm not in AVAILABLE_ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {AVAILABLE_ALGORITHMS}")

        # Setup MLflow tracking
        setup_mlflow()
        with mlflow.start_run(run_name=f"{algorithm}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Load and preprocess data
            train_data = pd.read_csv(TRAIN_CSV)
            train_data = train_data.set_index(train_data.columns[0])
            train_data.index.names = ['']
            
            # Calculate data hash for versioning
            data_hash = calculate_data_hash(train_data)
            
            # Create environments
            train_env = create_env(train_data)
            eval_env = create_env(train_data)  # Using same data for eval, but with different seed
            
            # Log training parameters
            mlflow.log_params({
                "algorithm": algorithm,
                "data_hash": data_hash,
                "total_timesteps": TRAINING_CONFIG["total_timesteps"],
                "eval_freq": TRAINING_CONFIG["eval_freq"]
            })
            
            # Hyperparameter optimization if requested
            if optimize:
                print(f"Optimizing hyperparameters for {algorithm}...")
                best_params = optimize_hyperparameters(train_env, eval_env, algorithm)
                hyperparameters = {**BASE_MODEL_CONFIG[algorithm], **best_params}
                mlflow.log_params(best_params)
            else:
                hyperparameters = BASE_MODEL_CONFIG[algorithm]
            
            # Perform cross-validation
            print("Performing cross-validation...")
            cv_mean, cv_std = perform_cross_validation(train_data, algorithm, hyperparameters)
            mlflow.log_metrics({
                "cv_mean_reward": cv_mean,
                "cv_std_reward": cv_std
            })
            print(f"Cross-validation results - Mean: {cv_mean:.2f}, Std: {cv_std:.2f}")
            
            # Train final model
            print(f"Training final {algorithm.upper()} model...")
            model_class = get_model_class(algorithm)
            model = model_class("MlpPolicy", train_env, **hyperparameters)
            
            # Train with callbacks for evaluation and checkpoints
            final_model = model.learn(
                total_timesteps=TRAINING_CONFIG["total_timesteps"],
                callback=create_callbacks(eval_env, algorithm)
            )
            
            # Save final model
            model_path = f"{MODEL_VERSION_CONFIG['base_dir']}/{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            final_model.save(model_path)
            
            # Evaluate final model
            print("Evaluating final model...")
            mean_reward = 0
            n_eval_episodes = TRAINING_CONFIG["n_eval_episodes"]
            for _ in range(n_eval_episodes):
                obs = eval_env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = final_model.predict(obs, deterministic=True)
                    obs, reward, done, _ = eval_env.step(action)
                    episode_reward += reward
                mean_reward += episode_reward
            mean_reward /= n_eval_episodes
            
            # Log final metrics
            final_metrics = {
                "final_mean_reward": mean_reward,
                "cv_mean_reward": cv_mean,
                "cv_std_reward": cv_std
            }
            mlflow.log_metrics(final_metrics)
            
            # Save model version information
            save_model_version(
                model_path=model_path,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                performance_metrics=final_metrics,
                data_hash=data_hash
            )
            
            print(f"Training completed. Model saved at: {model_path}")
            return {
                "model_path": model_path,
                "metrics": final_metrics,
                "hyperparameters": hyperparameters
            }

    except Exception as e:
        print(f"[Error in finrl training.py] -> {e}")
        mlflow.log_param("error", str(e))
        raise


if __name__ == "__main__":
    # Train models with different algorithms
    for algo in AVAILABLE_ALGORITHMS:
        print(f"\nTraining {algo.upper()} model...")
        train(algorithm=algo, optimize=True)

    # Train A2C with hyperparameter optimization
    results_a2c = train(algorithm="a2c", optimize=True)

    # Train PPO without optimization
    results_ppo = train(algorithm="ppo", optimize=False)

    # Train DDPG with optimization
    results_ddpg = train(algorithm="ddpg", optimize=True)