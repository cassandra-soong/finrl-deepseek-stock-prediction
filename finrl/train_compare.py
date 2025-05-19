import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from custom_env import StockTradingEnv
from training_config import ENV_CONFIG, TRAINING_CONFIG, DEVICE
from training_utils import prepare_data, train_model, evaluate_model
import mlflow
from mlflow.tracking import MlflowClient

def train_and_evaluate(use_sentiment=True):
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///finrl/models/mlflow.db")
    experiment_name = f"hourly_trading_{'with' if use_sentiment else 'without'}_sentiment"
    mlflow.set_experiment(experiment_name)
    
    # Prepare data
    train_data, trade_data = prepare_data()
    
    # Configure environment
    env_config = ENV_CONFIG.copy()
    env_config['use_sentiment'] = use_sentiment
    
    # Create and train model
    env = StockTradingEnv(train_data, **env_config)
    model = A2C("MlpPolicy", env, verbose=1, device=DEVICE)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(env_config)
        mlflow.log_params(TRAINING_CONFIG)
        
        # Train model
        model = train_model(model, env, TRAINING_CONFIG['total_timesteps'])
        
        # Evaluate on trade data
        trade_env = StockTradingEnv(trade_data, **env_config)
        returns, sharpe_ratio, max_drawdown = evaluate_model(model, trade_env)
        
        # Log metrics
        mlflow.log_metric("returns", returns)
        mlflow.log_metric("sharpe_ratio", sharpe_ratio)
        mlflow.log_metric("max_drawdown", max_drawdown)
        
        # Save model
        model_path = f"finrl/models/a2c_{'with' if use_sentiment else 'without'}_sentiment"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        return returns, sharpe_ratio, max_drawdown

def plot_comparison(with_sentiment_metrics, without_sentiment_metrics):
    metrics = ['Returns', 'Sharpe Ratio', 'Max Drawdown']
    with_sentiment = [with_sentiment_metrics[0], with_sentiment_metrics[1], with_sentiment_metrics[2]]
    without_sentiment = [without_sentiment_metrics[0], without_sentiment_metrics[1], without_sentiment_metrics[2]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, with_sentiment, width, label='With Sentiment')
    rects2 = ax.bar(x + width/2, without_sentiment, width, label='Without Sentiment')
    
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('finrl/models/performance_comparison.png')
    plt.close()

def main():
    print("Training model with sentiment data...")
    with_sentiment_metrics = train_and_evaluate(use_sentiment=True)
    
    print("\nTraining model without sentiment data...")
    without_sentiment_metrics = train_and_evaluate(use_sentiment=False)
    
    print("\nGenerating comparison plot...")
    plot_comparison(with_sentiment_metrics, without_sentiment_metrics)
    
    print("\nTraining and comparison complete!")
    print(f"Results saved to finrl/models/performance_comparison.png")

if __name__ == "__main__":
    main() 