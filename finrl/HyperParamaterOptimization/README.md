# 🔧 Hyperparameter Optimization for FinRL Agents

## Overview

This module implements hyperparameter optimization for reinforcement learning agents applied to financial trading environments. 
It uses **Optuna** to tune key parameters for **PPO**, **A2C**, **SAC**, and **TD3** agents based on backtested reward performance.

The agents are evaluated using Stable-Baselines3’s `evaluate_policy` and trained on data formatted for the FinRL pipeline. And used out training data for NVDIA stock.
Optimization results are saved in CSV and JSON formats for traceability and deployment.

---

## 📁 Project Structure

HyperParamaterOptimization/
├── optimizer.py               # Full Optuna-based optimization script (PPO, A2C, SAC, TD3)
├── *_trials_log.csv         # Trial logs (auto-generated)
├── best_hyperparams_*.json    # Best hyperparameter sets for each algorithm (auto-generated)

Make sure your environment includes:
	•	optuna
	•	stable-baselines3
	•	pandas
	•	FinRL (custom or base repo)


2. Prepare Data

Ensure TRAIN_CSV and TRADE_CSV paths in the script are valid and point to preprocessed CSVs that contain tic, turbulence, and all required indicators.

3. Run Optimization
python hyper.py



This script:
	•	Initializes training and evaluation environments using FinRL’s StockTradingEnv
	•	Defines objective functions for each RL algorithm
	•	Runs Optuna trials to maximize average validation reward
	•	Logs all trial results to .csv files
	•	Stores best hyperparameters to .json


Each RL agent produces:
	•	*_trials_log.csv: Records all trial parameters and rewards
	•	best_hyperparams_*.json: Contains best parameter set from the study

