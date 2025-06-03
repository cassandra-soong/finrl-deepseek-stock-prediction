import optuna
import csv
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config_tickers import DOW_30_TICKER 
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

def make_env():
    df = YahooDownloader(start_date="2015-01-01", end_date="2021-01-01", ticker_list=DOW_30_TICKER).fetch_data()
    fe = FeatureEngineer()
    df = fe.preprocess_data(df)
    train = data_split(df, "2015-01-01", "2020-01-01")
    if 'turbulence' not in train.columns:
        train['turbulence'] = 0

    stock_dim = len(train['tic'].unique())
    state_space = 1 + 2 * stock_dim + len(fe.tech_indicator_list) * stock_dim
    action_space = stock_dim

    env = StockTradingEnv(
        df=train,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=fe.tech_indicator_list,
        turbulence_threshold=80
    )
    return env

def make_eval_env():
    df = YahooDownloader(start_date="2015-01-01", end_date="2021-01-01", ticker_list=DOW_30_TICKER).fetch_data()
    fe = FeatureEngineer()
    df = fe.preprocess_data(df)
    val = data_split(df, "2020-01-01", "2021-01-01")
    if 'turbulence' not in val.columns:
        val['turbulence'] = 0

    stock_dim = len(val['tic'].unique())
    state_space = 1 + 2 * stock_dim + len(fe.tech_indicator_list) * stock_dim
    action_space = stock_dim

    eval_env = StockTradingEnv(
        df=val,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=fe.tech_indicator_list,
        turbulence_threshold=80
    )
    return eval_env


def log_trial_result_factory(filename):
    def log_trial_result(study, trial):
        fieldnames = list(trial.params.keys()) + ['value', 'number']
        file_exists = False
        try:
            with open(filename, 'r'):
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = {**trial.params, 'value': trial.value, 'number': trial.number}
            writer.writerow(row)
    return log_trial_result


def log_trial_result(study, trial):
    # Log each trial's result to a CSV file
    fieldnames = list(trial.params.keys()) + ['value', 'number']
    file_exists = False
    try:
        with open('ppo_trials_log.csv', 'r'):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open('ppo_trials_log.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {**trial.params, 'value': trial.value, 'number': trial.number}
        writer.writerow(row)

def objective(trial):
    algo_class = PPO  
    env = make_env()  # Your custom training env
    eval_env = make_eval_env()  # Optional validation env

    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512])

    model = algo_class(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=n_steps,
        verbose=0,
        device="cuda"
    )

    model.learn(total_timesteps=100_000)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)

    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, callbacks=[log_trial_result_factory('ppo_trials_log.csv')])

# Save best trial hyperparameters to a JSON file
import json
with open("best_hyperparams_ppo.json", "w") as f:
    json.dump(study.best_trial.params, f, indent=4)

print("Best trial PPO:")
print(f"  Value (Reward): {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

def objective_a2c(trial):
    algo_class = A2C
    env = make_env()
    eval_env = make_eval_env()

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512])

    model = algo_class(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=n_steps,
        verbose=0,
        device="cuda"
    )

    model.learn(total_timesteps=100_000)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)

    return mean_reward

study_a2c = optuna.create_study(direction="maximize")
study_a2c.optimize(objective_a2c, n_trials=30, callbacks=[log_trial_result_factory('a2c_trials_log.csv')])

with open("best_hyperparams_a2c.json", "w") as f:
    json.dump(study_a2c.best_trial.params, f, indent=4)

print("Best trial A2C:")
print(f"  Value (Reward): {study_a2c.best_trial.value}")
print(f"  Params: {study_a2c.best_trial.params}")

def objective_sac(trial):
    algo_class = SAC
    env = make_env()
    eval_env = make_eval_env()

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512])

    model = algo_class(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0,
        device="cuda"
    )

    model.learn(total_timesteps=100_000)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)

    return mean_reward

study_sac = optuna.create_study(direction="maximize")
study_sac.optimize(objective_sac, n_trials=30, callbacks=[log_trial_result_factory('sac_trials_log.csv')])

with open("best_hyperparams_sac.json", "w") as f:
    json.dump(study_sac.best_trial.params, f, indent=4)

print("Best trial SAC:")
print(f"  Value (Reward): {study_sac.best_trial.value}")
print(f"  Params: {study_sac.best_trial.params}")

def objective_td3(trial):
    algo_class = TD3
    env = make_env()
    eval_env = make_eval_env()

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512])

    model = algo_class(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=0,
        device="cuda"
    )

    model.learn(total_timesteps=100_000)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)

    return mean_reward

study_td3 = optuna.create_study(direction="maximize")
study_td3.optimize(objective_td3, n_trials=30, callbacks=[log_trial_result_factory('td3_trials_log.csv')])

with open("best_hyperparams_td3.json", "w") as f:
    json.dump(study_td3.best_trial.params, f, indent=4)

print("Best trial TD3:")
print(f"  Value (Reward): {study_td3.best_trial.value}")
print(f"  Params: {study_td3.best_trial.params}")