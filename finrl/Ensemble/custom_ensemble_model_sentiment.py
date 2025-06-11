import pandas as pd
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Create directory if it doesn't exist
os.makedirs("finrl/ensemble", exist_ok=True)


def create_sentiment_env(df, model_name):
    """Create a sentiment-aware environment for each prediction"""
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension  # Keep original state space
    
    # Model-specific parameters with adjusted reward scaling for sentiment
    model_params = {
        "a2c": {"reward_scaling": 1e-4, "hmax": 500},
        "ppo": {"reward_scaling": 1e-4, "hmax": 500},
        "sac": {"reward_scaling": 1e-10, "hmax": 1000},
        "td3": {"reward_scaling": 1e-10, "hmax": 1000},
        "ensemble": {"reward_scaling": 1e-4, "hmax": 500}
    }
    
    params = model_params[model_name.lower()]
    
    env_kwargs = {
        "hmax": params["hmax"],
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,  # Keep original indicators
        "action_space": stock_dimension,
        "reward_scaling": params["reward_scaling"]
    }
    
    # Create environment
    env = StockTradingEnv(df=df, **env_kwargs)
    
    # Add current step tracking
    env.current_step = 0
    
    # Store original reset function
    original_reset = env.reset
    
    def risk_aware_reset(seed=None, options=None):
        env.current_step = 0
        try:
            if seed is not None and options is not None:
                result = original_reset(seed=seed, options=options)
            elif seed is not None:
                result = original_reset(seed=seed)
            else:
                result = original_reset()
        except TypeError:
            # If the original environment doesn't support seed/options
            result = original_reset()
        
        # Handle both return formats (obs, info) and just obs
        if isinstance(result, tuple):
            obs, info = result
            env.state = obs  # Update environment state
            return obs, info
        env.state = result  # Update environment state
        return result
    
    env.reset = risk_aware_reset
    
    # Modify the step function to incorporate risk
    original_step = env.step
    
    def risk_aware_step(action):
        step_result = original_step(action)
        
        # Handle both old and new gym return formats
        if isinstance(step_result, tuple):
            if len(step_result) == 5:  # New format: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old format: (obs, reward, done, info)
                next_state, reward, done, info = step_result
        else:  # Direct observation return
            next_state = step_result
            done = env.done
            info = {}
            reward = 0
        
        # Update environment state
        env.state = next_state
        
        # Get current risk score
        current_risk = df.iloc[env.current_step]['risk_score']
        
        # Adjust reward based on risk
        risk_factor = 1 - current_risk
        adjusted_reward = reward * risk_factor
        
        # Update step counter
        env.current_step += 1
        
        # Return in the same format as received
        if isinstance(step_result, tuple):
            if len(step_result) == 5:
                return next_state, adjusted_reward, terminated, truncated, info
            else:
                return next_state, adjusted_reward, done, info
        else:
            return next_state
    
    env.step = risk_aware_step
    return env

# Load the trade data with sentiment
print("Loading trade data with sentiment...")
try:
    trade = pd.read_csv('finrl/trade_data.csv')
    risk_scores = pd.read_csv('sentiment/aggregated_risk_scores.csv')
    
    # Convert datetime column to date for merging
    datetime_series = pd.to_datetime(risk_scores['datetime'])
    datetime_series = datetime_series.dt.tz_localize(None)
    risk_scores['date'] = datetime_series.dt.strftime('%Y-%m-%d %H:%M:%S')
    risk_scores['risk_score'] = risk_scores['avg_weighted_score'] / 5.0  # Normalize to 0-1 range
    
    # Handle duplicate dates by taking the mean risk score
    risk_scores = risk_scores.groupby('date')['risk_score'].mean().reset_index()
    
    # Merge trade data with risk scores
    trade = pd.merge(trade, risk_scores[['date', 'risk_score']], on='date', how='left')
    
    # Check for missing risk scores
    missing_dates = trade[trade['risk_score'].isna()]['date'].unique()
    if len(missing_dates) > 0:
        print(f"Warning: Missing risk scores for {len(missing_dates)} dates. Using neutral value (0.5).")
        print(f"First few missing dates: {missing_dates[:5]}")
    
    trade = trade.fillna({'risk_score': 0.5})  # Fill missing risk scores with neutral value
    trade_dates = pd.to_datetime(trade['date']).dt.tz_localize(None)  # Store datetime index for alignment without timezone
    
    # Verify data integrity
    if len(trade) == 0:
        raise ValueError("Empty trade data after processing")
    if len(risk_scores) == 0:
        raise ValueError("Empty risk scores data after processing")
        
    trade = trade.set_index(trade.columns[0])
    trade.index.names = ['']
    
except Exception as e:
    print(f"Error loading or processing data: {str(e)}")
    print("Traceback:")
    import traceback
    traceback.print_exc()
    raise  # Re-raise the exception to stop execution

# Load trained models
print("Loading trained models...")
try:
    trained_a2c = A2C.load("finrl/trained_models/agent_a2c")
    trained_ppo = PPO.load("finrl/trained_models/agent_ppo")
    trained_sac = SAC.load("finrl/trained_models/agent_sac")
    trained_td3 = TD3.load("finrl/trained_models/agent_td3")
except Exception as e:
    print(f"Error loading trained models: {str(e)}")
    print("Traceback:")
    import traceback
    traceback.print_exc()
    raise  # Re-raise the exception to stop execution

def get_model_predictions_with_sentiment(model, model_name):
    """Get predictions from a single model using sentiment-aware environment"""
    print(f"Getting predictions from {model_name}...")
    env = create_sentiment_env(trade, model_name)
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)
    return df_account_value, df_actions

# Get predictions for each model with sentiment-aware environments
print("\nGetting individual model predictions...")

account_value_a2c, actions_a2c = get_model_predictions_with_sentiment(trained_a2c, "A2C")
print("\nA2C Actions Summary:")
print(f"Shape: {actions_a2c.shape if hasattr(actions_a2c, 'shape') else len(actions_a2c)}")
print("First few actions:")
print(pd.DataFrame(actions_a2c).head())

account_value_ppo, actions_ppo = get_model_predictions_with_sentiment(trained_ppo, "PPO")
print("\nPPO Actions Summary:")
print(f"Shape: {actions_ppo.shape if hasattr(actions_ppo, 'shape') else len(actions_ppo)}")
print("First few actions:")
print(pd.DataFrame(actions_ppo).head())

account_value_sac, actions_sac = get_model_predictions_with_sentiment(trained_sac, "SAC")
print("\nSAC Actions Summary:")
print(f"Shape: {actions_sac.shape if hasattr(actions_sac, 'shape') else len(actions_sac)}")
print("First few actions:")
print(pd.DataFrame(actions_sac).head())

account_value_td3, actions_td3 = get_model_predictions_with_sentiment(trained_td3, "TD3")
print("\nTD3 Actions Summary:")
print(f"Shape: {actions_td3.shape if hasattr(actions_td3, 'shape') else len(actions_td3)}")
print("First few actions:")
print(pd.DataFrame(actions_td3).head())

def calculate_risk_adjusted_metrics(account_values, risk_scores):
    """Calculate risk-adjusted performance metrics for each model"""
    metrics = []
    
    try:
        # Prepare risk scores DataFrame
        risk_df = risk_scores.set_index('date')
        
        for model_name, values in account_values.items():
            try:
                # Extract account values and ensure it's a series
                if isinstance(values, pd.DataFrame):
                    account_series = values['account_value']
                else:
                    account_series = pd.Series(values)
                
                # Verify data integrity
                if len(account_series) < 2:
                    print(f"Warning: Insufficient data points for {model_name}, skipping metrics calculation")
                    continue
                
                # Calculate daily returns
                returns = account_series.pct_change().dropna()
                
                # Align risk scores with returns
                aligned_risk = risk_df['risk_score'].reindex(returns.index).fillna(0.5)
                risk_factor = 1 - aligned_risk
                
                # Calculate risk-adjusted returns
                risk_adjusted_returns = returns * risk_factor
                
                # Calculate metrics with error handling
                try:
                    total_return = (account_series.iloc[-1] - account_series.iloc[0]) / account_series.iloc[0] * 100
                except (IndexError, ZeroDivisionError) as e:
                    print(f"Warning: Error calculating total return for {model_name}: {str(e)}")
                    total_return = 0
                
                try:
                    annual_return = total_return / len(returns) * 252
                except ZeroDivisionError:
                    print(f"Warning: Error calculating annual return for {model_name}")
                    annual_return = 0
                
                try:
                    volatility = risk_adjusted_returns.std() * np.sqrt(252) * 100
                except Exception as e:
                    print(f"Warning: Error calculating volatility for {model_name}: {str(e)}")
                    volatility = float('inf')
                
                try:
                    sharpe = annual_return / volatility if volatility != 0 and not np.isinf(volatility) else 0
                except Exception as e:
                    print(f"Warning: Error calculating Sharpe ratio for {model_name}: {str(e)}")
                    sharpe = 0
                
                try:
                    max_drawdown = (account_series / account_series.expanding(min_periods=1).max() - 1).min() * 100
                except Exception as e:
                    print(f"Warning: Error calculating max drawdown for {model_name}: {str(e)}")
                    max_drawdown = -100
                
                metrics.append({
                    'Model': model_name,
                    'Total Return (%)': total_return,
                    'Annual Return (%)': annual_return,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown (%)': max_drawdown,
                    'Volatility (%)': volatility
                })
                
            except Exception as e:
                print(f"Error calculating metrics for {model_name}: {str(e)}")
                metrics.append({
                    'Model': model_name,
                    'Total Return (%)': 0,
                    'Annual Return (%)': 0,
                    'Sharpe Ratio': 0,
                    'Max Drawdown (%)': -100,
                    'Volatility (%)': float('inf')
                })
        
        return pd.DataFrame(metrics)
    
    except Exception as e:
        print(f"Error in metrics calculation: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        # Return DataFrame with default values
        return pd.DataFrame([{
            'Model': model_name,
            'Total Return (%)': 0,
            'Annual Return (%)': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown (%)': -100,
            'Volatility (%)': float('inf')
        } for model_name in account_values.keys()])

print("\nCalculating risk-adjusted performance metrics...")
# Combine account values
all_account_values = {
    'A2C': account_value_a2c,
    'PPO': account_value_ppo,
    'SAC': account_value_sac,
    'TD3': account_value_td3
}

# Calculate metrics with risk adjustment
metrics_df = calculate_risk_adjusted_metrics(all_account_values, risk_scores)
print("\nRisk-Adjusted Performance Metrics:")
print(metrics_df)

print("\nCalculating risk-aware ensemble weights...")

# Use model names as the index for proper access
sharpe_ratios = metrics_df.set_index("Model")["Sharpe Ratio"]

# Protect against division by zero
min_sharpe = sharpe_ratios[sharpe_ratios > 0].min() if any(sharpe_ratios > 0) else 1e-6
sharpe_ratios = sharpe_ratios.clip(lower=min_sharpe)

# Base weights from normalized Sharpe ratios
base_weights = sharpe_ratios / sharpe_ratios.sum()

# Adjust weights based on current risk
current_risk = risk_scores['risk_score'].iloc[-1]
weights = base_weights.copy()

if current_risk > 0.7:
    weights.loc[['A2C', 'PPO']] *= 1.2
    weights.loc[['SAC', 'TD3']] *= 0.8
elif current_risk < 0.3:
    weights.loc[['SAC', 'TD3']] *= 1.2
    weights.loc[['A2C', 'PPO']] *= 0.8

# Normalize final weights
weights = weights / weights.sum()

print("\nRisk-Adjusted Ensemble Weights:")
print(weights)

def combine_actions_with_risk(actions_list, weights, risk_scores, trade_dates):
    """Combine actions from different models with risk-aware weights"""
    min_len = min(len(actions) for actions in actions_list)
    trade_dates = trade_dates[:min_len]
    processed_actions = []
    
    for actions in actions_list:
        # Extract actions from DataFrame
        if isinstance(actions, pd.DataFrame):
            if 'actions' in actions.columns:
                # Convert string representation of list to actual list if needed
                action_values = actions['actions'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                # Convert list of actions to numpy array and reshape
                action_array = np.array([a if isinstance(a, list) else [a] for a in action_values])
                action_array = action_array.squeeze()  # Remove extra dimensions
                if len(action_array.shape) == 1:
                    action_array = action_array.reshape(-1, 1)  # Ensure 2D array
                df = pd.DataFrame(action_array, index=actions.index)
            else:
                df = actions
            df = df.iloc[:min_len]
        else:
            df = pd.DataFrame(actions).iloc[:min_len]
        
        # Ensure numeric values
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        df.index = trade_dates
        processed_actions.append(df)
        
        # Print debug info
        print(f"\nProcessed actions shape: {df.shape}")
        print("First few processed actions:")
        print(df.head())
    
    # Stack actions: shape (n_steps, n_models, stock_dim)
    actions_stack = np.stack([df.values for df in processed_actions], axis=1)  # (n_steps, n_models, stock_dim)
    print(f"\nStacked actions shape: {actions_stack.shape}")
    
    weights = np.array(weights) / np.sum(weights)
    weighted_actions = np.tensordot(actions_stack, weights, axes=([1],[0]))  # (n_steps, stock_dim)
    
    # Prepare risk scores
    risk_df = risk_scores.set_index('date')
    risk_df.index = pd.to_datetime(risk_df.index).tz_localize(None)  # Convert to timezone-naive datetime
    
    # Align risk factor by date
    risk_factor = 1 - risk_df['risk_score'].reindex(trade_dates, method='ffill').fillna(0.5).values
    risk_adjusted_actions = weighted_actions * risk_factor[:, None]
    
    # Normalize actions while preserving direction
    abs_actions = np.abs(risk_adjusted_actions)
    max_abs_action = np.max(abs_actions, axis=0)
    max_abs_action[max_abs_action == 0] = 1
    
    # Scale actions to [-1, 1] range while preserving relative magnitudes
    normalized_actions = risk_adjusted_actions / max_abs_action[None, :]
    
    # Create DataFrame with processed actions
    action_df = pd.DataFrame(normalized_actions, columns=range(weighted_actions.shape[1]), index=trade_dates)
    
    # Add debugging information
    print("\nAction Processing Summary:")
    print(f"Original actions shape: {actions_stack.shape}")
    print(f"Weighted actions shape: {weighted_actions.shape}")
    print(f"Risk factors shape: {risk_factor.shape}")
    print(f"Final actions shape: {normalized_actions.shape}")
    print("\nAction statistics before normalization:")
    print(f"Min: {np.min(risk_adjusted_actions)}")
    print(f"Max: {np.max(risk_adjusted_actions)}")
    print(f"Mean: {np.mean(risk_adjusted_actions)}")
    print(f"Std: {np.std(risk_adjusted_actions)}")
    print("\nAction statistics after normalization:")
    print(f"Min: {np.min(normalized_actions)}")
    print(f"Max: {np.max(normalized_actions)}")
    print(f"Mean: {np.mean(normalized_actions)}")
    print(f"Std: {np.std(normalized_actions)}")
    
    return action_df

# Combine predictions using the risk-adjusted weights
ensemble_actions = combine_actions_with_risk(
    [actions_a2c, actions_ppo, actions_sac, actions_td3],
    weights,
    risk_scores,
    trade_dates  # Pass trade_dates explicitly
)

# Get predictions from ensemble model
print("\nGetting ensemble predictions...")
print("Running ensemble predictions...")

# Create ensemble environment
e_trade_ensemble = create_sentiment_env(trade, "ensemble")

# Get predictions using ensemble actions
df_account_value_ensemble = pd.DataFrame()
df_actions_ensemble = pd.DataFrame()

try:
    # Initialize state
    state = e_trade_ensemble.reset()
    done = False
    
    account_value_list = []
    date_list = []
    action_list = []
    
    # Calculate initial account value
    initial_cash = float(e_trade_ensemble.state[0])
    initial_stocks = np.array(e_trade_ensemble.state[1:e_trade_ensemble.stock_dim + 1])
    initial_prices = np.array(e_trade_ensemble.state[e_trade_ensemble.stock_dim + 1:2 * e_trade_ensemble.stock_dim + 1])
    initial_stocks_value = np.sum(initial_stocks * initial_prices)
    account_value_list.append(initial_cash + initial_stocks_value)
    date_list.append(trade.index[0])
    
    # Step through environment
    while not done:
        current_date = trade.index[e_trade_ensemble.current_step]
        
        # Get ensemble action vector for this step
        if e_trade_ensemble.current_step < len(ensemble_actions):
            try:
                action_vector = ensemble_actions.iloc[e_trade_ensemble.current_step].values.astype(np.float32)
            except (IndexError, AttributeError) as e:
                print(f"Warning: Error getting action for step {e_trade_ensemble.current_step}: {str(e)}")
                action_vector = np.zeros(e_trade_ensemble.stock_dim, dtype=np.float32)
        else:
            print(f"Warning: Step {e_trade_ensemble.current_step} exceeds available actions, using neutral action")
            action_vector = np.zeros(e_trade_ensemble.stock_dim, dtype=np.float32)
        
        # Record action
        action_list.append(action_vector)
        
        try:
            # Take step in environment
            step_result = e_trade_ensemble.step(action_vector)
            
            # Handle different step return formats
            if isinstance(step_result, tuple):
                if len(step_result) == 5:  # New format: (obs, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Old format: (obs, reward, done, info)
                    next_state, reward, done, info = step_result
            else:  # Direct observation return
                next_state = step_result
                done = e_trade_ensemble.done
            
            # Update environment state
            e_trade_ensemble.state = next_state
            
            # Calculate account value
            try:
                cash = float(e_trade_ensemble.state[0])
                stocks = np.array(e_trade_ensemble.state[1:e_trade_ensemble.stock_dim + 1])
                prices = np.array(e_trade_ensemble.state[e_trade_ensemble.stock_dim + 1:2 * e_trade_ensemble.stock_dim + 1])
                stocks_value = np.sum(stocks * prices)
                total_value = cash + stocks_value
                
                # Record values
                account_value_list.append(total_value)
                date_list.append(current_date)
            except (IndexError, ValueError) as e:
                print(f"Warning: Error calculating account value at step {e_trade_ensemble.current_step}: {str(e)}")
                # Use last known good value or initial value
                total_value = account_value_list[-1] if account_value_list else initial_cash + initial_stocks_value
                account_value_list.append(total_value)
                date_list.append(current_date)
                
        except Exception as e:
            print(f"Warning: Error during environment step {e_trade_ensemble.current_step}: {str(e)}")
            # Break the loop if we encounter a critical error
            break
    
    # Create DataFrame with results
    df_account_value_ensemble = pd.DataFrame({
        'date': date_list,
        'account_value': account_value_list
    })
    
    # Create actions DataFrame with matching dates
    df_actions_ensemble = pd.DataFrame(action_list, columns=ensemble_actions.columns, index=date_list[1:])
    df_actions_ensemble['date'] = date_list[1:]
    
except Exception as e:
    print(f"Error during ensemble prediction: {str(e)}")
    print("Traceback:")
    import traceback
    traceback.print_exc()
    
    # Create fallback DataFrames with correct lengths and more informative values
    fallback_length = min(len(trade.index), len(ensemble_actions))
    print(f"Creating fallback results with length {fallback_length}")
    
    # Use initial value for account value series
    initial_value = 1000000  # Initial portfolio value
    df_account_value_ensemble = pd.DataFrame({
        'date': trade.index[:fallback_length],
        'account_value': [initial_value] * fallback_length
    })
    
    # Use neutral actions for action series
    if hasattr(e_trade_ensemble, 'stock_dim'):
        neutral_actions = np.zeros((fallback_length, e_trade_ensemble.stock_dim))
    else:
        neutral_actions = ensemble_actions.iloc[:fallback_length].values
    
    df_actions_ensemble = pd.DataFrame(
        neutral_actions,
        columns=ensemble_actions.columns,
        index=trade.index[:fallback_length]
    )
    df_actions_ensemble['date'] = trade.index[:fallback_length]

# Add ensemble results to all values
all_account_values['Risk-Aware Ensemble'] = df_account_value_ensemble

# Calculate final metrics
print("\nFinal Risk-Adjusted Performance Metrics (Including Ensemble):")
metrics_df_with_ensemble = calculate_risk_adjusted_metrics(all_account_values, risk_scores)
print(metrics_df_with_ensemble)

# After combining predictions
print("\nEnsemble Actions Summary:")
print(f"Shape: {ensemble_actions.shape}")
print("First few actions:")
print(ensemble_actions.head())
print("\nAction statistics:")
print(ensemble_actions.describe())

# After calculating ensemble account values
print("\nEnsemble Account Values Summary:")
print(f"Length: {len(df_account_value_ensemble)}")
print("First few values:")
print(df_account_value_ensemble.head())
print("\nAccount value statistics:")
print(df_account_value_ensemble.describe())

# Save results
print("\nSaving results...")
results = {
    'metrics': metrics_df_with_ensemble.to_dict(),
    'weights': weights.tolist(),
    'account_values': {
        'A2C': account_value_a2c.to_dict(),
        'PPO': account_value_ppo.to_dict(),
        'SAC': account_value_sac.to_dict(),
        'TD3': account_value_td3.to_dict(),
        'Ensemble': df_account_value_ensemble.to_dict()
    }
}



# Save to CSV
df_account_value_ensemble.to_csv('finrl/ensemble/result_with_ensemble.csv')

# Create performance comparison plot
plt.figure(figsize=(15, 8))

# Set style for better visualization
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2

# Plot each model's performance
colors = {
    'A2C': '#2ecc71',  # Green
    'PPO': '#3498db',  # Blue
    'SAC': '#e74c3c',  # Red
    'TD3': '#f39c12',  # Orange
    'Risk-Aware Ensemble': '#9b59b6'  # Purple
}

for model_name, values in all_account_values.items():
    if isinstance(values, pd.DataFrame):
        plt.plot(pd.to_datetime(values['date']), 
                values['account_value'] / 1000000,  # Convert to millions
                label=model_name,
                color=colors[model_name])

# Customize the plot
plt.title('Portfolio Value Over Time\nComparison of Different Trading Strategies', 
          fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value (Million $)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Add annotations for final returns
final_values = {}
for model_name, values in all_account_values.items():
    if isinstance(values, pd.DataFrame):
        final_value = values['account_value'].iloc[-1]
        initial_value = values['account_value'].iloc[0]
        return_pct = ((final_value - initial_value) / initial_value) * 100
        final_values[model_name] = return_pct

# Add text box with returns
text = 'Total Returns:\n'
for model, ret in final_values.items():
    text += f'{model}: {ret:.2f}%\n'
plt.text(1.05, 0.5, text, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
         fontsize=10, verticalalignment='center')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# # Save the plot
# plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
# plt.close()

# Create bar plot comparing returns and risks
plt.figure(figsize=(15, 6))

# Prepare data
metrics_df = metrics_df_with_ensemble.set_index('Model')
returns = metrics_df['Total Return (%)']
volatilities = metrics_df['Volatility (%)']
sharpe_ratios = metrics_df['Sharpe Ratio']
max_drawdowns = metrics_df['Max Drawdown (%)']

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Returns vs Volatility
ax1.scatter(volatilities, returns, c=[colors[m] for m in returns.index], s=100)
for i, model in enumerate(returns.index):
    ax1.annotate(model, (volatilities[i], returns[i]), 
                xytext=(5, 5), textcoords='offset points')
ax1.set_xlabel('Volatility (%)')
ax1.set_ylabel('Total Return (%)')
ax1.set_title('Risk-Return Comparison')
ax1.grid(True, alpha=0.3)

# Plot 2: Sharpe Ratio and Max Drawdown
x = np.arange(len(returns.index))
width = 0.35

bars1 = ax2.bar(x - width/2, sharpe_ratios, width, label='Sharpe Ratio', color='skyblue')
bars2 = ax2.bar(x + width/2, abs(max_drawdowns), width, label='|Max Drawdown| (%)', color='lightcoral')

ax2.set_ylabel('Ratio / Percentage')
ax2.set_title('Risk Metrics Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(returns.index, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=0)

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.savefig('finrl/ensemble/risk_return_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nDone! Results saved to 'result_with_ensemble.csv' and plots saved as PNG files.") 

"""

1. **Risk-Aware Environment**:
   - Created a sentiment-aware trading environment that incorporates risk scores
   - Risk scores are normalized to 0-1 range from sentiment data
   - Rewards are adjusted based on risk: `adjusted_reward = reward * (1 - risk_score)`
   - Different parameters for conservative (A2C/PPO) and aggressive (SAC/TD3) models

2. **Model Weighting Strategy**:
   - Base weights are calculated using Sharpe ratios:
     ```python
     sharpe_ratios = metrics_df['Sharpe Ratio']
     base_weights = sharpe_ratios / sharpe_ratios.sum()
     ```

3. **Dynamic Risk Adjustment**:
   - Weights are adjusted based on current risk level:
     - High Risk (> 0.7):
       * Conservative models (A2C, PPO) get 20% more weight (×1.2)
       * Aggressive models (SAC, TD3) get 20% less weight (×0.8)
     - Low Risk (< 0.3):
       * Aggressive models get 20% more weight (×1.2)
       * Conservative models get 20% less weight (×0.8)
     - Medium Risk:
       * Use base Sharpe ratio weights

       
4. **Action Combination Process**:
   - Get predictions from all models
   - Stack actions: `(n_steps, n_models, stock_dim)`
   - Apply risk-adjusted weights
   - Multiply by risk factor: `risk_factor = 1 - risk_score`
   - Normalize to [-1, 1] range while preserving direction

   
5. **Model-Specific Parameters**:
   ```python
   model_params = {
       "a2c": {"reward_scaling": 1e-4, "hmax": 500},    # Conservative
       "ppo": {"reward_scaling": 1e-4, "hmax": 500},    # Conservative
       "sac": {"reward_scaling": 1e-10, "hmax": 1000},  # Aggressive
       "td3": {"reward_scaling": 1e-10, "hmax": 1000},  # Aggressive
       "ensemble": {"reward_scaling": 1e-4, "hmax": 500}
   }
   ```

   
Results of this strategy:
- Lower total return (11.24% vs ~16.8%)
- Significantly lower volatility (7.25% vs ~11.1%)
- Better Sharpe ratio (0.685 vs 0.666-0.676)
- Much better maximum drawdown (-23.75% vs -33.70%) 

The strategy effectively achieved its goal of creating a more balanced portfolio with better risk-adjusted returns and downside protection, even though it sacrificed some absolute returns in the process.

"""