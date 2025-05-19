import pandas as pd
import numpy as np
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO

# Load sentiment scores
sentiment_df = pd.read_csv('sentiment/sentiment_scores.csv')

# Load stock data
stock_df = pd.read_csv('finrl/train_data.csv')  # Updated path to stock data

# Remove timezone from 'date' column
stock_df['date'] = pd.to_datetime(stock_df['date']).dt.tz_localize(None)

# Merge sentiment scores with stock data
sentiment_df['datetime'] = pd.to_datetime(sentiment_df['datetime'])
merged_df = pd.merge(stock_df, sentiment_df, left_on='date', right_on='datetime', how='left')

# Fill missing sentiment scores with 0
merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0)

# Feature engineering
fe = FeatureEngineer(
    use_technical_indicator=True,
    use_turbulence=True,
    user_defined_feature=False,
)

# Process data with sentiment
processed_df = fe.preprocess_data(merged_df)

# Define the environment
env = StockTradingEnv(
    df=processed_df,
    initial_amount=10000,
    state_space=len(processed_df.columns),
    action_space=3,
    reward_scaling=1e-4,
    stock_dim=1,  # Assuming only one stock
    hmax=100,  # Maximum number of shares to buy or sell
    num_stock_shares=[0],  # Initial number of shares
    buy_cost_pct=0.001,  # Cost percentage for buying stocks
    sell_cost_pct=0.001,  # Cost percentage for selling stocks
    tech_indicator_list=['rsi', 'cci', 'dx'],  # Correct technical indicators
)

# Train agent with sentiment
agent = DRLAgent(env=env)
model = agent.get_model("ppo")
model.learn(total_timesteps=10000)  # Adjust timesteps as needed

# Evaluate agent with sentiment
obs = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
print(f"Agent with sentiment - Total Reward: {total_reward}")

# Train agent without sentiment
# Remove sentiment column
processed_df_no_sentiment = processed_df.drop(columns=['sentiment_score'])
env_no_sentiment = StockTradingEnv(
    df=processed_df_no_sentiment,
    initial_amount=10000,
    state_space=len(processed_df_no_sentiment.columns),
    action_space=3,
    reward_scaling=1e-4,
    stock_dim=1,  # Assuming only one stock
    hmax=100,  # Maximum number of shares to buy or sell
    num_stock_shares=[0],  # Initial number of shares
    buy_cost_pct=0.001,  # Cost percentage for buying stocks
    sell_cost_pct=0.001,  # Cost percentage for selling stocks
    tech_indicator_list=['rsi', 'cci', 'dx'],  # Correct technical indicators
)

agent_no_sentiment = DRLAgent(env=env_no_sentiment)
model_no_sentiment = agent_no_sentiment.get_model("ppo")
model_no_sentiment.learn(total_timesteps=10000)  # Adjust timesteps as needed

# Evaluate agent without sentiment
obs = env_no_sentiment.reset()
done = False
total_reward_no_sentiment = 0
while not done:
    action, _ = model_no_sentiment.predict(obs)
    obs, reward, done, _ = env_no_sentiment.step(action)
    total_reward_no_sentiment += reward
print(f"Agent without sentiment - Total Reward: {total_reward_no_sentiment}")

# Compare results
print(f"Difference in Total Reward: {total_reward - total_reward_no_sentiment}") 