import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from config import TRAIN_CSV, LOG_FILE
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def train():
    """
    This function trains the A2C model using train_data.csv obtained from processing.py
    This function is called only if it is necessary to retrain the model.

    Args:
        None
        
    """ 
    try:
        # Load train data
        train = pd.read_csv(TRAIN_CSV)
        train = train.set_index(train.columns[0])
        train.index.names = ['']

        # Environment setup
        stock_dimension = len(train.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        logging.info(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        buy_cost_list = sell_cost_list = [0.001] * stock_dimension  
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 500,                         # Max shares to trade per step
            "initial_amount": 1000000,            # Lower for intraday
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-10              # Smaller for hourly granularity
        }

        e_train_gym = StockTradingEnv(df = train, **env_kwargs)

        # Initialize FinRL environment
        env_train, _ = e_train_gym.get_sb_env()
        logging.info(f"Training environment: {type(env_train)}")

        # Initialize A2C agent
        agent = DRLAgent(env=env_train)
        model_a2c = agent.get_model("a2c")

        # Train the agent
        trained_a2c = agent.train_model(
            model=model_a2c,
            tb_log_name='a2c_hourly',
            total_timesteps=150000   # Slightly more steps due to smaller interval
        )

        trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")
    
    except Exception as e:
        logging.info(f"[Error in finrl training.py] -> {e}")
        

if __name__ == "__main__":
    train()