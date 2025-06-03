from utils import *
from config import RESULTS_CSV, LOG_FILE
import sys
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def get_inference():
    """
    This function executes the steps in the inference stage.
    It loads the train_data.csv and trade_data.csv obtained from the processing stage, and loads the aggregated_risk_scores.csv from the /sentiment part.
    It runs the model prediction for Agent 1 (only stock data) and Agent 2 (stock data + sentiment data).
    It calculates the MVO and loads the DJIA benchmark index hourly data as a base comparison.
    It merges the results from Agent 1, Agent 2, MVO, and DJIA into results.csv to be plotted in the final dashboard.

    Args:
        None
        
    """ 
    try:
        # Load train and trade data
        trade = load_trade()
        logging.info("Loaded train and trade csv.")
        
        # Load the pre-trained A2C model
        trained_a2c = load_trained_a2c()
        logging.info("Loaded trained A2C model.")
        
        # Load the pre-trained SAC model
        trained_sac = load_trained_sac()
        logging.info("Loaded trained SAC model.")
         
        # Load aggregated_risk_scores
        trade_sentiment = load_aggregated_risk_score(trade)
        logging.info("Loaded aggregated risk scores and merged with trade data.")

        # Predict A2C Agent 1
        df_account_value_a2c_agent1, _ = predict_agent_1(trade, trained_a2c)
        logging.info("A2C Agent 1 prediction done.")
        
        # Predict SAC Agent 1
        df_account_value_sac_agent1, _ = predict_agent_1(trade, trained_sac)
        logging.info("SAC Agent 1 prediction done.")
        
        # Predict A2C Agent 2
        df_account_value_a2c_agent2, _ = predict_agent_2(trade, trained_a2c, trade_sentiment)
        logging.info("A2C Agent 2 prediction done.")
        
        # Predict SAC Agent 2
        df_account_value_sac_agent2, _ = predict_agent_2(trade, trained_sac, trade_sentiment)
        logging.info("SAC Agent 2 prediction done.")

        # Calculate Mean Variance Optimization (MVO)
        StockData, arStockPrices, rows, cols = calculate_mvo(trade)
        
        # Calculate Mean Returns and Covariance Matrix
        meanReturns, covReturns = calculate_mean_cov(arStockPrices, rows, cols)
        logging.info("Mean Returns:", meanReturns)
        logging.info("Covariance Matrix:", covReturns)
        
        # Calculate Efficient Frontier
        MVO_result = calculate_efficient_frontier(meanReturns, covReturns, trade, StockData)
        logging.info("MVO calculation done.")
        
        # Get hourly data from DJIA benchmark index
        dji = get_djia_index(trade)
        logging.info("Loaded DJIA hourly data.")
        
        # Merge results
        result = merge_results(
                    df_account_value_a2c_agent1, 
                    df_account_value_a2c_agent2, 
                    df_account_value_sac_agent1,
                    df_account_value_sac_agent2,
                    MVO_result, 
                    dji
                )
        
        # Save results to csv
        result.to_csv(RESULTS_CSV)
        logging.info("Results merged and saved as results.csv")
        
        
    except Exception as e:
        logging.info(f"[Error in finrl inference.py] -> {e}")
        sys.exit(1)