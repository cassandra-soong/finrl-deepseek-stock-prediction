from utils import *
from config import RESULTS_CSV
import pandas as pd
import sys


def get_inference():
    try:
        # Load train and trade data
        _, trade = load_train_trade()
        print("Loaded train and trade csv.")
        
        # Load the pre-trained model
        trained_a2c = load_trained_a2c()
        print("Loaded trained A2C model.")
         
        # Load aggregated_risk_scores
        trade_sentiment = load_aggregated_risk_score(trade)
        print("Loaded aggregated risk scores and merged with trade data.")

        # Predict Agent 1
        df_account_value_a2c_agent1, _ = predict_agent_1(trade, trained_a2c)
        print("Agent 1 prediction done.")
        
        # Predict Agent 2
        df_account_value_a2c_agent2, _ = predict_agent_2(trade, trained_a2c, trade_sentiment)
        print("Agent 2 prediction done.")

        # Calculate Mean Variance Optimization (MVO)
        StockData, arStockPrices, rows, cols = calculate_mvo(trade)
        
        # Calculate Mean Returns and Covariance Matrix
        meanReturns, covReturns = calculate_mean_cov(arStockPrices, rows, cols)
        print("Mean Returns:", meanReturns)
        print("Covariance Matrix:", covReturns)
        
        # Calculate Efficient Frontier
        MVO_result = calculate_efficient_frontier(meanReturns, covReturns, trade, StockData)
        print("MVO calculation done.")
        
        # Get hourly data from DJIA benchmark index
        dji = get_djia_index(trade)
        print("Loaded DJIA hourly data.")
        
        # Merge results
        result = merge_results(df_account_value_a2c_agent1, df_account_value_a2c_agent2, MVO_result, dji)
        
        # Save results to csv
        result.to_csv(RESULTS_CSV)
        print("Results merged and saved as results.csv")
        
        
    except Exception as e:
        print(f"[Error in finrl inference.py] -> {e}")
        sys.exit(1)