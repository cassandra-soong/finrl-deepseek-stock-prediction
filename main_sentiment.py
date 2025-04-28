from config import RAW_DATA_CSV, TEMP_PROCESSED_JSON, NEWS_WITH_SCORE_CSV, TEMP_DATE_RISK_CSV
from data_preprocessing import data_preprocessing
from risk_score_generation import get_all_scores, append_score_to_csv
from risk_score_aggregation import aggregate_risk_score
import json

if __name__ == "__main__":
    try:
        # Extract yesterday's news from news.csv and save it as JSON file
        data_preprocessing(path=RAW_DATA_CSV)

        # Get the processed data JSON file
        with open(TEMP_PROCESSED_JSON, 'r') as f:
            json_data = json.load(f)
        
        # Generate risk score for each news
        risk_scores = get_all_scores(json_data)
        
        # Append new data with risk score to news_with_risk_score.csv for tracing purposes
        append_score_to_csv(json_data, risk_scores, NEWS_WITH_SCORE_CSV)
        
        # Aggregate risk score
        aggregate_risk_score(TEMP_DATE_RISK_CSV)
        

    except Exception as e:
        print(f"Error occured -> {e}")
