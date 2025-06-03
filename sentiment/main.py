from config import RAW_DATA_CSV, TEMP_PROCESSED_JSON, NEWS_WITH_SCORE_CSV, TEMP_DATE_RISK_CSV, LOG_FILE
from data_preprocessing import data_preprocessing
from risk_score_generation import get_all_scores, append_score_to_csv
from risk_score_aggregation import aggregate_risk_score
from risk_score_validation import validate_all_scores, regeneration
import json
import sys
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

logging.info("Sentiment started.")


if __name__ == "__main__":
    try:
        # Extract yesterday's news from news.csv and save it as JSON file
        data_preprocessing(path=RAW_DATA_CSV)
        logging.info("\n--- Data preprocessing completed ---\n")

        # Get the processed data JSON file
        with open(TEMP_PROCESSED_JSON, 'r') as f:
            json_data = json.load(f)
        logging.info("\n--- Processed JSON Data loaded successfully ---\n")
        
        # Generate risk score for each news
        scored_articles = get_all_scores(json_data)
        logging.info("\n--- Risk Score Generation Completed ---\n")

        # Validate risk scores
        validated_articles = validate_all_scores(scored_articles)
        logging.info("\n--- Validation Completed ---\n")

        # Regeneration
        regenerated_articles = regeneration(validated_articles)
        logging.info("\n--- Regeneration Completed ---\n")

        # Extract risk scores from all articles
        risk_scores = [article['risk_score'] for article in regenerated_articles]
        
        # Append new data with risk score for tracing purposes
        append_score_to_csv(json_data, risk_scores, NEWS_WITH_SCORE_CSV)
        logging.info("\n--- Appending risk score to CSV Completed ---\n")
        
        # Aggregate risk score
        aggregate_risk_score(TEMP_DATE_RISK_CSV)
        logging.info("\n--- Aggregating risk score Completed ---\n")
    except Exception as e:
        logging.info(f"Error occured in main.py -> {e}")
        sys.exit(1)