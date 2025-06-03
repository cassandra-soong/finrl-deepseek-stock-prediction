import pandas as pd
from config import SOURCE_WEIGHTS, AGGREGATED_WEIGHTS_CSV, LOG_FILE
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def _floor_time_half_hour(date):
    """
    Rounds a given datetime object down to the nearest half-hour e.g 8:30, 9:30, 10:30

    Args:
        date (datetime or str): The datetime object or string representation of the datetime to be rounded

    Returns:
        datetime: A new datetime object representing the rounded-down time
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    minutes = date.minute
    if minutes < 30:
        round_date = date.replace(minute=30, second=0, microsecond=0) - timedelta(hours=1)
    else:
        round_date = date.replace(minute=30, second=0, microsecond=0)
        
    return round_date
    
    
def get_source_weight(specific_source):
    """
    Creates a new column to match each row's source to its respective weight

    Args:
        specific_source (str): The specific source of the data, e.g yahoo finance, foxbusiness etc
        
    Returns:
        source weight (float): The weight of the respective source
    """
    if specific_source in SOURCE_WEIGHTS.keys():
        return SOURCE_WEIGHTS[specific_source]

    else:
        return 0.0400  # Default weight if source is unknown
    
    
def aggregate_risk_score(filename):
    """
    Aggregates the hourly bin risk scores by calculating the calculated average score.

    Args:
        filename (str): temp/date_risk.csv
        
    Returns:
        result csv: aggregated_risk_scores.csv with only columns 'datetime' and 'avg_weighted_score' for finrl stage
    """
    df = pd.read_csv(filename)    

    # Apply the flooring to datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['floored_datetime'] = df['datetime'].apply(_floor_time_half_hour)

    # Assign weights according to source
    df['weight'] = df['specific_source'].apply(get_source_weight)
    
    aggregated = []
    grouped = df.groupby('floored_datetime')

    # Calculate average weighted risk score for each hourly bin
    for floored_time, group in tqdm(grouped, desc="Aggregating risk scores"):
        weighted_sum = (group['risk score'] * group['weight']).sum()
        total_weight = group['weight'].sum()
        avg_weighted_score = weighted_sum / total_weight if total_weight != 0 else 0
        avg_weighted_score = round(avg_weighted_score)
        aggregated.append({
            'datetime': floored_time,
            'avg_weighted_score': avg_weighted_score
        })

    # Convert result into new dataframe and append to aggregated_risk_scores.csv
    result_df = pd.DataFrame(aggregated)
    result_df.to_csv(AGGREGATED_WEIGHTS_CSV , mode='a', header=False, index=False)
    logging.info(f"Appended aggregated weights to existing CSV: {AGGREGATED_WEIGHTS_CSV }")