from datetime import datetime, timedelta
import json
import pandas as pd
import re
from config import TEMP_PROCESSED_JSON, LOG_FILE
import sys
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def data_preprocessing(path):
    """
    This function reads from news.csv and extracts only the subset of news data from yesterday to today.
    After extracting, df_subset is saved in temp/processed_data.json.
    
    Args:
        path(str): Path to news.csv
    """
    try:
        # Load the CSV file
        df = pd.read_csv(path, delimiter=";", encoding='utf-8')
        
       # Rename columns accordingly
        df = df.rename(columns={
            "Date and Timestamp": "datetime",
            "Title": "header",
            "Full Text": "content",
            "Source": "source",
            "SpecificSource": "specific_source",
            "Link": "link",
        })

        # Reformat the date column to datetime object and sort in ascending order
        def standardize_datetime_format(dt_str):
            # Match format like '16.3.2025 12:51'
            if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}:\d{2}$', dt_str):
                # Convert to 'YYYY-MM-DD HH:MM:SS'
                return pd.to_datetime(dt_str, dayfirst=True).strftime('%Y-%m-%d %H:%M:%S')
            return dt_str  # Leave other formats unchanged

        # Apply the function to each row
        df['datetime'] = df['datetime'].astype(str).apply(standardize_datetime_format)

        # Convert the datetime column to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime', ascending=True)

        # Set startdate as yesterday's date and end date as today's date
        start_date = (datetime.now() - timedelta(days=1)).date()
        end_date = datetime.now().date()
        
        logging.info(f'Start date: {start_date}')
        logging.info(f'End date: {end_date}')

        df_subset = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date < end_date)]
        df_subset.reset_index(drop=True)

        # Check if df_subset is empty
        if df_subset.empty:
            raise ValueError("No data available for the given date range. df_subset is empty.")

        # Convert each row to a dictionary and collect in a list
        data_dicts = df_subset.to_dict(orient='records')

        # Save to JSON file
        with open(TEMP_PROCESSED_JSON, 'w') as f:
            json.dump(data_dicts, f, indent=4, default=str)
            logging.info('Processed data saved as JSON file')

    except Exception as e:
        logging.info(f"[Error in data preprocessing pipeline] -> {e}")
        sys.exit(1)
