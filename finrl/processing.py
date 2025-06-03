import pandas as pd
import yfinance as yf
import itertools
import datetime
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
from config import TRAIN_CSV, TRADE_CSV, LOG_FILE
import sys
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def process():
    """
    This function helps to download hourly NVDA stock data from yahoo finance 600 days ago from the current date.
    After loading the hourly data, it is pre-processed and combined with stock indicators using FinRL's FeatureEngineer.
    Finally, it splits the data into train/trade set (80%-20%) and saves them into csv files for the next stage.
    
    Args:
        None
        
    """
    try:
        # Specify period for train and trade data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=600)

        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        logging.info("Start date: ", start_date)
        logging.info("End date: ", end_date)

        # Download NVDA hourly data
        logging.info(f'Downloading hourly data from yfinance between {start_date} - {end_date}')
        nvda_df_yf = yf.download(
            tickers="NVDA",
            start=start_date,
            end=end_date,
            interval="1h",
        )

        # Reset index and add 'tic' column for FinRL compatibility
        nvda_df_yf.columns = nvda_df_yf.columns.get_level_values(0)
        nvda_df_yf.reset_index(inplace=True)
        nvda_df_yf.rename(columns={'Datetime': 'date'}, inplace=True)
        nvda_df_yf['tic'] = 'NVDA'

        # Reorder and rename columns to lowercase
        nvda_df_yf = nvda_df_yf.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Add a column for the day of the week (0 = Monday, 6 = Sunday)
        nvda_df_yf['day'] = nvda_df_yf['date'].dt.dayofweek

        # Optionally filter out weekends (since the market is closed)
        nvda_df_yf = nvda_df_yf[nvda_df_yf['day'] < 5]

        # Keep only needed columns for FinRL
        nvda_df_yf = nvda_df_yf[['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']]

        # Add technical indicators using FinRL's FeatureEngineer
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,  # Not available for hourly data
            use_turbulence=False,
            user_defined_feature=False
        )

        processed = fe.preprocess_data(nvda_df_yf)

        # Fill missing hourly time gaps
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max(), freq='1H'))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
        processed_full = processed_full.sort_values(['date', 'tic'])
        processed_full = processed_full.fillna(0)

        columns = [
            'open', 'high', 'low', 'close', 'volume', 'macd',
            'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
            'close_30_sma', 'close_60_sma'
        ]

        for col in columns:
            processed_full = processed_full[processed_full[col] != 0]

        processed_full = processed_full.reset_index(drop=True)

        logging.info(f'processed df shape: {processed_full.shape}')

        # Split into train/test
        TRAIN_START_DATE = processed_full['date'].min()
        TRAIN_END_DATE = processed_full['date'].iloc[int(len(processed_full) * 0.8)]
        TRADE_START_DATE = processed_full['date'].iloc[int(len(processed_full) * 0.8) + 1]
        TRADE_END_DATE = processed_full['date'].max()

        train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

        logging.info(f'Train start date: {TRAIN_START_DATE}')
        logging.info(f'Train end date: {TRAIN_END_DATE}')
        logging.info(f'Trade start date: {TRADE_START_DATE}')
        logging.info(f'Trade end date: {TRADE_END_DATE}')

        logging.info(f"Train size: {len(train)} rows")
        logging.info(f"Trade size: {len(trade)} rows")

        # Save train and trade csv
        train.to_csv(TRAIN_CSV)
        logging.info("Saved train_date.csv")
        trade.to_csv(TRADE_CSV)
        logging.info("Saved trade_date.csv")
    
    except Exception as e:
        logging.info(f"[Error in finrl processing.py] -> {e}")
        sys.exit(1)