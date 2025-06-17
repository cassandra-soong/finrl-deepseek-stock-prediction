import torch
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Paths
RAW_DATA_CSV = '../scrape/news.csv'
TEMP_PROCESSED_JSON = 'temp/processed_data.json'
NEWS_WITH_SCORE_CSV = 'news_with_risk_score.csv'
TEMP_DATE_RISK_CSV = 'temp/date_risk.csv'
AGGREGATED_WEIGHTS_CSV = 'aggregated_risk_scores.csv'
MODEL_CACHE_DIR = 'cache_models'
LOG_FILE = "../../main_branch.log"

# Model
G_LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
VALIDATION_LLM = "meta-llama/Llama-3.2-3B-Instruct"

# Specific source weights
SOURCE_WEIGHTS = {
                  "investing": 0.0338, 
                  "StockMarket": 0.0338, 
                  "stocks": 0.0410,
                  "wallstreetbets": 0.0600, 
                  "Etfdailynews": 0.0338, 
                  "Ndtv": 0.0338,
                  "Forbes": 0.0338, 
                  "Globenewswire": 0.0456, 
                  "Nbcnews": 0.0338,
                  "Investopedia": 0.0338, 
                  "Bostonherald": 0.0338, 
                  "Yahoo Finance": 0.0338,
                  "Coindesk": 0.0338, 
                  "Foxbusiness": 0.0338, 
                  "Telegraph": 0.0391,
                  "Fool": 0.0338, 
                  "Techcrunch": 0.0338, 
                  "Cnn": 0.0338, 
                  "Thestreet": 0.0338,
                  "Verdict": 0.0367, 
                  "Denverpost": 0.0338, 
                  "Marketwatch": 0.0347,
                  "Medicaldevice-network": 0.0338,
                  "Just-auto": 0.1002,
                  "Investmentmonitor": 0.0338,
                  "Retailbankerinternational": 0.0338
                }

# Device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
