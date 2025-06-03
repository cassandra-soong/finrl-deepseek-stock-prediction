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
G_LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
VALIDATION_LLM = "meta-llama/Llama-3.2-3B-Instruct"

# Specific source weights
SOURCE_WEIGHTS = {
                  "investing": 0.0360, 
                  "StockMarket": 0.0647, 
                  "stocks": 0.0297,
                  "wallstreetbets": 0.0609, 
                  "Etfdailynews": 0.0445, 
                  "Ndtv": 0.0411,
                  "Forbes": 0.0411, 
                  "Globenewswire": 0.0322, 
                  "Nbcnews": 0.0402,
                  "Investopedia": 0.0492, 
                  "Bostonherald": 0.0587, 
                  "Yahoo Finance": 0.0411,
                  "Coindesk": 0.0405, 
                  "Foxbusiness": 0.0411, 
                  "Telegraph": 0.0645,
                  "Fool": 0.0322, 
                  "Techcrunch": 0.0617, 
                  "Cnn": 0.0514, 
                  "Thestreet": 0.0411,
                  "Verdict": 0.0480, 
                  "Denverpost": 0.0386, 
                  "Marketwatch": 0.0411
                }

# Device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
