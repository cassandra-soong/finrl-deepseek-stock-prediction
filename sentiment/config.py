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
                  "investing": 0.0346, 
                  "StockMarket": 0.0398, 
                  "stocks": 0.0487,
                  "wallstreetbets": 0.0397, 
                  "Etfdailynews": 0.0400, 
                  "Ndtv": 0.0400,
                  "Forbes": 0.0541, 
                  "Globenewswire": 0.0400, 
                  "Nbcnews": 0.0672,
                  "Investopedia": 0.0400, 
                  "Bostonherald": 0.0378, 
                  "Yahoo Finance": 0.0257,
                  "Coindesk": 0.0409, 
                  "Foxbusiness": 0.0395, 
                  "Telegraph": 0.0397,
                  "Fool": 0.0250, 
                  "Techcrunch": 0.0400, 
                  "Cnn": 0.0302, 
                  "Thestreet": 0.0364,
                  "Verdict": 0.0394, 
                  "Denverpost": 0.0440, 
                  "Marketwatch": 0.0400,
                  "Medicaldevice-network": 0.0377,
                  "Just-auto": 0.0399,
                  "Investmentmonitor": 0.0399
                }

# Device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
