import pandas as pd
import re
from tqdm import tqdm
from config import G_LLM, TEMP_DATE_RISK_CSV, LOG_FILE
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import sys
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


tokenizer = AutoTokenizer.from_pretrained(G_LLM)
model = AutoModelForCausalLM.from_pretrained(G_LLM).to("cuda")

scored_articles = []

def get_risk_score(source, header, content):
    """
    Generate risk score for each news/reddit/tweet entry using the provided pipeline.

    Args:
        pipeline (pipeline object): Text generation pipeline.
        source (str): Source of the data ("News", "Reddit", or "Tweet").
        header (str): Header/title of the news/reddit/tweet entry.
        content (str): Content/body of the news/reddit/tweet entry.

    Returns:
        int: Risk score assigned to the news/reddit/tweet entry.
    """
    try:
        if (source.lower() == "news") or (source.lower() == "reddit"):
            prompt_instruction = f"""### Instruction:
                    You are a financial expert specializing in risk assessment for Nvidia stock recommendations.
                    Given the headline and content of a {source.lower()} article, assess the risk level for investing in Nvidia stock.
                    """
        else:
            prompt_instruction = f"""### Instruction:
                    You are a financial expert specializing in risk assessment for Nvidia stock recommendations.
                    Given the content of a tweet, assess the stock market condition and how it relates to the
                    risk level for investing in Nvidia stock.
                    """

        prompt_body = f"""
                    Assign a risk score from 1 to 5, where:
                    1 = very low risk,
                    2 = low risk,
                    3 = moderate risk (use this if risk is unclear),
                    4 = high risk,
                    5 = very high risk.

                    ### Generation Rules:
                        - Only output an integer between 1 and 5 inclusive.
                        - Do not include any other text besides the number.
                        - The risk score should be determined by analyzing the sentiment expressed in the input text.
                        - Consider both positive and negative sentiments when assessing the risk level.
                        - Avoid assigning scores solely based on specific keywords without considering their context within the text.
                        - Think step-by-step before making your decision.

                    ### Input:
                    Headline: {header}
                    Content: {content}

                    ### Response:
                    Risk score:"""

        prompt = prompt_instruction + prompt_body

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0,
            temperature=0.1,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )

        response = pipeline(prompt)[0]['generated_text']

        # Extract risk score from response
        match = re.search(r'(\d)\s*<\/think>', response)
        if not match:
            match = re.search(r'\b([1-5])\b', response)

        if match:
            score = int(match.group(1))
            # risk_scores.append(score)
            logging.info(f"Extracted risk score: {score}")
            return score
        else:
            logging.info("⚠️ Could not extract risk score from response:", response)
            # risk_scores.append(3)
            return 3

    except Exception as e:
        logging.info(f"[Error in risk score generation] -> {e}")
        sys.exit(1)


def get_all_scores(json_data):
    """Function to get risk score for every news in the processed json file.

    Args:
        json_data (json object): The loaded data from reading processed_data.json.

    Returns:
        risk_score (list): A list of generated risk score for each news extracted.
    """
    for i in tqdm(range(len(json_data)), desc="Generating risk scores"):
        logging.info(f"Data entry number {i}:\n")
        datetime = json_data[i]['datetime']
        source = json_data[i]['source']
        header = json_data[i]['header']
        content = json_data[i]['content']

        risk_score = get_risk_score(source, header, content)
        scored_articles.append(
            {
            "datetime": datetime,
            "source": source,
            "header": header,
            "content": content,
            "risk_score": risk_score
            }
        )
    
    # logging.info(f'\nAll risk scores: {risk_scores}')
    logging.info("Risk score generation completed.")
    
    return scored_articles


def save_tmp_csv(temp_df):
    """Function to save only datetime, source, and risk columns as temp/date_risk.csv for aggregation stage.

    Args:
        temp_df (pandas DataFrame): json_data loaded as pandas df.

    Returns:
        temp/date_risk.csv file with columns 'datetime', 'specific source', 'source', 'risk score'
    """
    df_new = temp_df[["datetime", "source", "specific_source", "risk score"]]
    df_new.to_csv(TEMP_DATE_RISK_CSV, index=False)
    
    
def append_score_to_csv(json_data, risk_scores, filename):
    """Function to get append the new data with its risk score to news_with_risk_score.csv for tracing purposes.

    Args:
        json_data (json object): The loaded data from reading processed_data.json.
        risk_scores (list): A list of generated risk score for each news extracted.
        filename (str): The path of news_with_risk_score.csv

    Returns:
        None
    """
    try:
        temp_df = pd.DataFrame(json_data)

        if len(risk_scores) == len(temp_df):
            # Add risk score as new column
            temp_df['risk score'] = risk_scores
            
            # Append new data with risk score to existing news_with_risk_score.csv
            temp_df.to_csv(filename, mode='a', header=False, index=False)
            logging.info(f"Appended to existing CSV: {filename}")
            
            # Save only datetime, source, specific source, and risk columns to temp/date_risk.csv for aggregation step.
            save_tmp_csv(temp_df)
            logging.info(f'Saved datetime and risk result to date_risk.csv')
            
    except Exception as e:
        logging.info(f"[Error in risk score generation] -> {e}")
        sys.exit(1)