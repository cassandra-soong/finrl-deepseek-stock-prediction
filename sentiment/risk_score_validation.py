# sentiment_validation.py
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import VALIDATION_LLM, HF_TOKEN, device, MODEL_CACHE_DIR, LOG_FILE
from risk_score_generation import get_risk_score
import torch
import re
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def validation_pipeline_setup():
    """
    Load LLM and tokenizer from HuggingFace Hub for validation.

    Returns:
        pipeline (pipeline object): The text generation pipeline for validation.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(VALIDATION_LLM, trust_remote_code=True)
        
        # Ensure padding token is set for models like GPT-2 if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            VALIDATION_LLM,
            torch_dtype=torch.float16,  # Use float32 if float16 causes issues or for CPU
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )

        validation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            top_k=10,
            num_return_sequences=1,
            max_new_tokens=50,  # Adjust as needed for the length of the validation response
            return_full_text=False,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id # Set pad_token_id
        )

        logging.info("Validation pipeline loaded successfully.")
        return validation_pipeline
    except Exception as e:
        logging.info(f"Error in validation pipeline loading: {e}")
        return None

def validate_risk_score(val_pipeline, header, content, assigned_score):
    """
    Validates if the assigned risk score is appropriate for the given news article
    using an LLM.

    Args:
        val_pipeline (pipeline object): The text generation pipeline for validation.
        header (str): Header/title of the news article.
        content (str): Content/body of the news article.
        assigned_score (int): The pre-assigned risk score (1-5).

    Returns:
        tuple: (is_valid (bool), explanation (str))
               is_valid is True if the score is deemed valid, False otherwise.
               explanation contains the LLM's reasoning.
    """
    if val_pipeline is None:
        logging.info("Validation pipeline not loaded.")
        val_pipeline = validation_pipeline_setup()
        if val_pipeline is None:
            raise ValueError("Validation pipeline setup failed.")

    try:
        prompt_instruction = f"""### Instruction:
        You are a meticulous financial analyst. Your task is to validate a pre-assigned risk score for Nvidia stock based on the provided news article.
        The risk score ranges from 1 to 5:
        1 = Very Low Risk (Strongly positive sentiment, clear tailwinds for Nvidia)
        2 = Low Risk (Generally positive sentiment, minor or distant concerns for Nvidia)
        3 = Moderate Risk (Mixed sentiment, unclear impact, or balanced positive/negative factors for Nvidia)
        4 = High Risk (Generally negative sentiment, significant headwinds or threats for Nvidia)
        5 = Very High Risk (Strongly negative sentiment, severe immediate threats or problems for Nvidia)

        Analyze the sentiment and implications of the news article (headline and content) concerning Nvidia.
        Then, evaluate if the 'Assigned Risk Score' is appropriate.

        Respond with only 'VALID' or 'INVALID'. After your 'VALID' or 'INVALID' response, provide a brief, one-sentence explanation for your decision. Do not add any other text.

        ### Input:
        Headline: '{header}'
        Content: '{content}'
        Assigned Risk Score: '{assigned_score}'

        ### Response:
        Validation Result:"""

        response_text = val_pipeline(prompt_instruction)[0]['generated_text'].strip()
        logging.info(f"\nRaw Validation Response for score {assigned_score} ('{header}'):\n'{response_text}'")

        # Regex to capture VALID/INVALID and the explanation
        match = re.match(r"(VALID|INVALID)\s*[:.-]*\s*(.*)", response_text, re.IGNORECASE)

        if match:
            validation_status_str = match.group(1).upper()
            explanation = match.group(2).strip()
            is_valid = validation_status_str == "VALID"
            
            if not explanation: # If explanation is empty, try a simpler parse
                if "VALID" in response_text.upper():
                    is_valid = True
                    explanation = response_text.upper().replace("VALID", "").strip()
                elif "INVALID" in response_text.upper():
                    is_valid = False
                    explanation = response_text.upper().replace("INVALID", "").strip()
                else: # Fallback if parsing fails
                    is_valid = False # Default to invalid if unclear
                    explanation = "Could not clearly parse validation status. Raw: " + response_text
            
            return is_valid, explanation
        else:
            # Fallback if the primary regex doesn't match - check for keywords
            if "VALID" in response_text.upper():
                return True, "Response indicates 'VALID', but formatting was unexpected. Raw: " + response_text
            elif "INVALID" in response_text.upper():
                return False, "Response indicates 'INVALID', but formatting was unexpected. Raw: " + response_text
            else:
                return False, "Could not parse validation status from response. Raw: " + response_text

    except Exception as e:
        logging.info(f"[Error in risk score validation] -> {e}")
        return False, f"Exception during validation: {e}"

def validate_all_scores(scored_articles):
    """Main function to perform sentiment validation."""
    # --- 0. Login to Hugging Face Hub ---
    login(token=HF_TOKEN)

    # --- 1. Setup Validation Pipeline ---
    validation_ppl = validation_pipeline_setup()

    if validation_ppl:
        logging.info("\n--- Starting Sentiment Validation Task ---")
        logging.info("------------------------------------")
        for i, scored_article in enumerate(scored_articles):
            risk_score = scored_article["risk_score"]
            if risk_score:
                is_valid, explanation = validate_risk_score(
                    validation_ppl,
                    scored_article["header"],
                    scored_article["content"],
                    risk_score
                )

                logging.info(f"\nValidating Article {i+1}: '{scored_article['header']}'")
                logging.info(f"Assigned Score: {risk_score}")
                logging.info(f"\nValidation Status: {'VALID' if is_valid else 'INVALID'}")
                logging.info(f"Explanation: {explanation}")
                logging.info("------------------------------------\n")

                scored_articles[i].update({
                    "is_valid": is_valid,
                    "explanation": explanation
                })
            else:
                logging.info(f"⚠️ No risk score found for this article. Skipping validation for Article {i+1}: {scored_article['header']}")
                continue
            
        return scored_articles
    else:
        logging.info("Could not start validation task as the pipeline failed to load.")

def regeneration(validated_articles):
    logging.info("\n--- Starting Re-generation check ---")
    logging.info("------------------------------------")
    for i, validated_article in enumerate(validated_articles):
        if not validated_article["is_valid"]:
            logging.info(f"\nRe-generating risk score for invalidated Article {i+1}: {validated_article['header']}")
            regenerated_score = get_risk_score(validated_article["source"], validated_article["header"], validated_article["content"])
            logging.info(f"\nRegenerated Score: {regenerated_score}")
            validated_articles[i]["risk_score"] = regenerated_score
            validated_articles[i]["is_valid"] = "Re-generated / No re-validation performed."
            validated_articles[i]["explanation"] += "\nSCORE REGENERATED DUE TO INVALIDATION"
            logging.info("------------------------------------")

        else:
            logging.info(f"No need to regenerate for Validated Article {i+1}: {validated_article['header']}")
            logging.info("------------------------------------")
            continue

    return validated_articles