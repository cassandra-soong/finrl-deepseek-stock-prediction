from processing import process
from training import train_a2c, train_sac
from inference import get_inference
from config import LOG_FILE
import sys
import logging

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

logging.info("FinRL started.")


if __name__ == "__main__":
    try:
        '''
        Stage 1: processing.py
        '''
        process()
        logging.info("Stage 1: Processing completed.")
        logging.info("---------------------------------------------")
        
        '''
        Stage 2: training.py
        
        - Retrain the model if necessary
        - Skipped for now, using pre-trained model
        '''
        # train_a2c()
        # train_sac()
        logging.info("Stage 2: Training skipped.")
        logging.info("---------------------------------------------")
        
        '''
        Stage 3: inference.py
        '''
        get_inference()
        logging.info("Stage 3: Inference completed.")
        logging.info("---------------------------------------------")
        

    except Exception as e:
        logging.info(f"Error in main occured -> {e}")
        sys.exit(1)