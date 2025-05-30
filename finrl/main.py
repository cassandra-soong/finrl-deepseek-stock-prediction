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
        print("Stage 1: Processing completed.")
        print("---------------------------------------------")
        
        '''
        Stage 2: training.py
        
        - Retrain the model if necessary
        - Skipped for now, using pre-trained model
        '''
        # train_a2c()
        # train_sac()
        print("Stage 2: Training skipped.")
        print("---------------------------------------------")
        
        '''
        Stage 3: inference.py
        '''
        get_inference()
        print("Stage 3: Inference completed.")
        print("---------------------------------------------")
        

    except Exception as e:
        print(f"Error in main occured -> {e}")
        sys.exit(1)