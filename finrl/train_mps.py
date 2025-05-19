import os
import torch
from training import train
from training_config import DEVICE, AVAILABLE_ALGORITHMS

def main():
    """
    Run the training pipeline with MPS/GPU support.
    """
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {DEVICE}")
    
    # Create necessary directories
    os.makedirs("finrl/models", exist_ok=True)
    os.makedirs("finrl/models/logs", exist_ok=True)
    os.makedirs("finrl/models/checkpoints", exist_ok=True)
    
    # Train each algorithm
    results = {}
    for algo in AVAILABLE_ALGORITHMS:
        print(f"\n{'='*50}")
        print(f"Training {algo.upper()} model...")
        print(f"{'='*50}")
        
        try:
            result = train(algorithm=algo, optimize=True)
            results[algo] = result
            
            print(f"\nTraining completed for {algo.upper()}:")
            print(f"Model saved at: {result['model_path']}")
            print("\nMetrics:")
            for metric, value in result['metrics'].items():
                print(f"- {metric}: {value:.4f}")
            
        except Exception as e:
            print(f"Error training {algo}: {str(e)}")
            continue
    
    print("\nTraining completed for all algorithms!")
    return results

if __name__ == "__main__":
    main() 