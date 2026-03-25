import numpy as np
import tensorflow as tf
import os
import sys

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import EnhancedFeatureExtractor
from src.model_trainer import TimeShuffleAttention, LightweightConvTransformer

# Paths relative to backend/ directory
MODEL_PATH = "../models/audio_model_simple.h5"

def test_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    custom_objs = {
        'TimeShuffleAttention': TimeShuffleAttention,
        'LightweightConvTransformer': LightweightConvTransformer
    }
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objs)
        print("Model loaded successfully.")
        
        # Dummy data for testing (63 features)
        dummy_input = np.random.rand(1, 63).astype(np.float32)
        preds = model.predict(dummy_input)
        print(f"Prediction successful. Output shape: {preds.shape}")
        print(f"Probabilities: {preds}")
    except Exception as e:
        print(f"Inference test failed: {e}")

if __name__ == "__main__":
    test_inference()