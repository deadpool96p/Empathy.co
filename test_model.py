# test_model.py
import numpy as np
import joblib
import tensorflow as tf
from src.audio_processor import AudioFeatureExtractor

# Load model, scaler, and label encoder
model = tf.keras.models.load_model('models/audio_model.h5')
scaler = joblib.load('models/scaler.pkl')
label_encoder = np.load('data/processed/label_encoder.npy', allow_pickle=True)

extractor = AudioFeatureExtractor()

# Test with a sample file (change path to any .wav from your datasets)
file_path = "data/raw/ravdess/Actor_01/03-01-01-01-01-01-01.wav"  # neutral
y, sr = extractor.load_audio(file_path)
features = extractor.extract_features(y, sr)
features_scaled = scaler.transform([features])
pred_probs = model.predict(features_scaled)[0]
pred_idx = np.argmax(pred_probs)
pred_emotion = label_encoder[pred_idx]
confidence = pred_probs[pred_idx]

print(f"Predicted emotion: {pred_emotion} ({confidence:.2f})")
print("All probabilities:")
for i, prob in enumerate(pred_probs):
    print(f"  {label_encoder[i]}: {prob:.3f}")