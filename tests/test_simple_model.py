import numpy as np
import joblib
import tensorflow as tf
import librosa
import os

from train_audio_advanced import EnhancedFeatureExtractor

# Load the simple model (assuming it exists as audio_model.h5)
model_path = 'models/audio_model.h5'
scaler_path = 'models/scaler.pkl'   # Let's hope scaler.pkl exists, or we use scaler_advanced.pkl

if not os.path.exists(model_path):
    print(f"Simple model not found at {model_path}.")
    exit(1)

model = tf.keras.models.load_model(model_path)

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    print(f"Scaler not found at {scaler_path}. Using scaler_advanced.pkl")
    scaler = joblib.load('models/scaler_advanced.pkl')

extractor = EnhancedFeatureExtractor()

# Test on a dummy audio array
print("Testing on dummy audio...")
y = np.random.randn(16000 * 3) # 3 seconds of noise
sr = 16000
import soundfile as sf
os.makedirs("temp", exist_ok=True)
audio_path = "temp/dummy.wav"
sf.write(audio_path, y, sr)

try:
    feat = extractor.extract(audio_path)
    feat_scaled = scaler.transform([feat])
    probs = model.predict(feat_scaled)[0]
    emotion_idx = np.argmax(probs)
    print(f"Predicted emotion index: {emotion_idx}")
    print(f"Probabilities: {probs}")
except Exception as e:
    print(f"Error extracting features: {e}")

if os.path.exists(audio_path):
    os.remove(audio_path)
