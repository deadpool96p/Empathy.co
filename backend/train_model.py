import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import EnhancedFeatureExtractor, UnifiedEmotionMapper
from src.model_trainer import create_simple_model

# Paths relative to backend/ directory
RAVDESS_PATH = "../data/raw/ravdess"
TESS_PATH = "../data/raw/tess"
CREMAD_PATH = "../data/raw/cremad"

os.makedirs('../models', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)

def main():
    print("Loading datasets and extracting features...")
    # Using local collectors from train_audio_advanced logic if needed, 
    # but here we'll just demonstrate simple loading if possible.
    # For now, let's assume we use the advanced collector logic from train_audio_advanced.
    from train_audio_advanced import collect_ravdess, collect_tess, collect_cremad
    
    mapper = UnifiedEmotionMapper()
    extractor = EnhancedFeatureExtractor()
    
    all_paths, all_labels = [], []
    for path, collector in [(RAVDESS_PATH, collect_ravdess), (TESS_PATH, collect_tess), (CREMAD_PATH, collect_cremad)]:
        if os.path.exists(path):
            p, l = collector(path, mapper)
            all_paths.extend(p); all_labels.extend(l)
    
    X_list, y_list = [], []
    for p, l in zip(all_paths, all_labels):
        X_list.append(extractor.extract(p))
        y_list.append(l)
    
    X, y = np.array(X_list), np.array(y_list)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    np.save('../data/processed/label_encoder.npy', le.classes_)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, '../models/scaler.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    model = create_simple_model(input_dim=X.shape[1], num_emotions=len(le.classes_))
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
              class_weight=dict(enumerate(cw)),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    
    model.save('../models/audio_model.h5')
    print("Training complete. Model saved to ../models/audio_model.h5")

if __name__ == "__main__":
    main()