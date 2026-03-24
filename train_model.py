# train_model.py
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # <-- NEW IMPORT
import joblib
import tensorflow as tf
from src.audio_processor import AudioFeatureExtractor
from src.model_trainer import create_audio_model
import os

# Create necessary directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Paths to your datasets
RAVDESS_PATH = "data/raw/ravdess"
TESS_PATH = "data/raw/tess"
CREMAD_PATH = "data/raw/cremad"

print("Loading datasets and extracting features...")
extractor = AudioFeatureExtractor()
X, y = extractor.load_all_datasets(RAVDESS_PATH, TESS_PATH, CREMAD_PATH)

print(f"Total samples: {len(X)}")
print(f"Feature dimension: {X.shape[1]}")
print(f"Class distribution: {np.bincount(y)}")

# Encode labels if they aren't already integers (y should already be ints from our mapping)
# But just in case, let's ensure they are consistent
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # if y are strings, else use y directly
np.save('data/processed/label_encoder.npy', le.classes_)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use in the app
joblib.dump(scaler, 'models/scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---- NEW: Compute class weights for imbalanced data ----
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Create and train model
model = create_audio_model(input_shape=(X.shape[1],), num_emotions=len(np.unique(y_encoded)))
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,  # <-- ADDED CLASS WEIGHTS
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('models/audio_model.h5', save_best_only=True)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save final model
model.save('models/audio_model.h5')
print("Training complete. Model saved to models/audio_model.h5")