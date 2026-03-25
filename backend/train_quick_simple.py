import numpy as np
import tensorflow as tf
import os
import joblib
from sklearn.model_selection import train_test_split
from train_audio_advanced import create_simple_model

print("Loading cached features for simple model training...")
X = np.load('data/processed/X_scaled_advanced.npy')
y_encoded = np.load('data/processed/y_encoded_advanced.npy')

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

num_emotions = len(np.unique(y_encoded))
print(f"Training simple model for {num_emotions} emotions on {X.shape[1]} features...")

model = create_simple_model(input_dim=X.shape[1], num_emotions=num_emotions)

# Fast training phase
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

os.makedirs('models', exist_ok=True)
save_path = 'models/audio_model_simple.h5'
model.save(save_path)
print(f"Successfully trained and saved simple 63-dim model to {save_path}")
