# backend/analyze_advanced_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import itertools

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import UnifiedEmotionMapper, EnhancedFeatureExtractor
from src.model_trainer import TimeShuffleAttention, LightweightConvTransformer

def create_advanced_model_custom(input_dim, num_emotions, learning_rate, num_blocks, use_attention):
    transformer_dim = 128
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)

    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    if use_attention:
        x = TimeShuffleAttention(reduction=8)(x)

    for _ in range(num_blocks):
        x = LightweightConvTransformer(dim=transformer_dim, num_heads=4)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_emotions, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def analyze_model_configurations(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, num_emotions):
    learning_rates = [1e-3, 1e-4, 1e-5]
    num_blocks = [1, 2]
    use_attention = [True, False]

    results = []
    for lr, blocks, att in itertools.product(learning_rates, num_blocks, use_attention):
        print(f"Testing: lr={lr}, blocks={blocks}, attention={att}")
        model = create_advanced_model_custom(input_dim, num_emotions, lr, blocks, att)
        history = model.fit(X_train_scaled, y_train, batch_size=32, epochs=5, validation_data=(X_val_scaled, y_val), verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        results.append({'lr': lr, 'blocks': blocks, 'attention': att, 'val_accuracy': val_acc})
        tf.keras.backend.clear_session()

    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.sort_values('val_accuracy', ascending=False).to_string(index=False))
    return df

if __name__ == "__main__":
    raw_feat_path = '../data/processed/X_raw.npy'
    raw_label_path = '../data/processed/y_raw.npy'

    if not os.path.exists(raw_feat_path):
        print("Raw features not found. Run train_audio_advanced.py first.")
        sys.exit(1)

    X, y = np.load(raw_feat_path), np.load(raw_label_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    analyze_model_configurations(X_train_scaled, y_train, X_val_scaled, y_val, X.shape[1], len(le.classes_))