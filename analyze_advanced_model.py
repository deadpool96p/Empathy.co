# analyze_advanced_model.py
"""
Diagnostic script to test different configurations of the advanced CNN‑Transformer model
on the emotion recognition dataset. Helps identify why the advanced model underperforms.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import itertools

# Import all required classes from your existing training script
# Make sure train_audio_advanced.py is in the same directory or in PYTHONPATH
try:
    from train_audio_advanced import (
        UnifiedEmotionMapper,
        EnhancedFeatureExtractor,
        AdvancedAudioAugmenter,
        collect_ravdess,
        collect_tess,
        collect_cremad,
        collect_iemocap,
        collect_emodb,
        collect_meld,
        TimeShuffleAttention,
        LightweightConvTransformer
    )
except ImportError as e:
    print("Error: Could not import from train_audio_advanced.py")
    print("Make sure the file is in the same directory and contains all class definitions.")
    raise e

# ============================================
# Custom model builder with flexible parameters
# ============================================
def create_advanced_model_custom(input_dim, num_emotions, learning_rate, num_blocks, use_attention):
    """
    Build advanced model with flexible hyperparameters.
    Transformer dimension is fixed at 128 to match the CNN output.
    """
    transformer_dim = 128  # fixed, because CNN output is 128

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)

    # CNN blocks (same as before)
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

    # Optional TimeShuffleAttention
    if use_attention:
        x = TimeShuffleAttention(reduction=8)(x)

    # Transformer blocks
    for _ in range(num_blocks):
        x = LightweightConvTransformer(dim=transformer_dim, num_heads=4)(x)

    x = layers.GlobalAveragePooling1D()(x)

    # Dense head
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


# ============================================
# Configuration testing function
# ============================================
def analyze_model_configurations(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, num_emotions):
    """
    Test multiple configurations of the advanced model on a small subset of data.
    Returns a DataFrame with results.
    """
    # Hyperparameter grid
    learning_rates = [1e-3, 1e-4, 1e-5]
    num_blocks = [1, 2]                     # number of transformer blocks
    use_attention = [True, False]            # whether to include TimeShuffleAttention

    results = []
    total_configs = len(learning_rates) * len(num_blocks) * len(use_attention)
    print(f"\nTesting {total_configs} configurations (training for 5 epochs each)...")

    for lr, blocks, att in itertools.product(learning_rates, num_blocks, use_attention):
        print(f"\n--- Testing: lr={lr}, blocks={blocks}, attention={att} ---")

        # Build model with these parameters
        model = create_advanced_model_custom(
            input_dim=input_dim,
            num_emotions=num_emotions,
            learning_rate=lr,
            num_blocks=blocks,
            use_attention=att
        )

        # Train for a few epochs
        history = model.fit(
            X_train_scaled, y_train,
            batch_size=32,
            epochs=5,
            validation_data=(X_val_scaled, y_val),
            verbose=0
        )

        val_acc = history.history['val_accuracy'][-1]  # last epoch accuracy
        results.append({
            'lr': lr,
            'blocks': blocks,
            'attention': att,
            'val_accuracy': val_acc
        })

        # Clear session to free memory
        tf.keras.backend.clear_session()

    # Convert to DataFrame
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("Configuration test results (validation accuracy after 5 epochs):")
    print(df.sort_values('val_accuracy', ascending=False).to_string(index=False))
    return df


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Dataset paths (same as in training script)
    config = {
        'datasets': {
            'ravdess': 'data/raw/ravdess',
            'tess': 'data/raw/tess',
            'cremad': 'data/raw/cremad',
            'iemocap': 'data/raw/iemocap',
            'emodb': 'data/raw/emodb',
            'meld': 'data/raw/meld',
        },
        'use_augmentation': False,
        'augment_factor': 0,
        'raw_features_path': 'data/processed/X_raw.npy',
        'raw_labels_path': 'data/processed/y_raw.npy',
    }

    os.makedirs('data/processed', exist_ok=True)

    # Load cached raw features (must exist from previous run of train_audio_advanced.py)
    if os.path.exists(config['raw_features_path']) and os.path.exists(config['raw_labels_path']):
        print("Loading cached raw features...")
        X = np.load(config['raw_features_path'])
        y = np.load(config['raw_labels_path'])
    else:
        print("Features not cached. Please run train_audio_advanced.py first (with use_simple_model=True) to extract features.")
        exit(1)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Run analysis
    df_results = analyze_model_configurations(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        input_dim=X.shape[1],
        num_emotions=len(np.unique(y_encoded))
    )

    # Optionally, train the best configuration fully (you can extend this)
    best_row = df_results.loc[df_results['val_accuracy'].idxmax()]
    print("\n" + "="*60)
    print("Best configuration found:")
    print(best_row)
    print("\nYou can now train the full model with these parameters.")