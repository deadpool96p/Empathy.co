# fusion_advanced.py
"""
Advanced Multimodal Fusion with Confidence-Aware Weighting and Knowledge Distillation
Based on research from:
- Dynamic confidence-aware fusion (Zhu et al., 2024) [citation:4][citation:9]
- Knowledge distillation for emotion recognition (Stevens et al., 2025) [citation:8]
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib
from sklearn.preprocessing import StandardScaler


# ============================================
# 1. CONFIDENCE REGRESSION NETWORK
# ============================================

class ConfidenceRegressor(layers.Layer):
    """
    Estimates prediction uncertainty for each modality
    Based on True Class Probability (TCP) estimation [citation:4]
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.confidence_out = layers.Dense(1, activation='sigmoid')
        
    def call(self, features):
        x = self.dense1(features)
        x = self.dense2(x)
        confidence = self.confidence_out(x)
        return confidence


# ============================================
# 2. DYNAMIC CONFIDENCE-AWARE FUSION NETWORK
# ============================================

class DynamicConfidenceFusion(Model):
    """
    Complete fusion network that:
    1. Estimates confidence for each modality
    2. Dynamically weights modalities based on confidence
    3. Handles missing modalities gracefully
    """
    def __init__(self, audio_input_dim, text_input_dim, num_emotions=8):
        super().__init__()
        
        # Modality encoders
        self.audio_encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
        ])
        
        self.text_encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
        ])
        
        # Confidence regressors
        self.audio_confidence = ConfidenceRegressor()
        self.text_confidence = ConfidenceRegressor()
        
        # Fusion layer
        self.fusion_layer = layers.Concatenate()
        
        # Final classifier
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_emotions, activation='softmax')
        ])
        
    def call(self, inputs, training=False):
        audio_feat, text_feat = inputs
        
        # Handle missing modalities
        audio_available = audio_feat is not None
        text_available = text_feat is not None
        
        fused_features = []
        confidence_weights = []
        
        if audio_available:
            audio_encoded = self.audio_encoder(audio_feat)
            audio_conf = self.audio_confidence(audio_feat)
            fused_features.append(audio_encoded * audio_conf)
            confidence_weights.append(audio_conf)
        
        if text_available:
            text_encoded = self.text_encoder(text_feat)
            text_conf = self.text_confidence(text_feat)
            fused_features.append(text_encoded * text_conf)
            confidence_weights.append(text_conf)
        
        if len(fused_features) == 1:
            # Single modality
            fused = fused_features[0]
        else:
            # Concatenate weighted features
            fused = self.fusion_layer(fused_features)
        
        # Classify
        output = self.classifier(fused)
        
        return output, confidence_weights


# ============================================
# 3. KNOWLEDGE DISTILLATION FOR SMALLER MODELS
# ============================================

class DistilledEmotionModel:
    """
    Creates a smaller, faster student model by distilling knowledge
    from a larger teacher ensemble [citation:8]
    """
    def __init__(self, teacher_model, input_dim, num_emotions=8, temperature=3.0):
        self.teacher = teacher_model
        self.temperature = temperature
        self.num_emotions = num_emotions
        
        # Create student model (smaller architecture)
        self.student = self._create_student(input_dim, num_emotions)
        
    def _create_student(self, input_dim, num_emotions):
        """Smaller, faster model for deployment"""
        inputs = layers.Input(shape=(input_dim,))
        
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_emotions, activation='softmax')(x)
        
        return Model(inputs, outputs)
    
    def distillation_loss(self, y_true, y_pred):
        """
        Combined loss: standard cross-entropy + distillation loss
        """
        # Standard loss
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Distillation loss (using teacher's soft targets)
        teacher_logits = self.teacher.predict(y_true)  # Simplified - needs actual teacher predictions
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
        student_soft = tf.nn.softmax(y_pred / self.temperature)
        distill_loss = tf.keras.losses.kl_divergence(teacher_soft, student_soft)
        
        return 0.5 * ce_loss + 0.5 * (self.temperature ** 2) * distill_loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        self.student.compile(
            optimizer='adam',
            loss=self.distillation_loss,
            metrics=['accuracy']
        )
        
        history = self.student.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        return history


# ============================================
# 4. MODEL QUANTIZATION FOR EFFICIENCY
# ============================================

def quantize_model_for_deployment(model, representative_dataset, model_name):
    """
    Convert model to TensorFlow Lite with 16x8 quantization [citation:5]
    Reduces model size by ~75% while maintaining accuracy
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 16x8 quantization (activations: 16-bit, weights: 8-bit)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    
    # Representative dataset for calibration
    def representative_dataset_gen():
        for sample in representative_dataset.take(100):
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset_gen
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(f'models/{model_name}_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Quantized model saved to models/{model_name}_quantized.tflite")
    print(f"Original size: {model.count_params() * 4 / 1024:.2f} KB")
    print(f"Quantized size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model


# ============================================
# 5. MAIN FUSION TRAINING SCRIPT
# ============================================

def train_fusion_model(audio_features_path, text_features_path, labels_path):
    """
    Train the complete dynamic confidence-aware fusion model
    """
    # Load data
    print("Loading features...")
    X_audio = np.load(audio_features_path)
    X_text = np.load(text_features_path)
    y = np.load(labels_path)
    
    print(f"Audio features: {X_audio.shape}")
    print(f"Text features: {X_text.shape}")
    print(f"Labels: {y.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
    X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create fusion model
    print("\nCreating Dynamic Confidence-Aware Fusion Model...")
    fusion_model = DynamicConfidenceFusion(
        audio_input_dim=X_audio.shape[1],
        text_input_dim=X_text.shape[1],
        num_emotions=len(np.unique(y))
    )
    
    # Custom training loop (simplified - use model.fit with custom loss in practice)
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\nTraining fusion model...")
    history = fusion_model.fit(
        [X_audio_train, X_text_train],
        y_train,
        validation_data=([X_audio_test, X_text_test], y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Evaluate
    test_loss, test_acc = fusion_model.evaluate([X_audio_test, X_text_test], y_test)
    print(f"\nFusion model test accuracy: {test_acc:.4f}")
    
    # Save
    fusion_model.save('models/fusion_model_advanced.h5')
    print("Fusion model saved to models/fusion_model_advanced.h5")
    
    return fusion_model, history


if __name__ == "__main__":
    # Example usage - you'll need actual feature files
    # train_fusion_model(
    #     'data/processed/audio_features.npy',
    #     'data/processed/text_features.npy',
    #     'data/processed/labels.npy'
    # )
    
    print("Advanced fusion module loaded.")
    print("To use: train_fusion_model(audio_feats, text_feats, labels)")