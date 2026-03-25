# backend/fusion_advanced.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import sys

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ConfidenceRegressor(layers.Layer):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.confidence_out = layers.Dense(1, activation='sigmoid')
        
    def call(self, features):
        x = self.dense1(features)
        x = self.dense2(x)
        return self.confidence_out(x)

class DynamicConfidenceFusion(Model):
    def __init__(self, audio_input_dim, text_input_dim, num_emotions=8):
        super().__init__()
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
        self.audio_confidence = ConfidenceRegressor()
        self.text_confidence = ConfidenceRegressor()
        self.fusion_layer = layers.Concatenate()
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_emotions, activation='softmax')
        ])
        
    def call(self, inputs, training=False):
        audio_feat, text_feat = inputs
        audio_encoded = self.audio_encoder(audio_feat)
        audio_conf = self.audio_confidence(audio_feat)
        text_encoded = self.text_encoder(text_feat)
        text_conf = self.text_confidence(text_feat)
        fused = self.fusion_layer([audio_encoded * audio_conf, text_encoded * text_conf])
        return self.classifier(fused), [audio_conf, text_conf]

def train_fusion_model(audio_features_path, text_features_path, labels_path):
    X_audio = np.load(audio_features_path)
    X_text = np.load(text_features_path)
    y = np.load(labels_path)
    
    fusion_model = DynamicConfidenceFusion(audio_input_dim=X_audio.shape[1], text_input_dim=X_text.shape[1], num_emotions=len(np.unique(y)))
    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = fusion_model.fit([X_audio, X_text], y, epochs=50, batch_size=32, validation_split=0.2)
    fusion_model.save('../models/fusion_model_advanced.h5')
    return fusion_model, history

if __name__ == "__main__":
    print("Advanced fusion module loaded.")