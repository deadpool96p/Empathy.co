import tensorflow as tf
from tensorflow.keras import layers, models

def create_audio_model(input_shape, num_emotions=8):
    """CNN + LSTM hybrid model for audio emotion recognition"""
    model = models.Sequential([
        # Reshape for 1D CNN (adds channel dimension)
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),

        # Convolutional layers for local feature extraction
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),

        # LSTM for temporal patterns
        layers.LSTM(64, return_sequences=False),

        # Dense layers for classification
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_emotions, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model