# src/text_analyzer.py
from transformers import pipeline
import torch

class TextEmotionAnalyzer:
    """
    A flexible text emotion analyzer that can use different Hugging Face models.
    Available models (key: (model_id, emotion_labels)):
        - 'emotion'     : bhadresh-savani/bert-base-uncased-emotion (6 emotions)
        - 'go_emotions' : SamLowe/roberta-base-go_emotions (28 emotions)
        - 'distilroberta': j-hartmann/emotion-english-distilroberta-base (7 emotions)
        - 'twitter'     : cardiffnlp/twitter-roberta-base-emotion (4 emotions: anger, joy, optimism, sadness)
    """
    # Registry of available models
    MODELS = {
        'emotion': {
            'id': "bhadresh-savani/bert-base-uncased-emotion",
            'labels': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        },
        'go_emotions': {
            'id': "SamLowe/roberta-base-go_emotions",
            'labels': ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
                       'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                       'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                       'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                       'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
                       'neutral']
        },
        'distilroberta': {
            'id': "j-hartmann/emotion-english-distilroberta-base",
            'labels': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        },
        'twitter': {
            'id': "cardiffnlp/twitter-roberta-base-emotion",
            'labels': ['anger', 'joy', 'optimism', 'sadness']
        }
    }

    def __init__(self, model_name='go_emotions', use_gpu=True):
        """
        Initialize the text emotion classifier.

        Args:
            model_name (str): Key from MODELS dict. Default 'go_emotions'.
            use_gpu (bool): Whether to use GPU if available.
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(self.MODELS.keys())}")

        self.model_name = model_name
        model_info = self.MODELS[model_name]
        self.emotions = model_info['labels']
        model_id = model_info['id']

        device = 0 if use_gpu and torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model=model_id,
            return_all_scores=True,
            device=device
        )
        print(f"Loaded text model: {model_name} ({model_id})")

    def analyze(self, text):
        """
        Predict emotion from text.

        Args:
            text (str): Input text.

        Returns:
            dict: Emotion names -> confidence scores (0-1). All emotions in self.emotions are included.
        """
        if not text.strip():
            return {emotion: 0.0 for emotion in self.emotions}

        results = self.classifier(text)[0]  # list of dicts: [{'label': 'joy', 'score': 0.98}, ...]

        # Convert to dict with all emotions present (fill missing with 0.0)
        probs = {item['label']: item['score'] for item in results}
        # Ensure all emotions in self.emotions are present (in case model output misses some)
        for emotion in self.emotions:
            if emotion not in probs:
                probs[emotion] = 0.0

        return probs