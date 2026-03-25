from transformers import pipeline
import torch

class TextEmotionAnalyzer:
    """
    Multilingual Text Emotion Analyzer utilizing Hugging Face Transformers.
    """
    def __init__(self, model_id="MilaNLProc/xlm-emo-t", use_gpu=True):
        self.model_id = model_id
        device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_id,
                top_k=None,
                device=device
            )
            print(f"Loaded multilingual text model: {model_id}")
        except Exception as e:
            print(f"Failed to load text pipeline {model_id}: {e}")
            self.classifier = None

        # Standardize output naming
        self.mapping = {
            "joy": "happy", 
            "sadness": "sad", 
            "anger": "angry",
            "optimism": "happy"
        }

    def analyze(self, text):
        if not self.classifier or not text.strip():
            return []

        results = self.classifier(text)[0] # returns list of dicts [{'label': 'joy', 'score': 0.99}, ...]
        
        processed_results = []
        for res in results:
            mapped_label = self.mapping.get(res['label'], res['label'])
            processed_results.append({
                "emotion": mapped_label,
                "score": float(res['score'])
            })
        
        # Sort by score descending
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        return processed_results