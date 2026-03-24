# src/fusion.py
import numpy as np

class MultimodalFusion:
    def __init__(self, audio_weight=0.6, text_weight=0.4):
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        # Mapping from audio emotion indices (0-7) to text emotion names
        self.audio_to_text = {
            0: None,      # neutral – no direct match
            1: 'joy',     # calm
            2: 'joy',     # happy
            3: 'sadness', # sad
            4: 'anger',   # angry
            5: 'fear',    # fearful
            6: None,      # disgust – not in text model
            7: 'surprise' # surprised
        }

    def fuse(self, audio_probs, text_probs_dict):
        """
        audio_probs: numpy array of 8 probabilities from audio model.
        text_probs_dict: dict from text model, keys are emotion names.
        Returns a dict of fused scores for text emotions.
        """
        fused = {}
        for text_emotion, text_score in text_probs_dict.items():
            # Find corresponding audio index
            audio_idx = None
            for idx, name in self.audio_to_text.items():
                if name == text_emotion:
                    audio_idx = idx
                    break
            audio_score = audio_probs[audio_idx] if audio_idx is not None else 0.0
            fused[text_emotion] = self.audio_weight * audio_score + self.text_weight * text_score
        return fused