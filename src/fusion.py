import numpy as np

class MultimodalFusion:
    def __init__(self, audio_weight=0.6, text_weight=0.4):
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        # Mapping table for text emotions to audio indices
        # Consistent with UnifiedEmotionMapper
        self.text_to_audio_idx = {
            "happy": 2,
            "sad": 3,
            "angry": 4
        }

    def fuse(self, audio_probs, text_emotion_results, label_encoder_classes):
        """
        audio_probs: list/array of 8 probabilities from audio model.
        text_emotion_results: list of dicts [{'emotion': 'happy', 'score': 0.9}, ...]
        label_encoder_classes: list of emotion strings.
        Returns (final_emotion, final_confidence)
        """
        fused_probs = np.array(audio_probs)
        
        for tr in text_emotion_results:
            idx = self.text_to_audio_idx.get(tr['emotion'])
            if idx is not None:
                # Weighted combine: final_prob = audio_p * 0.6 + text_p * 0.4
                fused_probs[idx] = (fused_probs[idx] * self.audio_weight) + (tr['score'] * self.text_weight)
        
        # Re-normalize
        fused_probs = fused_probs / np.sum(fused_probs)
        
        final_idx = np.argmax(fused_probs)
        final_emotion = str(label_encoder_classes[final_idx])
        final_confidence = float(fused_probs[final_idx])
        
        return final_emotion, final_confidence