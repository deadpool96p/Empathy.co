from transformers import pipeline
import torch

import unicodedata

class ShortTextLexicon:
    """
    Lexicon fallback for high-signal emotional keywords in English, Hindi, and Marathi.
    Handles short inputs (1-3 words) with higher reliability than transformer models.
    """
    @staticmethod
    def normalize(text):
        if not text: return ""
        # Normalize to NFKC and lowercase
        return unicodedata.normalize('NFKC', str(text).strip().lower())

    # Pre-normalized lexicon sets for fast lookup
    _NORMALIZED_LEXICON = {
        "happy": {"happy", "joy", "glad", "yay", "excited", "awesome", "great", "wonderful", "खुश", "आनंद", "आनंदी", "मज्जा", "छान", "मस्त", "सुख", "हर्ष", "khush", "anand", "majja", "chhan", "mast"},
        "sad": {"sad", "upset", "depressed", "cry", "lonely", "unhappy", "blue", "pain", "उदास", "दुखी", "रडणे", "दु:खी", "खंत", "खिन्न", "वाईट", "त्रास", "udas", "dukhi", "radne", "vait", "tras"},
        "angry": {"angry", "mad", "furious", "rage", "hate", "irritated", "annoyed", "गुस्सा", "राग", "चिडलो", "ताप", "क्रोध", "तिडक", "संताप", "guissa", "raag", "chidlo", "santap", "gussa", "gvssa", "rak"},
        "fearful": {"fear", "scared", "afraid", "panic", "worry", "horror", "spooky", "डर", "भय", "घाबरलो", "भीती", "धडकी", "घाबरणे", "dar", "bheeti", "ghabarlo"},
        "disgust": {"disgust", "gross", "ew", "yuck", "vile", "revolting", "shame", "घृणा", "नफरत", "वीट", "तिरस्कार", "घाण", "ghrina", "nafrat", "veet", "ghan"},
        "surprised": {"wow", "surprise", "shock", "amazing", "unbelievable", "omg", "अचरज", "आश्चर्य", "नवल", "थक्क", "achraj", "ashcharya", "navala"},
        "calm": {"calm", "peace", "chill", "relax", "serene", "quiet", "steady", "शांत", "स्थिर", "विश्रांती", "समाधान", "shant", "sthir", "samadhan"}
    }

    @classmethod
    def match(cls, text):
        """
        Returns a high-confidence match if the text contains a known emotional keyword.
        """
        raw_words = text.lower().strip().split()
        normalized_words = [cls.normalize(w).strip(".,!?\"'") for w in raw_words]
        
        for clean_word in normalized_words:
            if not clean_word: continue
            for emotion, keywords in cls._NORMALIZED_LEXICON.items():
                # Check for direct match or normalized match
                if clean_word in keywords:
                    return emotion
                # Secondary check: check each keyword normalized
                for kw in keywords:
                    if cls.normalize(kw) == clean_word:
                        return emotion
        return None

class TextEmotionAnalyzer:
    """
    Multilingual Text Emotion Analyzer utilizing Hugging Face Transformers and Lexicon fallback.
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
        if not text.strip():
            return []

        # 1. Lexicon Fallback for short inputs (1-3 words)
        word_count = len(text.split())
        if word_count <= 3:
            lex_match = ShortTextLexicon.match(text)
            if lex_match:
                print(f"[LEXICON MATCH] Triggered '{lex_match}' for input: '{text}'")
                return [{
                    "emotion": lex_match,
                    "score": 0.90 # High confidence for direct keyword matches
                }]

        # 2. Transformer Pipeline Strategy
        if not self.classifier:
            return []

        results = self.classifier(text)[0]
        
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