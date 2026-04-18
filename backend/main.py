from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import uvicorn
import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import joblib
import uuid
import datetime
import csv
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import EnhancedFeatureExtractor
from src.model_trainer import TimeShuffleAttention, LightweightConvTransformer
from src.text_analyzer import TextEmotionAnalyzer
from src.fusion import MultimodalFusion
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

class EmotionSummarizer:
    TEMPLATES = {
        "neutral": [
            "The speaker maintains a steady and composed demeanor, showing no strong emotional bias.",
            "A balanced and objective tone is observed, with little variation in emotional intensity."
        ],
        "calm": [
            "The interaction is serene and relaxed, reflecting a high degree of emotional stability.",
            "A peaceful and gentle tone suggests a state of comfort and ease."
        ],
        "happy": [
            "The content is noticeably upbeat and positive, radiating a sense of joy and optimism.",
            "High spirits and enthusiasm are evident, indicating a very pleasant emotional state."
        ],
        "sad": [
            "A heavy and melancholic tone is present, suggesting feelings of sorrow or loss.",
            "The expression reflects a subdued and dejected mood, often associated with disappointment."
        ],
        "angry": [
            "The tone is sharp and confrontational, indicating significant frustration or hostility.",
            "A tense and aggressive emotional state is clear, directed towards a specific issue or person."
        ],
        "fearful": [
            "The speaker sounds anxious and apprehensive, suggesting a high level of stress or worry.",
            "A sense of vulnerability and unease is present, as if facing a perceived threat."
        ],
        "disgust": [
            "The expression conveys strong disapproval and aversion, often related to something offensive.",
            "A dismissive and repelled tone is observed, indicating a deep sense of distaste."
        ],
        "surprised": [
            "The tone reflects sudden astonishment or shock, responding to something unexpected.",
            "A high-arousal reaction is evident, indicating a significant break from normal expectations."
        ]
    }

    def generate(self, text, emotion, confidence):
        import random
        emotion = emotion.lower()
        template = random.choice(self.TEMPLATES.get(emotion, self.TEMPLATES["neutral"]))
        
        confidence_str = f"with {int(confidence * 100)}% certainty"
        summary = f"{template} Based on the analysis, the dominant emotion is {emotion} ({confidence_str})."
        
        # If text is available and long enough, add a small contextual bit
        if text and len(text.split()) > 3:
            # Simple keyword extraction (very basic)
            words = [w for w in text.split() if len(w) > 4]
            if words:
                summary += f" The language used points towards themes of '{words[0]}'."
                
        return summary

class LanguageTranslator:
    def __init__(self):
        self.pipelines = {}

    def translate(self, text, src_lang):
        if src_lang == 'en':
            return text
            
        model_map = {
            'hi': 'Helsinki-NLP/opus-mt-hi-en',
            'mr': 'Helsinki-NLP/opus-mt-mr-en'
        }
        
        if src_lang not in model_map:
            return text
            
        if src_lang not in self.pipelines:
            from transformers import pipeline
            print(f"Loading translation model for {src_lang}...")
            self.pipelines[src_lang] = pipeline("translation", model=model_map[src_lang])
            
        try:
            result = self.pipelines[src_lang](text)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Translation error: {e}")
            return text

SUMMARIZER = EmotionSummarizer()
TRANSLATOR = LanguageTranslator()

# Feedback Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
FEEDBACK_AUDIO_DIR = DATA_DIR / "feedback_audio"
FEEDBACK_CSV_PATH = DATA_DIR / "feedback_data.csv"

class FeedbackSubmission(BaseModel):
    analysis_id: str
    input_type: str
    text: Optional[str] = None
    predicted_emotion: str
    confidence: float
    user_correct: bool
    corrected_emotion: Optional[str] = None
    comment: Optional[str] = None

app = FastAPI(title="EmpathyCo Backend")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure robust absolute paths regardless of execution directory
BASE_DIR = Path(__file__).resolve().parent.parent

# --- MODEL LOADING LOGIC ---
SCALER_PATH = BASE_DIR / "models" / "scaler_advanced.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "data" / "processed" / "label_encoder_advanced.npy"

AUDIO_MODELS = {
    "en": BASE_DIR / "models" / "audio_model_simple.h5",
    "hi": BASE_DIR / "models" / "audio_model_simple.h5",
    "mr": BASE_DIR / "models" / "audio_model_simple.h5",
    "multi": BASE_DIR / "models" / "audio_model_simple.h5",
}

# Global model artifacts
SCALER = None
LABEL_ENCODER_CLASSES = None
MODELS = {}
MODEL_PATHS = {
    "en": BASE_DIR / "models" / "audio_model_simple.h5",
    "hi": BASE_DIR / "models" / "audio_model_simple.h5",
    "mr": BASE_DIR / "models" / "audio_model_simple.h5",
    "multi": BASE_DIR / "models" / "audio_model_simple.h5"  # Fallback
}
FEATURE_EXTRACTOR = EnhancedFeatureExtractor(sample_rate=16000)

# Global pipeline instances
TEXT_ANALYZER = None
ASR_PIPELINE = None
FUSION_ENGINE = MultimodalFusion(audio_weight=0.6, text_weight=0.4)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.on_event("startup")
async def load_artifacts():
    global SCALER, LABEL_ENCODER_CLASSES, MODELS, TEXT_ANALYZER, ASR_PIPELINE
    
    # Paths dynamically resolved via BASE_DIR
    scaler_path = BASE_DIR / "models" / "scaler_advanced.pkl"
    label_encoder_path = BASE_DIR / "data" / "processed" / "label_encoder_advanced.npy"
    
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
        print("Loaded scaler.")
    else:
        print(f"Warning: {scaler_path} not found!")
        
    if os.path.exists(label_encoder_path):
        LABEL_ENCODER_CLASSES = np.load(label_encoder_path, allow_pickle=True)
    else:
        print(f"Warning: {label_encoder_path} not found!")
        LABEL_ENCODER_CLASSES = np.array(["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])

    custom_objs = {
        'TimeShuffleAttention': TimeShuffleAttention,
        'LightweightConvTransformer': LightweightConvTransformer
    }

    # Ensure directories exist
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV if not exists
    if not FEEDBACK_CSV_PATH.exists():
        with open(FEEDBACK_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'analysis_id', 'input_type', 'text', 'audio_path', 
                             'predicted_emotion', 'confidence', 'user_correct', 
                             'corrected_emotion', 'comment', 'verified'])

    # Clean up old audio cache (simple approach: clear on startup, or just leave it for a cron)
    for f in AUDIO_CACHE_DIR.glob("*"):
        try:
            f.unlink()
        except:
            pass

    for lang, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                MODELS[lang] = tf.keras.models.load_model(path, custom_objects=custom_objs)
                print(f"Loaded {lang} model from {path}")
            except Exception as e:
                print(f"Failed to load {lang} model: {e}")
        else:
            print(f"Model path not found: {path} for language '{lang}'")

    print("Initializing Text Emotion Analyzer...")
    text_model_path = "../models/text_model_finetuned"
    if os.path.exists(text_model_path):
        print(f"Found fine-tuned text model at {text_model_path}, loading...")
        TEXT_ANALYZER = TextEmotionAnalyzer(model_id=text_model_path)
    else:
        TEXT_ANALYZER = TextEmotionAnalyzer(model_id="MilaNLProc/xlm-emo-t")

    print("Initializing ASR Pipeline (openai/whisper-small)...")
    try:
        from transformers import pipeline
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small"
        )
        print("ASR Pipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load ASR pipeline: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    return {"models": list(MODEL_PATHS.keys())}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # File size limit check
    content_length = audio.size if hasattr(audio, 'size') else 0
    if content_length > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max 50MB)")
        
    if not ASR_PIPELINE:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    try:
        temp_audio_path = f"temp_transcribe_{audio.filename}"
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Check size of saved file as fallback
        if os.path.getsize(temp_audio_path) > MAX_FILE_SIZE:
            os.remove(temp_audio_path)
            raise HTTPException(status_code=413, detail="File too large (Max 50MB)")

        result = ASR_PIPELINE(temp_audio_path)
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return {"text": result.get("text", "")}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/analyze")
async def analyze(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: str = Form("en")
):
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Must provide either audio or text.")

    # File size limit check
    if audio:
        content_length = audio.size if hasattr(audio, 'size') else 0
        if content_length > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (Max 50MB)")

    if language not in MODELS and language in MODEL_PATHS:
        # Fallback to English/Multi if the specific one failed to load
        print(f"Requested {language} but model not loaded. Falling back to multi.")
        if "multi" in MODELS:
            language = "multi"
        elif "en" in MODELS:
            language = "en"

    # Initialize results
    analysis_id = str(uuid.uuid4())
    response = {"analysis_id": analysis_id}
    context_text = text

    # --- AUDIO INFERENCE ---
    audio_probs = None
    if audio:
        try:
            cached_audio_path = AUDIO_CACHE_DIR / f"{analysis_id}.wav"
            with open(cached_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            
            # 1. Prediction Probabilities
            features = FEATURE_EXTRACTOR.extract(str(cached_audio_path), augment=False)
            features_scaled = SCALER.transform([features]) if SCALER else np.array([features])
            
            model_key = language if language in MODELS else "en"
            model = MODELS.get(model_key)
            if model:
                preds = model.predict(features_scaled)[0]
                audio_probs = [float(p) for p in preds]
                pred_idx = np.argmax(preds)
                response["audio"] = {
                    "emotion": str(LABEL_ENCODER_CLASSES[pred_idx]),
                    "confidence": float(preds[pred_idx]),
                    "probabilities": audio_probs
                }
            
            # 2. Transcription for Summary Context (if no text provided)
            if not context_text and ASR_PIPELINE:
                asr_res = ASR_PIPELINE(str(cached_audio_path))
                context_text = asr_res.get("text", "")
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            # We don't delete immediately; it will be cleaned up next startup

    # --- TEXT INFERENCE ---
    text_results = []
    if text:
        try:
            is_short = len(text.split()) <= 3
            text_results = TEXT_ANALYZER.analyze(text)
            
            if text_results:
                # If it's a lexicon hit, the analyzer will have printed it, 
                # but we can add a response-level log if needed.
                if is_short and text_results[0]['score'] == 0.90:
                    print(f"[API] Lexicon override applied for text: '{text}' -> {text_results[0]['emotion']}")
                
                response["text"] = {
                    "emotion": text_results[0]['emotion'],
                    "confidence": text_results[0]['score'],
                    "top_3": text_results[:3]
                }
        except Exception as e:
            print(f"Text analysis error: {e}")

    # --- MULTIMODAL FUSION ---
    final_emotion = "neutral"
    final_confidence = 0.5
    
    if audio and text and audio_probs:
        final_emotion, final_confidence = FUSION_ENGINE.fuse(audio_probs, text_results, LABEL_ENCODER_CLASSES)
    elif audio and "audio" in response:
        final_emotion = response["audio"]["emotion"]
        final_confidence = response["audio"]["confidence"]
    elif text and "text" in response:
        final_emotion = response["text"]["emotion"]
        final_confidence = response["text"]["confidence"]

    response["final_emotion"] = final_emotion
    response["final_confidence"] = final_confidence

    # --- TRANSLATION & SUMMARY ---
    summary = "No content available for summary."
    if context_text:
        try:
            # 1. Detect Language
            det_lang = detect(context_text)
            response["detected_language"] = det_lang
            
            # 2. Translate if necessary
            proc_text = TRANSLATOR.translate(context_text, det_lang)
            
            # 3. Generate Summary
            summary = SUMMARIZER.generate(proc_text, final_emotion, final_confidence)
        except Exception as e:
            print(f"Summary generation error: {e}")
            summary = f"Summary unavailable: {str(e)}"

    response["summary"] = summary

    return response

@app.post("/feedback")
async def handle_feedback(feedback: FeedbackSubmission, background_tasks: BackgroundTasks):
    timestamp = datetime.datetime.now().isoformat()
    verified = False
    feedback_audio_path = ""
    
    # Check if this involves audio and move it to permanent storage
    if feedback.input_type in ["audio", "both"]:
        cached_audio = AUDIO_CACHE_DIR / f"{feedback.analysis_id}.wav"
        if cached_audio.exists():
            perm_audio = FEEDBACK_AUDIO_DIR / f"{feedback.analysis_id}.wav"
            shutil.move(str(cached_audio), str(perm_audio))
            feedback_audio_path = str(perm_audio)
    
    # Save to CSV
    try:
        with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                feedback.analysis_id,
                feedback.input_type,
                feedback.text or "",
                feedback_audio_path,
                feedback.predicted_emotion,
                feedback.confidence,
                "1" if feedback.user_correct else "0",
                feedback.corrected_emotion or "",
                feedback.comment or "",
                "1" if verified else "0"
            ])
    except Exception as e:
        print(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return {"status": "success", "message": "Feedback recorded."}

@app.post("/verify_feedback")
async def verify_feedback(analysis_id: str = Form(...)):
    """Simple endpoint to manually verify a feedback entry by ID."""
    rows = []
    found = False
    if not FEEDBACK_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="No feedback data found.")
        
    try:
        with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            for row in reader:
                if row[1] == analysis_id:
                    row[10] = "1"  # Set verified to "1"
                    found = True
                rows.append(row)
                
        if found:
            with open(FEEDBACK_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return {"status": "success", "message": f"Feedback {analysis_id} verified."}
        else:
            raise HTTPException(status_code=404, detail="Feedback ID not found.")
    except Exception as e:
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_models")
async def reload_models():
    """Triggers the load_artifacts function to load new finetuned models without a restart."""
    await load_artifacts()
    return {"status": "success", "message": "Models reloaded."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
