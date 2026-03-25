from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import joblib

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import EnhancedFeatureExtractor
from src.model_trainer import TimeShuffleAttention, LightweightConvTransformer
from src.text_analyzer import TextEmotionAnalyzer
from src.fusion import MultimodalFusion

app = FastAPI(title="EmpathyCo Backend")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model artifacts
SCALER = None
LABEL_ENCODER_CLASSES = None
MODELS = {}
MODEL_PATHS = {
    "en": "../models/audio_model_simple.h5",
    "hi": "../models/audio_model_simple.h5",
    "mr": "../models/audio_model_simple.h5",
    "multi": "../models/audio_model_simple.h5"  # Fallback
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
    
    # Paths relative to backend/ directory
    scaler_path = "../models/scaler_advanced.pkl"
    label_encoder_path = "../data/processed/label_encoder_advanced.npy"
    
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
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

    response = {}
    
    # --- AUDIO INFERENCE ---
    audio_probs = None
    if audio:
        try:
            temp_audio_path = f"temp_analyze_{audio.filename}"
            with open(temp_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            
            if os.path.getsize(temp_audio_path) > MAX_FILE_SIZE:
                 os.remove(temp_audio_path)
                 raise HTTPException(status_code=413, detail="File too large (Max 50MB)")

            features = FEATURE_EXTRACTOR.extract(temp_audio_path, augment=False)
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            if SCALER:
                features_scaled = SCALER.transform([features])
            else:
                features_scaled = np.array([features])
                
            model = MODELS.get(language, MODELS.get("en"))
            if not model:
                 raise Exception(f"No model available for language {language}")
                 
            preds = model.predict(features_scaled)[0]
            audio_probs = [float(p) for p in preds]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx])
            
            emotion_label = str(LABEL_ENCODER_CLASSES[pred_idx])
            
            response["audio"] = {
                "emotion": emotion_label,
                "confidence": confidence,
                "probabilities": audio_probs,
                "model_used": MODEL_PATHS.get(language, "Unknown")
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"Audio inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Audio inference failed: {str(e)}")
            
    # --- TEXT INFERENCE ---
    text_results = []
    if text:
        try:
            text_results = TEXT_ANALYZER.analyze(text)
            if text_results:
                response["text"] = {
                    "emotion": text_results[0]['emotion'],
                    "confidence": text_results[0]['score'],
                    "top_3": text_results[:3]
                }
            else:
                response["text"] = {
                    "emotion": "neutral",
                    "confidence": 0.5,
                    "top_3": [{"emotion": "neutral", "score": 0.5}]
                }
        except Exception as e:
            print(f"Text inference error: {e}")
            response["text"] = {"emotion": "neutral", "confidence": 0.5}
        
    # --- FUSION LOGIC ---
    if audio and text:
        final_emotion, final_conf = FUSION_ENGINE.fuse(audio_probs, text_results, LABEL_ENCODER_CLASSES)
        response["final_emotion"] = final_emotion
        response["final_confidence"] = final_conf
    elif audio:
        response["final_emotion"] = response["audio"]["emotion"]
        response["final_confidence"] = response["audio"]["confidence"]
    else:
        response["final_emotion"] = response["text"]["emotion"]
        response["final_confidence"] = response["text"]["confidence"]

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
