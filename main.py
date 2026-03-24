from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os
import shutil
import numpy as np
import tensorflow as tf
import joblib

# Import custom layers and extractor from local module
from train_audio_advanced import (
    EnhancedFeatureExtractor,
    TimeShuffleAttention,
    LightweightConvTransformer
)

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
    "en": "models/audio_model_simple.h5",
    "hi": "models/audio_model_simple.h5",
    "mr": "models/audio_model_simple.h5",
    "multi": "models/audio_model_simple.h5"  # Fallback
}
FEATURE_EXTRACTOR = EnhancedFeatureExtractor(sample_rate=16000)

# Implement HuggingFace Text Classification via Transformers
from transformers import pipeline

# Global pipeline instance (loaded lazily or on startup)
TEXT_CLASSIFIER = None

# We can reuse the load_artifacts event to pre-fetch this pipeline or load it safely locally.
@app.on_event("startup")
async def load_artifacts():
    global SCALER, LABEL_ENCODER_CLASSES, MODELS, TEXT_CLASSIFIER
    
    # ... previous code ...
    if os.path.exists("models/scaler_advanced.pkl"):
        SCALER = joblib.load("models/scaler_advanced.pkl")
    else:
        print("Warning: scaler_advanced.pkl not found!")
        
    if os.path.exists("data/processed/label_encoder_advanced.npy"):
        LABEL_ENCODER_CLASSES = np.load("data/processed/label_encoder_advanced.npy", allow_pickle=True)
    else:
        print("Warning: label_encoder_advanced.npy not found!")
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

    print("Initializing HuggingFace Text Pipeline...")
    try:
        # j-hartmann model is freely available (no auth needed) and outputs:
        # anger, disgust, fear, joy, neutral, sadness, surprise
        TEXT_CLASSIFIER = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=3
        )
        print("Text Pipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load HuggingFace pipeline: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    return {"models": list(MODEL_PATHS.keys())}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Placeholder transcription
    return {"text": "This text is auto-transcribed from audio. (Real transcription engine not integrated to save memory)."}

@app.post("/analyze")
async def analyze(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: str = Form("en")
):
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Must provide either audio or text.")

    if language not in MODELS and language in MODEL_PATHS:
        # Fallback to English/Multi if the specific one failed to load
        print(f"Requested {language} but model not loaded. Falling back to multi.")
        if "multi" in MODELS:
            language = "multi"
        elif "en" in MODELS:
            language = "en"

    response = {}
    
    # --- AUDIO INFERENCE ---
    if audio:
        try:
            temp_audio_path = f"temp_{audio.filename}"
            with open(temp_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
                
            features = FEATURE_EXTRACTOR.extract(temp_audio_path, augment=False)
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            if SCALER:
                features_scaled = SCALER.transform([features])
            else:
                features_scaled = np.array([features])
                
            model = MODELS[language]
            preds = model.predict(features_scaled)[0]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx])
            
            emotion_label = str(LABEL_ENCODER_CLASSES[pred_idx])
            
            response["audio"] = {
                "emotion": emotion_label,
                "confidence": confidence,
                "probabilities": [float(p) for p in preds],
                "model_used": MODEL_PATHS[language]
            }
        except Exception as e:
            print(f"Audio inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Audio inference failed: {str(e)}")
            
    # --- TEXT INFERENCE ---
    if text:
        try:
            if TEXT_CLASSIFIER:
                results = TEXT_CLASSIFIER(text)[0] # returns list of dicts [{'label': 'joy', 'score': 0.99}, ...]
                
                # Standardize output naming
                top_emotion = results[0]['label']
                # j-hartmann outputs: anger, disgust, fear, joy, neutral, sadness, surprise
                mapping = {
                    "joy": "happy", "sadness": "sad", "anger": "angry",
                    "fear": "fearful", "surprise": "surprised",
                    "disgust": "disgust", "neutral": "neutral"
                }
                mapped_emotion = mapping.get(top_emotion, top_emotion)
                
                response["text"] = {
                    "emotion": mapped_emotion,
                    "confidence": float(results[0]['score']),
                    "top_3": [{"emotion": mapping.get(r['label'], r['label']), "score": float(r['score'])} for r in results]
                }
            else:
                raise Exception("TEXT_CLASSIFIER is not loaded.")
        except Exception as e:
            print(f"Text inference error: {e}")
            response["text"] = {
                "emotion": "neutral",
                "confidence": 0.5,
                "top_3": [{"emotion": "neutral", "score": 0.5}]
            }
        
    # --- FUSION LOGIC ---
    if audio and text:
        audio_conf = response["audio"]["confidence"]
        text_conf = response["text"]["confidence"]
        
        if audio_conf > 0.6:
            response["final_emotion"] = response["audio"]["emotion"]
            response["final_confidence"] = audio_conf
        else:
            response["final_emotion"] = response["text"]["emotion"]
            response["final_confidence"] = text_conf
    elif audio:
        response["final_emotion"] = response["audio"]["emotion"]
        response["final_confidence"] = response["audio"]["confidence"]
    else:
        response["final_emotion"] = response["text"]["emotion"]
        response["final_confidence"] = response["text"]["confidence"]

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
