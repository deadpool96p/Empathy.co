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

# Global pipeline instances
TEXT_CLASSIFIER = None
ASR_PIPELINE = None

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.on_event("startup")
async def load_artifacts():
    global SCALER, LABEL_ENCODER_CLASSES, MODELS, TEXT_CLASSIFIER, ASR_PIPELINE
    
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

    print("Initializing Multilingual Emotion Pipeline (cardiffnlp/twitter-xlm-roberta-base-emotion)...")
    try:
        # Multilingual model for Hindi/Marathi/English
        TEXT_CLASSIFIER = pipeline(
            "text-classification",
            model="MilaNLProc/xlm-emo-t",
            top_k=None  # Get all scores for better fusion
        )
        print("Text Pipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load HuggingFace pipeline: {e}")

    print("Initializing ASR Pipeline (openai/whisper-small)...")
    try:
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
    text_emotion_results = []
    if text:
        print(f"Analyzing text: {text}") # Debug print
        try:
            if TEXT_CLASSIFIER:
                results = TEXT_CLASSIFIER(text)[0] # returns list of dicts [{'label': 'joy', 'score': 0.99}, ...]
                print(f"Raw text model output: {results}") # Debug print
                
                # Standardize output naming
                # cardiffnlp/twitter-xlm-roberta-base-emotion outputs: anger, joy, optimism, sadness
                mapping = {
                    "joy": "happy", 
                    "sadness": "sad", 
                    "anger": "angry",
                    "optimism": "happy"
                }
                
                # Convert to our format
                text_emotion_results = []
                for res in results:
                    mapped_label = mapping.get(res['label'], res['label'])
                    text_emotion_results.append({
                        "emotion": mapped_label,
                        "score": float(res['score'])
                    })
                
                # Sort by score descending
                text_emotion_results.sort(key=lambda x: x['score'], reverse=True)
                
                response["text"] = {
                    "emotion": text_emotion_results[0]['emotion'],
                    "confidence": text_emotion_results[0]['score'],
                    "top_3": text_emotion_results[:3]
                }
            else:
                raise Exception("TEXT_CLASSIFIER is not loaded.")
        except Exception as e:
            import traceback
            print(f"Text inference error: {e}")
            traceback.print_exc()
            response["text"] = {
                "emotion": "neutral",
                "confidence": 0.5,
                "top_3": [{"emotion": "neutral", "score": 0.5}]
            }
        
    # --- FUSION LOGIC ---
    if audio and text:
        # Simple weighted fusion
        fused_probs = np.array(audio_probs)
        text_weight = 0.4
        audio_weight = 0.6
        
        # Mapping table for text emotions to audio indices
        text_to_audio_idx = {
            "happy": 2,
            "sad": 3,
            "angry": 4
        }
        
        for tr in text_emotion_results:
            idx = text_to_audio_idx.get(tr['emotion'])
            if idx is not None:
                # Weighted combine: final_prob = audio_p * 0.6 + text_p * 0.4
                fused_probs[idx] = (fused_probs[idx] * audio_weight) + (tr['score'] * text_weight)
        
        # Re-normalize
        fused_probs = fused_probs / np.sum(fused_probs)
        
        final_idx = np.argmax(fused_probs)
        response["final_emotion"] = str(LABEL_ENCODER_CLASSES[final_idx])
        response["final_confidence"] = float(fused_probs[final_idx])
        
    elif audio:
        response["final_emotion"] = response["audio"]["emotion"]
        response["final_confidence"] = response["audio"]["confidence"]
    else:
        response["final_emotion"] = response["text"]["emotion"]
        response["final_confidence"] = response["text"]["confidence"]

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
