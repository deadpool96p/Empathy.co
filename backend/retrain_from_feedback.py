import os
import sys
import csv
import shutil
import numpy as np
import tensorflow as tf
import torch
import joblib
from pathlib import Path

# Fix relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import EnhancedFeatureExtractor
from src.model_trainer import TimeShuffleAttention, LightweightConvTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEEDBACK_CSV_PATH = DATA_DIR / "feedback_data.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def load_verified_feedback():
    if not FEEDBACK_CSV_PATH.exists():
        return [], []
        
    text_data = []
    audio_data = []
    
    with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header: return [], []
        
        for row in reader:
            # columns: timestamp, analysis_id, input_type, text, audio_path, 
            # predicted_emotion, confidence, user_correct, corrected_emotion, comment, verified
            if len(row) < 11:
                continue
            
            verified = row[10] == "1"
            if not verified:
                continue
            
            user_correct = row[7] == "1"
            if user_correct:
                continue # We only retrain on corrections in this basic implementation
                
            input_type = row[2]
            text = row[3]
            audio_path = row[4]
            corrected_emotion = row[8]
            
            if not corrected_emotion:
                continue
            
            if input_type in ["text", "both"] and text.strip():
                text_data.append({"text": text, "label": corrected_emotion})
                
            if input_type in ["audio", "both"] and audio_path and os.path.exists(audio_path):
                audio_data.append({"audio_path": audio_path, "label": corrected_emotion})
                
    return text_data, audio_data

def retrain_text_model(text_data):
    if not text_data:
        print("No verified text feedback for retraining.")
        return

    print(f"Starting text model retraining with {len(text_data)} samples...")
    
    # Model target path
    finetuned_path = MODELS_DIR / "text_model_finetuned"
    base_model_id = str(finetuned_path) if finetuned_path.exists() else "MilaNLProc/xlm-emo-t"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_id)
    
    # Convert string labels to integer IDs mapped by the model
    # xlm-emo-t labels: anger, anticipation, disgust, fear, joy, sadness, surprise, trust (but our system maps it to our 8)
    # The pipeline returns joy, sadness, anger, etc. We must make sure we map correctly.
    # To keep it simple, we use the model's existing label2id
    label2id = model.config.label2id
    
    # Standardize our labels back to model's expectations if needed
    inverse_map = {"happy": "joy", "sad": "sadness", "angry": "anger"}
    
    valid_data = []
    for item in text_data:
        label = item['label'].lower()
        model_label = inverse_map.get(label, label)
        if model_label in label2id:
            valid_data.append({"text": item['text'], "label": label2id[model_label]})
    
    if not valid_data:
        print("No valid mapped labels found for text retraining.")
        return

    dataset = Dataset.from_list(valid_data)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3, # fine-tune multiple epochs for small datasets
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    
    # Save the updated model
    model.save_pretrained(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    print(f"Text model fine-tuned and saved to {finetuned_path}")

def retrain_audio_model(audio_data):
    if not audio_data:
        print("No verified audio feedback for retraining.")
        return
        
    print(f"Starting audio model retraining with {len(audio_data)} samples...")
    
    scaler_path = MODELS_DIR / "scaler_advanced.pkl"
    label_encoder_path = DATA_DIR / "processed/label_encoder_advanced.npy"
    model_path = MODELS_DIR / "audio_model_simple.h5"
    
    if not all([scaler_path.exists(), label_encoder_path.exists(), model_path.exists()]):
        print("Missing required audio model artifacts (scaler, encoder, or h5 model).")
        return
        
    scaler = joblib.load(scaler_path)
    label_classes = np.load(label_encoder_path, allow_pickle=True)
    extractor = EnhancedFeatureExtractor(sample_rate=16000)
    
    custom_objs = {
        'TimeShuffleAttention': TimeShuffleAttention,
        'LightweightConvTransformer': LightweightConvTransformer
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
    
    # Extract features
    X = []
    y = []
    for item in audio_data:
        try:
            features = extractor.extract(item['audio_path'], augment=False)
            X.append(features)
            
            # Find label index
            idx = np.where(label_classes == item['label'].lower())[0]
            if len(idx) > 0:
                y.append(idx[0])
            else:
                y.append(0) # fallback
        except Exception as e:
            print(f"Error processing {item['audio_path']}: {e}")
            
    if not X:
        print("No valid audio features extracted.")
        return
        
    X_scaled = scaler.transform(X)
    
    # Convert y to categorical based on the number of classes
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(label_classes))
    
    # Backup old model
    backup_path = str(model_path) + ".bak"
    shutil.copy2(model_path, backup_path)
    print(f"Backed up old model to {backup_path}")
    
    # Fine-tune the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower LR for fine-tuning
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    model.fit(np.array(X_scaled), np.array(y_cat), epochs=5, batch_size=2, verbose=1)
    
    # Save model
    model.save(model_path)
    print(f"Audio model fine-tuned and saved to {model_path}")

def mark_feedback_as_used():
    """Marks verified feedback as 'used_in_training' by adding a 12th column or resetting verified bit."""
    # A simple approach for this basic system is to reset the verified flag to "2" indicating used.
    if not FEEDBACK_CSV_PATH.exists():
        return
        
    rows = []
    with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 11 and row[10] == "1":
                row[10] = "2" # 2 = Verified and Used
            rows.append(row)
            
    with open(FEEDBACK_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Marked feedback entries as used in training.")

if __name__ == "__main__":
    t_data, a_data = load_verified_feedback()
    
    if not t_data and not a_data:
        print("No unwielded verified feedback found. Exiting.")
        sys.exit(0)
        
    if t_data:
        retrain_text_model(t_data)
        
    if a_data:
        retrain_audio_model(a_data)
        
    mark_feedback_as_used()
    print("Auto-retraining cycle complete.")
