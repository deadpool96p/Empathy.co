# backend/train_audio_advanced.py
import numpy as np
import tensorflow as tf
import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import audiomentations as A

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import UnifiedEmotionMapper, EnhancedFeatureExtractor
from src.model_trainer import create_advanced_model, create_simple_model

# ============================================
# 2. DATASET COLLECTORS (unchanged)
# ============================================
def collect_ravdess(base_path, mapper):
    paths, labels = [], []
    base = Path(base_path)
    for file in base.rglob("*.wav"):
        parts = file.stem.split('-')
        if len(parts) >= 3:
            emotion_code = int(parts[2])
            label = mapper.ravdess_map(emotion_code)
            if label is not None:
                paths.append(str(file))
                labels.append(label)
    return paths, labels

def collect_tess(base_path, mapper):
    paths, labels = [], []
    base = Path(base_path)
    for emotion_folder in base.iterdir():
        if emotion_folder.is_dir():
            label = mapper.tess_map(emotion_folder.name)
            if label is not None:
                for file in emotion_folder.glob("*.wav"):
                    paths.append(str(file))
                    labels.append(label)
    return paths, labels

def collect_cremad(base_path, mapper):
    paths, labels = [], []
    audio_dir = Path(base_path) / "AudioWAV"
    if not audio_dir.exists():
        return paths, labels
    for file in audio_dir.glob("*.wav"):
        parts = file.stem.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            label = mapper.cremad_map(emotion_code)
            if label is not None:
                paths.append(str(file))
                labels.append(label)
    return paths, labels

def collect_iemocap(base_path, mapper):
    paths, labels = [], []
    base = Path(base_path)
    eval_files = list(base.rglob("dialog/EmoEvaluation/*.txt"))
    for eval_file in eval_files:
        session_dir = eval_file.parent.parent.parent
        wav_dir = session_dir / "dialog" / "wav"
        with open(eval_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            utt_id = parts[1]
            emotion = parts[2].lower()
            label = mapper.iemocap_map(emotion)
            if label is None:
                continue
            wav_file = wav_dir / f"{utt_id}.wav"
            if not wav_file.exists():
                wav_file = wav_dir / f"{utt_id.replace('_', '')}.wav"
                if not wav_file.exists():
                    continue
            paths.append(str(wav_file))
            labels.append(label)
    return paths, labels

def collect_emodb(base_path, mapper):
    paths, labels = [], []
    wav_dir = Path(base_path) / "wav"
    if not wav_dir.exists():
        return paths, labels
    for file in wav_dir.glob("*.wav"):
        label = mapper.emodb_map(file.stem)
        if label is not None:
            paths.append(str(file))
            labels.append(label)
    return paths, labels

def collect_meld(base_path, mapper):
    paths, labels = [], []
    base = Path(base_path)
    csv_files = {
        'train': base / 'train_sent_emo.csv',
        'dev': base / 'dev_sent_emo.csv',
        'test': base / 'test_sent_emo.csv',
    }
    for split_name, csv_path in csv_files.items():
        if csv_path is None or not csv_path.exists():
            continue
        audio_dir = base / split_name
        if not audio_dir.exists():
            continue
        df = pd.read_csv(csv_path)
        required_cols = ['Emotion', 'Dialogue_ID', 'Utterance_ID']
        if not all(col in df.columns for col in required_cols):
            continue
        for idx, row in df.iterrows():
            emotion = row['Emotion']
            label = mapper.meld_map(emotion)
            if label is None:
                continue
            dia = row['Dialogue_ID']
            utt = row['Utterance_ID']
            possible_filenames = [
                f"dia{dia}_utt{utt}.wav",
                f"{dia}_{utt}.wav",
                f"dia{dia}_{utt}.wav",
                f"{dia}_utt{utt}.wav",
            ]
            wav_file = None
            for fname in possible_filenames:
                candidate = audio_dir / fname
                if candidate.exists():
                    wav_file = candidate
                    break
            if wav_file is None:
                continue
            paths.append(str(wav_file))
            labels.append(label)
    return paths, labels

def collect_common_voice_from_csv(csv_path, mapper):
    paths, labels = [], []
    if not os.path.exists(csv_path):
        return paths, labels
    try:
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            audio_path = row['path']
            label = int(row['label'])
            if os.path.exists(audio_path):
                paths.append(audio_path)
                labels.append(label)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
    return paths, labels

# ============================================
# 3. ADVANCED DATA AUGMENTATION
# ============================================
class AdvancedAudioAugmenter:
    def __init__(self, sample_rate=16000, intensity='mild'):
        self.sample_rate = sample_rate
        if intensity == 'mild':
            self.augment = A.Compose([
                A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
                A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
                A.PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
                A.Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.2),
            ])
        elif intensity == 'moderate':
            self.augment = A.Compose([
                A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                A.PitchShift(min_semitones=-4, max_semitones=-4, p=0.3),
                A.Shift(min_shift=-0.5, max_shift=0.5, p=0.3),
                A.Gain(min_gain_in_db=-12.0, max_gain_in_db=12.0, p=0.3),
            ])
        else:
            self.augment = None

    def apply(self, audio):
        if self.augment is None:
            return audio
        return self.augment(samples=audio, sample_rate=self.sample_rate)

# ============================================
# 6. TRAINING ANALYSIS FUNCTION
# ============================================
def analyze_training(history, config, model_type, final_val_acc):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    final_train_acc = train_acc[-1]
    gap = final_train_acc - final_val_acc
    print("\nTRAINING ANALYSIS")
    print(f"Model type: {model_type}")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Train-val gap: {gap:.4f}")

# ============================================
# 7. MAIN TRAINING SCRIPT
# ============================================
def train_advanced_model_script(use_simple_model=False, tuning_config=None, languages=None, fine_tune=False):
    config = {
        'datasets': {
            'ravdess': '../data/raw/ravdess',
            'tess': '../data/raw/tess',
            'cremad': '../data/raw/cremad',
            'iemocap': '../data/raw/iemocap',
            'emodb': '../data/raw/emodb',
            'meld': '../data/raw/meld',
        },
        'use_augmentation': True,
        'augment_factor': 1,
        'augmentation_intensity': 'mild',
        'batch_size': 32,
        'epochs': 200,
        'early_stopping_patience': 25,
        'scaler_save_path': '../models/scaler_advanced.pkl',
        'label_encoder_path': '../data/processed/label_encoder_advanced.npy',
        'raw_features_path': '../data/processed/X_raw.npy',
        'raw_labels_path': '../data/processed/y_raw.npy',
    }

    os.makedirs('../models', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)

    if os.path.exists(config['raw_features_path']) and os.path.exists(config['raw_labels_path']):
        X = np.load(config['raw_features_path'])
        y = np.load(config['raw_labels_path'])
    else:
        mapper = UnifiedEmotionMapper()
        extractor = EnhancedFeatureExtractor()
        augmenter = AdvancedAudioAugmenter(intensity=config['augmentation_intensity'])
        
        all_paths, all_labels = [], []
        collectors = {
            'ravdess': collect_ravdess, 'tess': collect_tess, 'cremad': collect_cremad,
            'iemocap': collect_iemocap, 'emodb': collect_emodb, 'meld': collect_meld
        }
        for name, path in config['datasets'].items():
            if os.path.exists(path):
                paths, labels = collectors[name](path, mapper)
                all_paths.extend(paths); all_labels.extend(labels)

        X_list, y_list = [], []
        for path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths), desc="Extracting"):
            feat = extractor.extract(path)
            X_list.append(feat); y_list.append(label)
            if config['use_augmentation']:
                feat_aug = extractor.extract(path, augment=True, augmenter=augmenter)
                X_list.append(feat_aug); y_list.append(label)
        X, y = np.array(X_list), np.array(y_list)
        np.save(config['raw_features_path'], X); np.save(config['raw_labels_path'], y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    np.save(config['label_encoder_path'], le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, config['scaler_save_path'])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    total_steps = (len(X_train) // config['batch_size']) * config['epochs']
    
    if use_simple_model:
        model = create_simple_model(input_dim=X.shape[1], num_emotions=len(le.classes_))
        m_name = "simple"
    else:
        model = create_advanced_model(input_dim=X.shape[1], num_emotions=len(le.classes_), total_steps=total_steps)
        m_name = "advanced"

    early_stop = tf.keras.callbacks.EarlyStopping(patience=config['early_stopping_patience'], restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), 
                        epochs=config['epochs'], batch_size=config['batch_size'], 
                        class_weight=class_weight_dict, callbacks=[early_stop])

    save_path = f'../models/audio_model_{m_name}.h5'
    model.save(save_path)
    analyze_training(history, config, m_name, history.history['val_accuracy'][-1])
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_advanced_model_script(use_simple_model=True)


# ============================================
# 6. TRAINING ANALYSIS FUNCTION (enhanced)
# ============================================
def analyze_training(history, config, model_type, final_val_acc):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    final_train_acc = train_acc[-1]
    gap = final_train_acc - final_val_acc

    print("\n" + "="*60)
    print("TRAINING ANALYSIS")
    print("="*60)
    print(f"Model type: {model_type}")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Train-val gap: {gap:.4f}")

    if gap > 0.05:
        print("⚠️  Possible overfitting. Suggestions:")
        print("   - Increase dropout (currently {})".format(config.get('dropout_rates', [0.4,0.3])))
        print("   - Add L2 regularization")
        print("   - Reduce number of transformer blocks (currently {})".format(config.get('num_blocks',1)))
        print("   - Increase data augmentation intensity")
    elif gap < -0.02:
        print("⚠️  Validation > training. Check for data leakage.")
    else:
        print("✅ Good generalization (gap within 5%).")

    # Plateau detection
    recent_val = val_acc[-10:]
    if len(recent_val) >= 5:
        if max(recent_val) - min(recent_val) < 0.01:
            print("ℹ️  Validation accuracy plateaued. Consider:")
            print("   - Reducing learning rate further")
            print("   - Using cosine decay restart")
            print("   - Early stopping already handled this")

    # Loss trend
    recent_loss = val_loss[-5:]
    if len(recent_loss) >= 2 and recent_loss[-1] < recent_loss[0]:
        print("ℹ️  Validation loss still decreasing – more epochs might help.")

    # Accuracy level
    if final_val_acc < 0.5:
        print("❌ Low accuracy. Model may be underfitting. Suggestions:")
        print("   - Increase model capacity (more blocks/dense units)")
        print("   - Check data quality")
        print("   - Adjust learning rate")
    elif final_val_acc < 0.7:
        print("📈 Moderate accuracy. Could improve with:")
        if not config.get('use_augmentation', False):
            print("   - Enable data augmentation")
        if config.get('num_blocks', 1) < 2:
            print("   - Try adding a second transformer block")
    else:
        print("🎉 Good accuracy. Model is performing well.")

    if not config.get('use_augmentation', False):
        print("💡 Data augmentation was not used – consider enabling it.")

    print("="*60)


# ============================================
# 7. MAIN TRAINING SCRIPT with tunable config
# ============================================
def train_advanced_model(use_simple_model=False, tuning_config=None, languages=None, fine_tune=False, fine_tune_model_path='models/audio_model_advanced_english.h5'):
    # Default config for advanced model
    default_advanced_config = {
        'num_blocks': 3,                    # try 2 blocks now
        'dense_units': [128, 64],            # unchanged
        'dropout_rates': [0.4, 0.3],
        'use_attention': True,
        'drop_path': 0.15,                    # stochastic depth
        'use_augmentation': True,
        'augmentation_intensity': 'mild',     # start mild
        'augment_factor': 1,                  # one augmented copy per original
        'batch_size': 32,
        'epochs': 200,
        'early_stopping_patience': 25,
    }
    if tuning_config is not None:
        default_advanced_config.update(tuning_config)

    config = {
        'datasets': {
            'ravdess': 'data/raw/ravdess',
            'tess': 'data/raw/tess',
            'cremad': 'data/raw/cremad',
            'iemocap': 'data/raw/iemocap',
            'emodb': 'data/raw/emodb',
            'meld': 'data/raw/meld',
        },
        # Dynamically add language-specific CSV datasets based on 'languages' argument
        # Expected language codes: 'hi' for Hindi, 'mr' for Marathi
        # CSV files contain pseudo-labeled audio paths
        # Paths are relative to the project root
        # Collectors for these will be added later in the collectors dict
        # We'll extend the config later after parsing languages

        'use_augmentation': default_advanced_config['use_augmentation'],
        'augment_factor': default_advanced_config['augment_factor'],
        'augmentation_intensity': default_advanced_config['augmentation_intensity'],
        'batch_size': default_advanced_config['batch_size'],
        'epochs': default_advanced_config['epochs'],
        'early_stopping_patience': default_advanced_config['early_stopping_patience'],
        # Determine model save path based on selected languages and fine-tuning
        'model_save_path': None,  # placeholder, will be set after processing languages
        'scaler_save_path': 'models/scaler_advanced.pkl',
        'label_encoder_path': 'data/processed/label_encoder_advanced.npy',
        'raw_features_path': 'data/processed/X_raw.npy',
        'raw_labels_path': 'data/processed/y_raw.npy',
    }

    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Load or extract raw features
    if os.path.exists(config['raw_features_path']) and os.path.exists(config['raw_labels_path']):
        print("=" * 50)
        print("Loading cached raw features...")
        print("=" * 50)
        X = np.load(config['raw_features_path'])
        y = np.load(config['raw_labels_path'])
        print(f"Loaded {len(X)} samples.")
    else:
        mapper = UnifiedEmotionMapper()
        extractor = EnhancedFeatureExtractor()
        augmenter = AdvancedAudioAugmenter(intensity=config['augmentation_intensity']) if config['use_augmentation'] else None

        all_paths, all_labels = [], []
        collectors = {
            'ravdess': collect_ravdess,
            'tess': collect_tess,
            'cremad': collect_cremad,
            'iemocap': collect_iemocap,
            'emodb': collect_emodb,
            'meld': collect_meld,
            # CSV-based collectors for Common Voice Hindi and Marathi
            'cv_hi': lambda path, mapper: collect_common_voice_from_csv(path, mapper),
            'cv_mr': lambda path, mapper: collect_common_voice_from_csv(path, mapper),
        }

        print("=" * 50)
        print("Collecting dataset files...")
        print("=" * 50)
        for name, path in config['datasets'].items():
            if os.path.exists(path):
                print(f"Collecting {name}...")
                paths, labels = collectors[name](path, mapper)
                if paths:
                    all_paths.extend(paths)
                    all_labels.extend(labels)
                    print(f"  -> {len(paths)} files")
            else:
                print(f"Skipping {name} (path not found)")
        # Add language-specific CSV datasets if requested
        if languages:
            for lang in languages:
                csv_key = f"cv_{lang}"
                csv_path = f"data/processed/cv_{lang}_labels.csv"
                if os.path.exists(csv_path):
                    config['datasets'][csv_key] = csv_path
                    print(f"Added CSV dataset for language '{lang}' at {csv_path}")
                else:
                    print(f"Warning: CSV dataset for language '{lang}' not found at {csv_path}")

        if not all_paths:
            raise ValueError("No dataset files found. Check your paths.")

        print(f"\nTotal files collected: {len(all_paths)}")
        print(f"Class distribution: {np.bincount(all_labels)}")

        # Extract features
        print("\n" + "=" * 50)
        print("Extracting features...")
        print("=" * 50)
        X_list, y_list = [], []
        for i, (path, label) in enumerate(tqdm(zip(all_paths, all_labels), total=len(all_paths), desc="Extracting")):
            feat = extractor.extract(path, augment=False)
            X_list.append(feat)
            y_list.append(label)
            if config['use_augmentation']:
                for _ in range(config['augment_factor']):
                    feat_aug = extractor.extract(path, augment=True, augmenter=augmenter)
                    X_list.append(feat_aug)
                    y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\nTotal samples: {len(X)}")
        print(f"Class distribution: {np.bincount(y)}")

        # Diagnostic prints
        print("\n" + "=" * 50)
        print("Feature diagnostics:")
        print("=" * 50)
        print(f"Feature shape: {X.shape}")
        print(f"Feature dtype: {X.dtype}")
        print(f"Any NaN: {np.isnan(X).any()}")
        print(f"Any Inf: {np.isinf(X).any()}")
        print(f"Mean of first 10 features: {np.mean(X, axis=0)[:10]}")
        print(f"Std of first 10 features: {np.std(X, axis=0)[:10]}")
        print(f"Min value: {np.min(X):.3f}, Max value: {np.max(X):.3f}")

        # Save raw features
        np.save(config['raw_features_path'], X)
        np.save(config['raw_labels_path'], y)
        print("Raw features saved to disk.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    np.save(config['label_encoder_path'], le.classes_)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, config['scaler_save_path'])

    # Fine-tuning: Prepare model (actual loading happens after model creation)
    # This block just checks if the path exists. The actual logic is moved below.
    if fine_tune and not os.path.exists(fine_tune_model_path):
        print(f"⚠️ Fine-tune model path {fine_tune_model_path} not found. Fallback to fresh training.")
        fine_tune = False

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # Compute total training steps for LR schedule
    steps_per_epoch = len(X_train) // config['batch_size']
    total_steps = config['epochs'] * steps_per_epoch

    # Choose model
    if use_simple_model:
        print("\n" + "=" * 50)
        print("Using SIMPLE baseline model")
        print("=" * 50)
        model = create_simple_model(input_dim=X.shape[1], num_emotions=len(np.unique(y_encoded)))
        model_type = "Simple"
        # Simple model uses its own config; we pass empty dict for analysis
        analysis_config = config.copy()
    else:
        print("\n" + "=" * 50)
        print("Using ADVANCED CNN-Transformer model")
        print("=" * 50)
        print("Tuning parameters:", default_advanced_config)
        model = create_advanced_model(
            input_dim=X.shape[1],
            num_emotions=len(np.unique(y_encoded)),
            total_steps=total_steps,
            num_blocks=default_advanced_config['num_blocks'],
            dense_units=default_advanced_config['dense_units'],
            dropout_rates=default_advanced_config['dropout_rates'],
            use_attention=default_advanced_config['use_attention'],
            drop_path=default_advanced_config['drop_path']
        )
        model_type = "Advanced"
        analysis_config = {**config, **default_advanced_config}
        
        # Update model_save_path with language and fine-tune suffixes
        suffix_parts = []
        if languages:
            # Map 'en' to 'english' for filename convention
            display_langs = ['english' if l == 'en' else l for l in languages]
            suffix_parts.append('_'.join(display_langs))
        else:
            suffix_parts.append('english')
            
        if fine_tune:
            suffix_parts.append('fine_tuned')
            
        suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''
        config['model_save_path'] = f"models/audio_model_advanced{suffix}.h5"
        print(f"Model will be saved to {config['model_save_path']}")

    # Apply fine-tuning after model creation
    if fine_tune:
        print(f"Applying weights for fine-tuning from {fine_tune_model_path}...")
        model.load_weights(fine_tune_model_path)
        # Freeze early layers (transformer blocks and CNNs)
        # Typically freeze up to the transformer blocks
        for layer in model.layers:
            if 'transformer' in layer.name or 'conv1d' in layer.name:
                layer.trainable = False
        
        # Recompile with a very low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model specialized and re-compiled for fine-tuning.")

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=config['early_stopping_patience'], restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ModelCheckpoint(config['model_save_path'], save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    if use_simple_model:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_accuracy'))

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_test_scaled, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Run training analysis
    analyze_training(history, analysis_config, model_type, test_acc)

    # Save final model
    model.save(config['model_save_path'])
    print(f"Model saved to {config['model_save_path']}")

    return model, history


if __name__ == "__main__":
    # Disable mixed precision for stability
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Option 1: Use simple model
    # use_simple = True
    # model, history = train_advanced_model(use_simple_model=use_simple)

    # Option 2: Use advanced model with custom tuning
    my_tuning = {
        'num_blocks': 3,
        'dense_units': [256, 128],
        'dropout_rates': [0.5, 0.4],
        'use_attention': True,
        'drop_path': 0.15,
        'use_augmentation': True,
        'augmentation_intensity': 'moderate',
        'augment_factor': 2,
        'batch_size': 32,
        'epochs': 200,
        'early_stopping_patience': 25,
    }

    # Example for Hindi only training
    # model, history = train_advanced_model(languages=['hi'])
    
    # Example for Marathi only training
    # model, history = train_advanced_model(languages=['mr'])
    
    # Example for Multilingual training
    # model, history = train_advanced_model(languages=['en', 'hi', 'mr'])
    
    # Example for Fine-tuning English on Hindi
    # model, history = train_advanced_model(languages=['hi'], fine_tune=True, fine_tune_model_path='models/audio_model_advanced_english.h5')

    # Default: Run training with provided tuning config (English default if languages=None)
    model, history = train_advanced_model(use_simple_model=False, tuning_config=my_tuning)