# src/gui_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers
import librosa
import soundfile as sf
import sounddevice as sd
import os
import pygame
import tempfile
import time

# For Whisper ASR
import torch
from transformers import pipeline

from src.text_analyzer import TextEmotionAnalyzer

# ============================================
# Custom layer definitions (must match training)
# ============================================
class TimeShuffleAttention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
    def build(self, input_shape):
        _, time, channels = input_shape
        self.temporal_dense = layers.Dense(time // self.reduction, activation='relu')
        self.temporal_restore = layers.Dense(time, activation='sigmoid')
        self.channel_dense = layers.Dense(channels // self.reduction, activation='relu')
        self.channel_restore = layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)
    def call(self, inputs):
        temporal_att = tf.reduce_mean(inputs, axis=2, keepdims=True)
        temporal_att = tf.squeeze(temporal_att, axis=2)
        temporal_att = self.temporal_dense(temporal_att)
        temporal_att = self.temporal_restore(temporal_att)
        temporal_att = tf.expand_dims(temporal_att, axis=2)
        channel_att = tf.reduce_mean(inputs, axis=1, keepdims=True)
        channel_att = tf.squeeze(channel_att, axis=1)
        channel_att = self.channel_dense(channel_att)
        channel_att = self.channel_restore(channel_att)
        channel_att = tf.expand_dims(channel_att, axis=1)
        attended = inputs * temporal_att * channel_att
        return attended
    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config

class LightweightConvTransformer(layers.Layer):
    def __init__(self, dim, num_heads=4, expansion=4, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.expansion = expansion
        self.drop_path = drop_path
        self.depthwise_conv = layers.DepthwiseConv1D(kernel_size=3, padding='same')
        self.pointwise_conv1 = layers.Conv1D(dim * expansion, kernel_size=1)
        self.pointwise_conv2 = layers.Conv1D(dim, kernel_size=1)
        self.layer_norm1 = layers.LayerNormalization()
        self.multi_head_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.layer_norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(dim * expansion, activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(dim)
        ])
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training=None):
        shortcut = x
        x_conv = self.depthwise_conv(x)
        x_conv = self.pointwise_conv1(x_conv)
        x_conv = tf.nn.gelu(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv, training=training)

        if self.drop_path > 0 and training:
            keep_prob = 1.0 - self.drop_path
            batch_size = tf.shape(x)[0]
            random_tensor = keep_prob + tf.random.uniform((batch_size,))
            binary_mask = tf.floor(random_tensor)
            binary_mask = tf.reshape(binary_mask, (batch_size, 1, 1))
            x_conv = (x_conv * binary_mask) / keep_prob

        x_conv = shortcut + x_conv
        x_norm = self.layer_norm1(x_conv)
        x_att = self.multi_head_att(x_norm, x_norm)
        x_att = self.dropout(x_att, training=training)
        x_att = x_conv + x_att
        x_norm2 = self.layer_norm2(x_att)
        x_ffn = self.ffn(x_norm2)
        x_ffn = self.dropout(x_ffn, training=training)
        x_out = x_att + x_ffn
        return x_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "expansion": self.expansion,
            "drop_path": self.drop_path
        })
        return config


# ============================================
# Feature extractor (must match training)
# ============================================
class EnhancedFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self, audio_path):
        """Extract 63‑dimensional features from audio file."""
        y, sr_audio = librosa.load(audio_path, sr=self.sample_rate)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        features = []
        # MFCCs (13) + deltas
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr_audio, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        features.extend(mfcc_delta_mean)

        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr_audio)[0]
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))

        spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr_audio)[0]
        features.append(np.mean(spec_rolloff))

        spec_bw = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr_audio)[0]
        features.append(np.mean(spec_bw))

        # Pitch
        f0, _, _ = librosa.pyin(y_trimmed, fmin=50, fmax=500)
        f0 = f0[~np.isnan(f0)]
        features.append(np.mean(f0) if len(f0) > 0 else 0)
        features.append(np.std(f0) if len(f0) > 0 else 0)

        # Energy
        rms = librosa.feature.rms(y=y_trimmed)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr_audio)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)

        # Mel-spectrogram stats
        mel = librosa.feature.melspectrogram(y=y_trimmed, sr=sr_audio)
        features.append(np.mean(mel))
        features.append(np.std(mel))

        return np.array(features)


class EmotiDeskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmotiDesk - Multimodal Emotion Recognition")
        self.root.geometry("850x750")
        self.root.resizable(False, False)

        # Load models
        self.load_models()

        # Initialize variables
        self.audio_path = None
        self.is_temp_recording = False
        self.asr_pipeline = None  # cache for Whisper

        # Define mapping from text emotions to audio indices
        self.text_to_audio_map = {
            'neutral': 0, 'approval': 0, 'realization': 0, 'relief': 0,
            'confusion': 0, 'curiosity': 0,
            'calm': 1,
            'joy': 2, 'amusement': 2, 'excitement': 2, 'optimism': 2,
            'love': 2, 'admiration': 2, 'caring': 2, 'desire': 2,
            'gratitude': 2, 'pride': 2,
            'sadness': 3, 'disappointment': 3, 'grief': 3, 'embarrassment': 3,
            'remorse': 3,
            'anger': 4, 'annoyance': 4, 'disapproval': 4,
            'fear': 5, 'nervousness': 5,
            'disgust': 6,
            'surprise': 7,
        }

        # Map audio indices to human‑readable emotion names
        self.audio_emotion_names = {
            0: "neutral",
            1: "calm",
            2: "happy",
            3: "sad",
            4: "angry",
            5: "fearful",
            6: "disgust",
            7: "surprised"
        }

        # Build UI
        self.setup_ui()

    def load_models(self):
        """Load all pre‑trained models and helpers (advanced model)."""
        try:
            self.audio_extractor = EnhancedFeatureExtractor()
            custom_objects = {
                'TimeShuffleAttention': TimeShuffleAttention,
                'LightweightConvTransformer': LightweightConvTransformer
            }
            self.audio_model = tf.keras.models.load_model(
                'models/audio_model_advanced.h5',
                custom_objects=custom_objects
            )
            self.scaler = joblib.load('models/scaler_advanced.pkl')

            # Load label encoder
            le_array = np.load('data/processed/label_encoder_advanced.npy', allow_pickle=True)
            self.label_encoder = np.array([str(x) for x in le_array])

            self.text_analyzer = TextEmotionAnalyzer(model_name='go_emotions', use_gpu=True)
            print("All models loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Could not load models:\n{e}")
            self.root.destroy()
            return

    def setup_ui(self):
        """Create the user interface."""
        title = tk.Label(
            self.root,
            text="🎭 EmotiDesk",
            font=("Arial", 28, "bold"),
            fg="#333"
        )
        title.pack(pady=20)

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Audio selection
        audio_frame = ttk.LabelFrame(main_frame, text="1. Audio Input (optional)", padding="10")
        audio_frame.pack(fill=tk.X, pady=10)

        self.audio_label = tk.Label(audio_frame, text="No file selected", fg="gray")
        self.audio_label.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(audio_frame)
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="Browse", command=self.browse_audio).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Record", command=self.record_audio).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Play", command=self.play_audio).pack(side=tk.LEFT, padx=2)

        # Text input
        text_frame = ttk.LabelFrame(main_frame, text="2. Text / Transcript (optional)", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.text_input = tk.Text(text_frame, height=5, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            text_frame,
            text="Auto‑transcribe",
            command=self.transcribe_audio
        ).pack(pady=5)

        # Analyze button
        ttk.Button(
            main_frame,
            text="🎯 Analyze Emotion",
            command=self.analyze_emotion,
            style="Accent.TButton"
        ).pack(pady=20)

        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="3. Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_label = tk.Label(
            results_frame,
            text="Ready to analyze",
            font=("Arial", 14),
            wraplength=600,
            justify=tk.LEFT
        )
        self.result_label.pack(pady=20, padx=10)

        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')

    # ---------- Helper methods ----------
    def browse_audio(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac")]
        )
        if filename:
            self.audio_path = filename
            self.is_temp_recording = False
            self.audio_label.config(text=os.path.basename(filename), fg="black")

    def record_audio(self, duration=5, samplerate=16000):
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            messagebox.showerror("Missing Library", "Please install sounddevice: pip install sounddevice")
            return

        record_win = tk.Toplevel(self.root)
        record_win.title("Record Audio")
        record_win.geometry("300x150")
        record_win.transient(self.root)
        record_win.grab_set()

        status = tk.Label(record_win, text=f"Recording for {duration} seconds...")
        status.pack(pady=20)
        record_win.update()

        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        filename = "temp_recording.wav"
        sf.write(filename, recording, samplerate)

        self.audio_path = filename
        self.is_temp_recording = True
        self.audio_label.config(text="temp_recording.wav", fg="black")
        status.config(text="Recording complete!")
        record_win.after(1000, record_win.destroy)

    def play_audio(self):
        if not self.audio_path:
            messagebox.showwarning("No Audio", "Please select or record an audio file first.")
            return

        thread = threading.Thread(target=self._play_audio_thread)
        thread.daemon = True
        thread.start()

    def _play_audio_thread(self):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Playback Error", str(e))

    # ---------- NEW: Multilingual Transcription (Whisper) ----------
    def transcribe_audio(self):
        if not self.audio_path:
            messagebox.showwarning("No Audio", "Please select or record an audio file first.")
            return

        self.progress.pack(pady=10)
        self.progress.start()
        self.root.update()

        thread = threading.Thread(target=self._run_transcription)
        thread.daemon = True
        thread.start()

    def _run_transcription(self, language="multilingual"):
        temp_wav = None
        try:
            print("Starting transcription...")
            # Convert to WAV
            y, sr = librosa.load(self.audio_path, sr=16000)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name
            sf.write(temp_wav, y, sr)
            print(f"Converted audio to temporary WAV: {temp_wav}")

            # Select model based on language preference
            if self.asr_pipeline is None:
                print("Loading ASR model...")
                device = 0 if torch.cuda.is_available() else -1

                # You can change this to any model from the table above
                model_id = "openai/whisper-small"  # Good multilingual default
                # model_id = "collabora/whisper-base-hindi"  # For Hindi-only
                # model_id = "ai4bharat/indicwav2vec_v1_marathi"  # For Marathi

                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=device
                )
                print(f"ASR model {model_id} loaded.")

            # Transcribe
            print("Transcribing...")
            result = self.asr_pipeline(temp_wav)
            transcription = result['text']
            print(f"Transcription result: '{transcription}'")

            if transcription and transcription.strip():
                self.root.after(0, self._insert_transcription, transcription)
            else:
                self.root.after(0, messagebox.showwarning, "Transcription Warning",
                                "No speech detected in the audio.")

        except Exception as e:
            print(f"Transcription error: {e}")
            self.root.after(0, messagebox.showerror, "Transcription Error", str(e))
        finally:
            # Clean up
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                    print("Temporary file deleted.")
                except Exception as e:
                    print(f"Could not delete temp file: {e}")
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)

    def _insert_transcription(self, text):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", text)
    # ----------------------------------------------------------------

    # ---------- Main analysis ----------
    def analyze_emotion(self):
        if not self.audio_path and not self.text_input.get("1.0", tk.END).strip():
            messagebox.showwarning("No Input", "Please provide an audio file, some text, or both.")
            return

        self.progress.pack(pady=10)
        self.progress.start()

        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()

    def _run_analysis(self):
        try:
            result_lines = []
            audio_result = None
            text_result = None

            # Audio analysis
            if self.audio_path:
                features = self.audio_extractor.extract(self.audio_path)
                features_scaled = self.scaler.transform([features])
                audio_probs = self.audio_model.predict(features_scaled)[0]
                audio_idx = np.argmax(audio_probs)
                audio_emotion = self.audio_emotion_names.get(audio_idx, str(audio_idx))
                audio_confidence = audio_probs[audio_idx]

                audio_result = {
                    'emotion': audio_emotion,
                    'confidence': audio_confidence,
                    'probs': audio_probs
                }
                result_lines.append("--- Audio Analysis ---")
                result_lines.append(f"  Predicted: {audio_emotion} ({audio_confidence*100:.1f}%)")

            # Text analysis
            text = self.text_input.get("1.0", tk.END).strip()
            if text:
                text_probs = self.text_analyzer.analyze(text)
                top_emotion = max(text_probs, key=text_probs.get)
                top_score = text_probs[top_emotion]

                text_result = {
                    'emotion': top_emotion,
                    'confidence': top_score,
                    'probs': text_probs
                }
                result_lines.append("--- Text Analysis ---")
                result_lines.append(f"  Predicted: {top_emotion} ({top_score*100:.1f}%)")

                sorted_text = sorted(text_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                result_lines.append("  Top text emotions:")
                for em, sc in sorted_text:
                    result_lines.append(f"    {em}: {sc*100:.1f}%")

            # Fusion & final result
            if audio_result and text_result:
                fused = self._fuse(audio_result['probs'], text_result['probs'])
                final_emotion = max(fused, key=fused.get)
                final_conf = fused[final_emotion]
                result_lines.insert(0, f"🎭 Final Emotion (fused): {final_emotion.upper()} ({final_conf*100:.1f}%)\n")
            elif audio_result:
                result_lines.insert(0, f"🎭 Final Emotion (audio only): {audio_result['emotion'].upper()} ({audio_result['confidence']*100:.1f}%)\n")
            elif text_result:
                result_lines.insert(0, f"🎭 Final Emotion (text only): {text_result['emotion'].upper()} ({text_result['confidence']*100:.1f}%)\n")

            self.root.after(0, self._update_results, "\n".join(result_lines))

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Analysis Error", str(e))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
            # Delete temporary recording file if it exists
            if self.is_temp_recording and self.audio_path and os.path.exists(self.audio_path):
                try:
                    os.unlink(self.audio_path)
                    self.audio_path = None
                    self.is_temp_recording = False
                    self.audio_label.config(text="No file selected", fg="gray")
                except Exception as e:
                    print(f"Could not delete temp file: {e}")

    def _fuse(self, audio_probs, text_probs_dict):
        fused = {}
        for text_emotion, text_score in text_probs_dict.items():
            text_emotion = str(text_emotion)
            audio_idx = self.text_to_audio_map.get(text_emotion, 0)
            audio_score = audio_probs[audio_idx] if audio_idx is not None else 0.0
            fused[text_emotion] = 0.6 * audio_score + 0.4 * text_score
        return fused

    def _update_results(self, text):
        self.result_label.config(text=text)


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 12, "bold"))
    app = EmotiDeskApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()