import os
import librosa
import numpy as np
from pathlib import Path

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """Load audio file and remove silence"""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed, sr

    def extract_features(self, y, sr):
        """Extract all relevant features for emotion detection"""
        features = []

        # MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.extend(mfcc_mean)

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroids))

        # Pitch (F0)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0 = f0[~np.isnan(f0)]
        features.append(np.mean(f0) if len(f0) > 0 else 0)

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        features.append(np.mean(rms))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))

        # Chroma features (first 6)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean[:6])

        # Mel-spectrogram mean
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.append(np.mean(mel))

        return np.array(features)

    def load_ravdess(self, path):
        """Load RAVDESS dataset (Actor_* folders)"""
        X, y = [], []
        path = Path(path)
        # RAVDESS emotion mapping: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        emotion_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
        for file in path.rglob("*.wav"):
            parts = file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                if emotion_code in emotion_map:
                    label = emotion_map[emotion_code]
                    try:
                        audio, sr = self.load_audio(file)
                        feat = self.extract_features(audio, sr)
                        X.append(feat)
                        y.append(label)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
        return np.array(X), np.array(y)

    def load_tess(self, path):
        """Load TESS dataset (OAF_* / YAF_* folders)"""
        X, y = [], []
        path = Path(path)
        emotion_folder_map = {
            'angry': 4,
            'disgust': 5,
            'fear': 6,
            'happy': 2,
            'neutral': 0,
            'ps': 3,      # TESS uses 'ps' for sad
            'sad': 3,
            'surprise': 7
        }
        for emotion_folder in path.iterdir():
            if emotion_folder.is_dir():
                folder_lower = emotion_folder.name.lower()
                matched_label = None
                for key, label in emotion_folder_map.items():
                    if key in folder_lower:
                        matched_label = label
                        break
                if matched_label is not None:
                    for file in emotion_folder.glob("*.wav"):
                        try:
                            audio, sr = self.load_audio(file)
                            feat = self.extract_features(audio, sr)
                            X.append(feat)
                            y.append(matched_label)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
        return np.array(X), np.array(y)

    def load_cremad(self, path):
        """Load CREMA-D dataset (AudioWAV folder)"""
        X, y = [], []
        path = Path(path) / "AudioWAV"
        emotion_map = {
            'ANG': 4,
            'DIS': 5,
            'FEA': 6,
            'HAP': 2,
            'NEU': 0,
            'SAD': 3,
            'SUR': 7
        }
        for file in path.glob("*.wav"):
            parts = file.stem.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = emotion_map[emotion_code]
                    try:
                        audio, sr = self.load_audio(file)
                        feat = self.extract_features(audio, sr)
                        X.append(feat)
                        y.append(label)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
        return np.array(X), np.array(y)

    def load_all_datasets(self, ravdess_path, tess_path, cremad_path):
        """Load and combine all three datasets"""
        X_list, y_list = [], []

        if ravdess_path and os.path.exists(ravdess_path):
            Xr, yr = self.load_ravdess(ravdess_path)
            if len(Xr) > 0:
                X_list.append(Xr); y_list.append(yr)
                print(f"RAVDESS: {len(Xr)} samples")

        if tess_path and os.path.exists(tess_path):
            Xt, yt = self.load_tess(tess_path)
            if len(Xt) > 0:
                X_list.append(Xt); y_list.append(yt)
                print(f"TESS: {len(Xt)} samples")

        if cremad_path and os.path.exists(cremad_path):
            Xc, yc = self.load_cremad(cremad_path)
            if len(Xc) > 0:
                X_list.append(Xc); y_list.append(yc)
                print(f"CREMA-D: {len(Xc)} samples")

        if not X_list:
            raise ValueError("No data loaded. Check dataset paths.")

        X_combined = np.vstack(X_list)
        y_combined = np.concatenate(y_list)
        return X_combined, y_combined