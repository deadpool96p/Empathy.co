import numpy as np
import librosa
import os
from pathlib import Path
import pandas as pd

class UnifiedEmotionMapper:
    def __init__(self):
        self.neutral = 0
        self.calm = 1
        self.happy = 2
        self.sad = 3
        self.angry = 4
        self.fearful = 5
        self.disgust = 6
        self.surprised = 7

    def ravdess_map(self, code):
        mapping = {1: self.neutral, 2: self.calm, 3: self.happy, 4: self.sad,
                   5: self.angry, 6: self.fearful, 7: self.disgust, 8: self.surprised}
        return mapping.get(code)

    def tess_map(self, folder_name):
        mapping = {
            'neutral': self.neutral,
            'happy': self.happy,
            'sad': self.sad,
            'angry': self.angry,
            'fear': self.fearful,
            'disgust': self.disgust,
            'surprise': self.surprised,
            'ps': self.sad,
        }
        for key, val in mapping.items():
            if key in folder_name.lower():
                return val
        return None

    def cremad_map(self, emotion_code):
        mapping = {
            'NEU': self.neutral,
            'HAP': self.happy,
            'SAD': self.sad,
            'ANG': self.angry,
            'FEA': self.fearful,
            'DIS': self.disgust,
            'SUR': self.surprised,
        }
        return mapping.get(emotion_code)

    def iemocap_map(self, emotion_label):
        mapping = {
            'neu': self.neutral,
            'hap': self.happy,
            'sad': self.sad,
            'ang': self.angry,
            'fea': self.fearful,
            'dis': self.disgust,
            'sur': self.surprised,
            'exc': self.happy,
            'fru': self.angry,
        }
        return mapping.get(emotion_label.lower())

    def emodb_map(self, filename):
        if len(filename) < 5:
            return None
        code = filename[4]
        mapping = {
            'W': self.angry,
            'L': self.calm,
            'E': self.disgust,
            'A': self.fearful,
            'F': self.happy,
            'T': self.sad,
            'N': self.neutral,
        }
        return mapping.get(code.upper())

    def meld_map(self, emotion_name):
        mapping = {
            'neutral': self.neutral,
            'joy': self.happy,
            'sadness': self.sad,
            'anger': self.angry,
            'fear': self.fearful,
            'disgust': self.disgust,
            'surprise': self.surprised,
        }
        return mapping.get(emotion_name.lower())


class EnhancedFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self, audio_path, augment=False, augmenter=None):
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        if augment and augmenter:
            y = augmenter.apply(y)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        features = []
        # MFCCs (13) + deltas
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        features.extend(mfcc_delta_mean)

        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))

        spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
        features.append(np.mean(spec_rolloff))

        spec_bw = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]
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
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)

        # Mel-spectrogram stats
        mel = librosa.feature.melspectrogram(y=y_trimmed, sr=sr)
        features.append(np.mean(mel))
        features.append(np.std(mel))

        return np.array(features)