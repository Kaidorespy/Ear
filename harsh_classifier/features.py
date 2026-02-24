"""
Feature extraction for harsh vocal classifier.
Extracts MFCCs + spectral features from audio clips.
"""

import numpy as np
import librosa
from pathlib import Path


def extract_features(audio_path: str, sr: int = 22050, duration: float = None) -> np.ndarray:
    """Extract features from an audio file.

    Returns a feature vector of ~40 values:
    - 13 MFCCs (mean + std = 26)
    - Spectral centroid (mean, std)
    - Spectral flatness (mean, std)
    - Spectral rolloff (mean, std)
    - ZCR (mean, std)
    - RMS energy (mean, std)
    - Spectral bandwidth (mean, std)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)

    if len(y) < sr * 0.5:  # Less than 0.5 seconds
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")

    features = []

    # MFCCs - the gold standard for voice
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))  # 13 means
    features.extend(np.std(mfccs, axis=1))   # 13 stds

    # Spectral centroid - brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # Spectral flatness - noise vs tonal
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    # Spectral rolloff - frequency below which 85% of energy
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # Zero crossing rate - texture
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # RMS energy - loudness
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Spectral bandwidth - width of frequencies
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    return np.array(features, dtype=np.float32)


def extract_features_from_array(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract features from numpy array (for real-time use)."""
    if len(y) < sr * 0.5:
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")

    features = []

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
    'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean', 'mfcc_10_mean',
    'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_13_mean',
    'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std',
    'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_std', 'mfcc_9_std', 'mfcc_10_std',
    'mfcc_11_std', 'mfcc_12_std', 'mfcc_13_std',
    'centroid_mean', 'centroid_std',
    'flatness_mean', 'flatness_std',
    'rolloff_mean', 'rolloff_std',
    'zcr_mean', 'zcr_std',
    'rms_mean', 'rms_std',
    'bandwidth_mean', 'bandwidth_std',
]


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        feats = extract_features(sys.argv[1])
        print(f"Extracted {len(feats)} features")
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"  {name}: {val:.4f}")
