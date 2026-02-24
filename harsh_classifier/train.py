"""
Train the harsh vocal classifier.

Usage:
    python train.py

Expects clips in:
    data/clean/*.wav
    data/harsh/*.wav
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from features import extract_features, FEATURE_NAMES

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "models"


def load_dataset():
    """Load all clips and extract features."""
    X = []  # Features
    y = []  # Labels (0 = clean, 1 = harsh)
    files = []  # For debugging

    # Load clean clips
    clean_dir = DATA_DIR / "clean"
    if clean_dir.exists():
        for audio_file in clean_dir.glob("*.wav"):
            try:
                features = extract_features(str(audio_file))
                X.append(features)
                y.append(0)  # clean = 0
                files.append(audio_file.name)
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")

    # Load harsh clips
    harsh_dir = DATA_DIR / "harsh"
    if harsh_dir.exists():
        for audio_file in harsh_dir.glob("*.wav"):
            try:
                features = extract_features(str(audio_file))
                X.append(features)
                y.append(1)  # harsh = 1
                files.append(audio_file.name)
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")

    return np.array(X), np.array(y), files


def train_model(X, y):
    """Train Random Forest classifier."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Clean/Harsh ratio: {sum(y_train==0)}/{sum(y_train==1)}")

    # Train
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    print("\n=== Test Set Results ===")
    print(classification_report(y_test, y_pred, target_names=['clean', 'harsh']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    print("\n=== Cross-Validation ===")
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Feature importance
    print("\n=== Top 10 Most Important Features ===")
    importances = list(zip(FEATURE_NAMES, clf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")

    return clf


def save_model(clf, name="harsh_classifier"):
    """Save trained model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")
    return model_path


def main():
    print("Loading dataset...")
    X, y, files = load_dataset()

    if len(X) == 0:
        print("No data found! Add clips to data/clean/ and data/harsh/")
        sys.exit(1)

    print(f"Loaded {len(X)} clips ({sum(y==0)} clean, {sum(y==1)} harsh)")

    if sum(y == 0) < 5 or sum(y == 1) < 5:
        print("Warning: Need at least 5 clips per class for reliable training")

    print("\nTraining model...")
    clf = train_model(X, y)

    save_model(clf)

    print("\nDone! Model ready for use.")


if __name__ == "__main__":
    main()
