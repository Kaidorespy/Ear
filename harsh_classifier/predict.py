"""
Predict harsh vs clean vocals on audio files.

Usage:
    python predict.py <audio_file> [--start SEC] [--duration SEC]
    python predict.py <audio_file> --scan  # scan entire song in chunks
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import joblib
import librosa

from features import extract_features_from_array

# Load model
MODEL_PATH = Path(__file__).parent / "models" / "harsh_classifier.joblib"


def predict_segment(y: np.ndarray, sr: int, clf) -> dict:
    """Predict on a single audio segment."""
    features = extract_features_from_array(y, sr)
    proba = clf.predict_proba([features])[0]
    pred = clf.predict([features])[0]

    return {
        "prediction": "harsh" if pred == 1 else "clean",
        "confidence": max(proba),
        "clean_prob": proba[0],
        "harsh_prob": proba[1],
    }


def predict_file(audio_path: str, start: float = None, duration: float = 5.0) -> dict:
    """Predict on a single file/segment."""
    clf = joblib.load(MODEL_PATH)

    if start is not None:
        y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=duration)
    else:
        # Default: 30 seconds in, 5 second clip
        y_full, sr = librosa.load(audio_path, sr=22050)
        total_dur = len(y_full) / sr

        if total_dur < 35:
            start = total_dur / 4
        else:
            start = 30

        y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=duration)

    return predict_segment(y, sr, clf)


def scan_file(audio_path: str, chunk_duration: float = 3.0, overlap: float = 0.5) -> list:
    """Scan entire file and return predictions for each chunk."""
    clf = joblib.load(MODEL_PATH)

    y, sr = librosa.load(audio_path, sr=22050)
    total_duration = len(y) / sr

    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))

    results = []
    pos = 0

    while pos + chunk_samples <= len(y):
        chunk = y[pos:pos + chunk_samples]
        time_start = pos / sr

        # Skip quiet sections
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 0.01:
            pos += hop_samples
            continue

        try:
            result = predict_segment(chunk, sr, clf)
            result["time"] = f"{time_start:.1f}s"
            result["time_start"] = time_start
            results.append(result)
        except Exception as e:
            pass

        pos += hop_samples

    return results


def summarize_scan(results: list) -> dict:
    """Summarize a full scan."""
    if not results:
        return {"error": "No valid segments found"}

    harsh_count = sum(1 for r in results if r["prediction"] == "harsh")
    clean_count = len(results) - harsh_count

    avg_harsh_prob = np.mean([r["harsh_prob"] for r in results])
    max_harsh_prob = max(r["harsh_prob"] for r in results)

    # Find harsh moments
    harsh_moments = [r for r in results if r["prediction"] == "harsh"]

    return {
        "total_segments": len(results),
        "harsh_segments": harsh_count,
        "clean_segments": clean_count,
        "harsh_ratio": harsh_count / len(results),
        "avg_harsh_probability": avg_harsh_prob,
        "max_harsh_probability": max_harsh_prob,
        "overall": "harsh" if harsh_count > clean_count else "clean",
        "harsh_moments": [(r["time"], r["harsh_prob"]) for r in harsh_moments[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict harsh vs clean vocals")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--scan", action="store_true", help="Scan entire file")

    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"File not found: {args.audio}")
        sys.exit(1)

    print(f"File: {Path(args.audio).name}")
    print()

    if args.scan:
        print("Scanning entire file...")
        results = scan_file(args.audio)
        summary = summarize_scan(results)

        print(f"Segments analyzed: {summary['total_segments']}")
        print(f"Harsh segments: {summary['harsh_segments']} ({summary['harsh_ratio']*100:.1f}%)")
        print(f"Clean segments: {summary['clean_segments']}")
        print(f"Average harsh probability: {summary['avg_harsh_probability']:.3f}")
        print(f"Max harsh probability: {summary['max_harsh_probability']:.3f}")
        print(f"\nOverall: {summary['overall'].upper()}")

        if summary['harsh_moments']:
            print(f"\nHarsh moments detected at:")
            for time, prob in summary['harsh_moments']:
                print(f"  {time} (prob: {prob:.3f})")
    else:
        result = predict_file(args.audio, args.start, args.duration)
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Clean probability: {result['clean_prob']:.3f}")
        print(f"Harsh probability: {result['harsh_prob']:.3f}")


if __name__ == "__main__":
    main()
